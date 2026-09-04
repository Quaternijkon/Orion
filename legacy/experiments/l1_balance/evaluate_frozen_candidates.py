#!/usr/bin/env python3
"""Evaluate immutable upper-only L1 owners in a separate L0-aware process."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance.l1_mass_partitioner import (  # noqa: E402
    MASS_SOURCE,
    MASS_SUPPORTED_ALGORITHMS,
    MASS_TRANSFORM,
    RAW_MASS_SOURCE,
    RAW_MASS_TRANSFORM,
    REGULARIZED_MASS_SOURCE,
    REGULARIZED_MASS_TRANSFORM,
    UPPER_NAVIGATION_TOP_K,
)
from experiments.l1_balance.run_offline_screen import (  # noqa: E402
    canonical_sha256,
    checked_binary_matrix,
    load_upper,
    local_hits,
    physical_membership,
    query_metrics,
    sha256_path,
    topology_metrics,
)


ATTACHMENT_SEARCH_EF = 100
PROTOCOL_PATH = REPO_ROOT / "experiments/l1_balance/PROTOCOL.md"
V2_AMENDMENT_PATH = (
    REPO_ROOT / "experiments/l1_balance/RAW_REGULARIZED_V2_AMENDMENT.json"
)
V2_AMENDMENT_ID = "post-exploratory-raw-regularized-v2-confirmation"
V2_METHOD = "mass-balanced-kmeans"

MASS_ESTIMATOR_CONTRACTS = {
    MASS_SOURCE: {
        "mass_mode": "self-debiased-floor1",
        "transform": MASS_TRANSFORM,
        "estimator_version": 1,
        "eligible_as_finalist": False,
    },
    RAW_MASS_SOURCE: {
        "mass_mode": "raw",
        "transform": RAW_MASS_TRANSFORM,
        "estimator_version": 1,
        "eligible_as_finalist": False,
    },
    REGULARIZED_MASS_SOURCE: {
        "mass_mode": "raw-regularized-v2",
        "transform": REGULARIZED_MASS_TRANSFORM,
        "estimator_version": 2,
        "eligible_as_finalist": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--attachments", required=True)
    parser.add_argument("--attachments-manifest", required=True)
    parser.add_argument("--row-count", type=int, required=True)
    parser.add_argument("--attachment-k", type=int, default=10)
    parser.add_argument("--query-hits")
    parser.add_argument("--query-hits-manifest")
    parser.add_argument("--ground-truth")
    parser.add_argument("--ground-truth-width", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def mismatches(
    actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }


def canonical_close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            canonical_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            canonical_close(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))
    return left == right


def validate_v2_protocol_amendment(value: Any) -> dict[str, Any]:
    """Validate the transparent post-exploratory promotion record."""

    if not isinstance(value, dict):
        raise ValueError("raw-regularized-v2 lacks its protocol amendment")
    expected = {
        "format_version": 1,
        "id": V2_AMENDMENT_ID,
        "status": "post_exploratory_promotion_not_preregistered",
        "mass_mode": "raw-regularized-v2",
        "source": REGULARIZED_MASS_SOURCE,
        "transform": REGULARIZED_MASS_TRANSFORM,
        "estimator_version": 2,
        "method": V2_METHOD,
        "topology_gates_unchanged": True,
        "old_raw_owner_byte_parity_required": True,
        "online_qps_only_confirmation": True,
    }
    errors = mismatches(value, expected)
    if errors:
        raise ValueError(f"v2 protocol amendment mismatch: {errors}")
    amendment_path = Path(value.get("amendment_path", "")).resolve()
    protocol_path = Path(value.get("protocol_path", "")).resolve()
    if amendment_path != V2_AMENDMENT_PATH.resolve():
        raise ValueError("v2 amendment path differs from the frozen repository file")
    if protocol_path != PROTOCOL_PATH.resolve():
        raise ValueError("v2 protocol path differs from the frozen repository file")
    if sha256_path(amendment_path) != value.get("amendment_sha256"):
        raise ValueError("v2 protocol amendment bytes drifted")
    if sha256_path(protocol_path) != value.get("protocol_sha256"):
        raise ValueError("v2 protocol document bytes drifted")
    amendment_file = json.loads(amendment_path.read_text(encoding="utf-8"))
    errors = mismatches(amendment_file, expected)
    if errors:
        raise ValueError(f"v2 amendment file mismatch: {errors}")
    return value


def validate_owner_parity(
    parity: Any,
    *,
    owner_path: Path,
    owner_sha256: str,
    artifact_sha256: str,
    method: str,
) -> bool:
    """Re-open both frozen manifests and prove exact v2/raw owner equality."""

    if not isinstance(parity, dict):
        raise ValueError("raw-regularized-v2 candidate lacks old-raw owner parity")
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
    reference_manifest_path = Path(
        parity.get("reference_candidate_manifest_path", "")
    ).resolve()
    reference_manifest_sha256 = sha256_path(reference_manifest_path)
    if reference_manifest_sha256 != parity.get(
        "reference_candidate_manifest_sha256"
    ):
        raise ValueError("raw parity-reference candidate manifest drifted")
    reference_sidecar = reference_manifest_path.with_name(
        reference_manifest_path.name + ".sha256"
    )
    if (
        reference_sidecar.read_text(encoding="ascii").strip()
        != reference_manifest_sha256
    ):
        raise ValueError("raw parity-reference manifest sidecar mismatch")
    reference = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    if reference.get("stage") != "upper_only_candidates_frozen":
        raise ValueError("owner parity reference is not a phase-A manifest")
    if (reference.get("parameters") or {}).get("mass_mode") != "raw":
        raise ValueError("owner parity reference is not the raw ablation")
    if (reference.get("construction_inputs") or {}).get(
        "artifact_sha256"
    ) != artifact_sha256:
        raise ValueError("owner parity reference uses another source artifact")
    matches = [
        candidate
        for candidate in reference.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("method") == method
    ]
    if len(matches) != 1:
        raise ValueError("raw parity reference lacks exactly one matching method")
    raw_candidate = matches[0]
    if raw_candidate.get("navigation_mass_source") != RAW_MASS_SOURCE:
        raise ValueError("owner parity record is not backed by raw hit count")
    reference_owner_path = Path(raw_candidate.get("owner_path", "")).resolve()
    if str(reference_owner_path) != parity.get("reference_owner_path"):
        raise ValueError("owner parity path differs from the raw manifest")
    if raw_candidate.get("owner_sha256") != owner_sha256:
        raise ValueError("v2 owner SHA differs from the frozen raw owner SHA")
    if sha256_path(reference_owner_path) != owner_sha256:
        raise ValueError("frozen raw owner bytes drifted")
    if sha256_path(owner_path) != owner_sha256:
        raise ValueError("v2 owner bytes drifted")
    if owner_path.read_bytes() != reference_owner_path.read_bytes():
        raise ValueError("v2 owner is not byte-for-byte identical to raw owner")
    return True


def semantic_mass_sha256(
    source: str,
    transform: str,
    values: np.ndarray,
    top_k: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(source.encode("ascii"))
    digest.update(b"\0")
    digest.update(transform.encode("ascii"))
    digest.update(struct.pack("<QQ", len(values), top_k))
    for value in values:
        digest.update(struct.pack("<Q", int(value)))
    return digest.hexdigest()


def assignment_bytes(point_id: int, shards: np.ndarray) -> bytes:
    return (
        f'{{"id":{point_id},"shards":['
        + ",".join(str(int(shard)) for shard in shards)
        + "]}\n"
    ).encode("ascii")


def membership_assignment_sha256(membership: np.ndarray) -> str:
    """Hash the exact JSONL bytes consumed by Orion's numeric importer."""

    digest = hashlib.sha256()
    for point_id, row in enumerate(membership):
        digest.update(assignment_bytes(point_id, np.flatnonzero(row)))
    return digest.hexdigest()


def validate_gate_record(
    candidate: dict[str, Any], gate_field: str, all_pass_field: str
) -> None:
    gates = candidate.get(gate_field)
    if not isinstance(gates, dict) or not gates:
        raise ValueError("candidate lacks preregistered topology gates")
    passes = []
    for metric, gate in gates.items():
        observed = float(candidate["topology_metrics"][metric])
        if not np.isclose(observed, float(gate["observed"]), rtol=1e-12, atol=1e-12):
            raise ValueError(f"candidate gate {metric} observed value drifted")
        operator = gate["operator"]
        threshold = float(gate["threshold"])
        passed = observed <= threshold if operator == "<=" else observed >= threshold
        if operator not in {"<=", ">="}:
            raise ValueError(f"candidate gate {metric} has invalid operator")
        if bool(gate["pass"]) != bool(passed):
            raise ValueError(f"candidate gate {metric} pass bit is inconsistent")
        passes.append(bool(passed))
    if bool(candidate.get(all_pass_field)) != all(passes):
        raise ValueError(f"candidate {all_pass_field} is inconsistent")


def duplicate_tie_proof(
    navigation_local: np.ndarray,
    labels: np.ndarray,
    vectors: np.ndarray,
) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    for row in range(len(navigation_local)):
        positions = np.flatnonzero(navigation_local[row] == row)
        if len(positions) != 1:
            raise ValueError(f"upper self query row {row} has {len(positions)} self hits")
        self_rank = int(positions[0])
        if self_rank == 0:
            continue
        query_bits = np.ascontiguousarray(vectors[row]).tobytes()
        preceding = navigation_local[row, :self_rank]
        for hit in preceding:
            if np.ascontiguousarray(vectors[int(hit)]).tobytes() != query_bits:
                raise ValueError(
                    f"upper self query row {row} has a non-duplicate hit before self"
                )
        records.append(
            {
                "row": row,
                "query_label": int(labels[row]),
                "self_rank": self_rank,
                "preceding_labels": [int(labels[int(hit)]) for hit in preceding],
                "vector_bits_sha256": hashlib.sha256(query_bits).hexdigest(),
            }
        )
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return records, hashlib.sha256(encoded).hexdigest()


def query_gate_results(
    observed: dict[str, float | int],
    reference: dict[str, float | int],
    query_count: int,
) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}

    def add(metric: str, operator: str, threshold: float) -> None:
        value = float(observed[metric])
        passed = value <= threshold if operator == "<=" else value >= threshold
        gates[metric] = {
            "observed": value,
            "reference": float(reference[metric]),
            "operator": operator,
            "threshold": threshold,
            "pass": bool(passed),
            "category": "search_induced",
        }

    for metric in (
        "query_owner_transitions_mean",
        "routed_shards_mean",
        "route_entry_points_mean",
        "route_ef_sum_mean",
    ):
        add(metric, "<=", float(reference[metric]) * 1.05)
    add(
        "gt_routing_coverage_mean",
        ">=",
        float(reference["gt_routing_coverage_mean"]) - 0.002,
    )
    add(
        "gt_queries_full_coverage_fraction",
        ">=",
        float(reference["gt_queries_full_coverage_fraction"]) - 0.02,
    )
    zero_value = int(observed["gt_queries_zero_coverage"])
    zero_reference = int(reference["gt_queries_zero_coverage"])
    zero_threshold = min(zero_reference + 2, int(math.floor(query_count * 0.001)))
    gates["gt_queries_zero_coverage"] = {
        "observed": zero_value,
        "reference": zero_reference,
        "operator": "<=",
        "threshold": zero_threshold,
        "pass": zero_value <= zero_threshold,
        "category": "search_induced",
    }
    return gates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_query_evaluation(
    args: argparse.Namespace,
    artifact_path: Path,
    labels: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
    if bool(args.query_hits) != bool(args.query_hits_manifest):
        raise ValueError("query-hits and query-hits-manifest must be provided together")
    if not args.query_hits:
        if args.ground_truth:
            raise ValueError("ground-truth requires query-hits")
        return None, None, None

    query_hits_path = Path(args.query_hits).expanduser().resolve()
    query_manifest_path = Path(args.query_hits_manifest).expanduser().resolve()
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    query_rows = int(query_manifest["row_count"])
    query_k = int(query_manifest["top_k"])
    query_hits = checked_binary_matrix(
        query_hits_path, rows=query_rows, width=query_k, dtype="<u8"
    )
    expected = {
        "artifact_sha256": sha256_path(artifact_path),
        "hits_sha256": sha256_path(query_hits_path),
    }
    errors = mismatches(query_manifest, expected)
    if errors:
        raise ValueError(f"query-hit manifest mismatch: {errors}")
    query_local = local_hits(query_hits, labels, query_rows)
    ground_truth = None
    if args.ground_truth:
        ground_truth = checked_binary_matrix(
            Path(args.ground_truth).expanduser().resolve(),
            rows=query_rows,
            width=args.ground_truth_width,
            dtype="<u4",
        )
    return query_local, ground_truth, query_manifest


def main() -> None:
    args = parse_args()
    if args.row_count <= 0 or args.attachment_k != UPPER_NAVIGATION_TOP_K:
        raise ValueError("row-count must be positive and attachment-k must equal 10")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    # Validate every construction byte before opening any post-freeze L0 input.
    candidate_manifest_path = Path(
        args.candidate_manifest
    ).expanduser().resolve()
    candidate_manifest_sha256 = sha256_path(candidate_manifest_path)
    checksum_path = candidate_manifest_path.with_name(
        candidate_manifest_path.name + ".sha256"
    )
    if checksum_path.read_text(encoding="ascii").strip() != candidate_manifest_sha256:
        raise ValueError("frozen candidate manifest checksum sidecar mismatch")
    frozen = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if frozen.get("stage") != "upper_only_candidates_frozen":
        raise ValueError("input is not a frozen upper-only candidate manifest")
    contract = frozen.get("contract") or {}
    required_contract = {
        "partition_input_scope": "production_upper_graph_and_upper_navigation_mass_only",
        "construction_process_accepts_l0_inputs": False,
        "construction_process_accepts_query_ground_truth": False,
        "partitioner_reads_full_attachments": False,
        "partitioner_reads_l0_load": False,
        "partitioner_reads_multi_assignment_state": False,
        "separate_evaluator_process_required": True,
        "l0_repair_or_flow": False,
    }
    errors = mismatches(contract, required_contract)
    if errors:
        raise ValueError(f"frozen candidate architecture mismatch: {errors}")

    construction = frozen["construction_inputs"]
    artifact_path = Path(construction["artifact"]).resolve()
    if sha256_path(artifact_path) != construction["artifact_sha256"]:
        raise ValueError("source artifact checksum drifted")
    partitioner_path = Path(construction["partitioner_source"]).resolve()
    if sha256_path(partitioner_path) != construction["partitioner_source_sha256"]:
        raise ValueError("partitioner source checksum drifted after candidate freeze")
    (
        artifact,
        _adjacency,
        vectors,
        labels,
        _entry_point,
        edge_left,
        edge_right,
        navigator_sha256,
    ) = load_upper(artifact_path)
    if navigator_sha256 != construction["navigator_sha256"]:
        raise ValueError("production navigator checksum drifted")
    if canonical_sha256(artifact["upper_graph"]) != construction["upper_graph_sha256"]:
        raise ValueError("production upper-graph checksum drifted")
    ordered_labels_sha256 = hashlib.sha256(
        labels.astype("<u8", copy=False).tobytes(order="C")
    ).hexdigest()
    ordered_vectors_sha256 = hashlib.sha256(
        vectors.astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()
    if ordered_labels_sha256 != construction["ordered_labels_sha256"]:
        raise ValueError("ordered upper-label checksum drifted")
    if ordered_vectors_sha256 != construction["ordered_vectors_sha256"]:
        raise ValueError("ordered upper-vector checksum drifted")
    if int(artifact["logical_point_count"]) != args.row_count:
        raise ValueError("row-count differs from frozen artifact")
    num_partitions = int(frozen["parameters"]["num_partitions"])
    if int(artifact["shard_count"]) != num_partitions:
        raise ValueError("partition count differs from frozen artifact")

    mass_info = frozen["mass_estimator"]
    if not isinstance(mass_info, dict):
        raise ValueError("frozen candidate lacks navigation-mass metadata")
    mass_source = mass_info.get("source")
    estimator_contract = MASS_ESTIMATOR_CONTRACTS.get(mass_source)
    if estimator_contract is None:
        raise ValueError("unknown navigation-mass source")
    mass_mode = estimator_contract["mass_mode"]
    errors = mismatches(
        mass_info,
        {
            "format_version": 1,
            "mass_mode": mass_mode,
            "transform": estimator_contract["transform"],
            "estimator_version": estimator_contract["estimator_version"],
            "eligible_as_finalist": estimator_contract["eligible_as_finalist"],
            "rank_weighting": False,
        },
    )
    if errors:
        raise ValueError(f"navigation-mass estimator contract mismatch: {errors}")
    parameters = frozen.get("parameters") or {}
    if parameters.get("mass_mode") != mass_mode:
        raise ValueError("frozen mass mode differs from estimator source")
    is_v2_confirmation = mass_source == REGULARIZED_MASS_SOURCE
    if is_v2_confirmation:
        if parameters.get("methods") != [V2_METHOD]:
            raise ValueError("raw-regularized-v2 must freeze only mass-balanced-kmeans")
        protocol_amendment = validate_v2_protocol_amendment(
            frozen.get("protocol_amendment")
        )
    else:
        if frozen.get("protocol_amendment") is not None:
            raise ValueError("v1 estimator or raw ablation cannot claim the v2 amendment")
        protocol_amendment = None
    mass_manifest_path = Path(mass_info["manifest_path"]).resolve()
    if sha256_path(mass_manifest_path) != mass_info["manifest_sha256"]:
        raise ValueError("navigation-mass manifest checksum drifted")
    mass_manifest = json.loads(mass_manifest_path.read_text(encoding="utf-8"))
    expected_mass_manifest = {
        "mass_mode": mass_mode,
        "source": mass_info["source"],
        "transform": mass_info["transform"],
        "estimator_version": mass_info["estimator_version"],
        "semantic_sha256": mass_info["sha256"],
        "source_artifact_sha256": construction["artifact_sha256"],
        "navigator_sha256": construction["navigator_sha256"],
        "upper_graph_sha256": construction["upper_graph_sha256"],
        "ordered_labels_sha256": construction["ordered_labels_sha256"],
        "ordered_vectors_sha256": construction["ordered_vectors_sha256"],
        "upper_navigation_hits_sha256": construction["upper_navigation_hits_sha256"],
        "upper_navigation_manifest_sha256": construction[
            "upper_navigation_manifest_sha256"
        ],
        "upper_node_count": len(labels),
        "dimension": int(construction["dimension"]),
        "top_k": UPPER_NAVIGATION_TOP_K,
        "search_ef": int(construction["upper_navigation_search_ef"]),
        "partitioner_source_sha256": construction["partitioner_source_sha256"],
        "protocol_amendment": protocol_amendment,
    }
    errors = mismatches(mass_manifest, expected_mass_manifest)
    if errors:
        raise ValueError(f"navigation-mass manifest mismatch: {errors}")
    mass_values_path = Path(mass_manifest["values_file"]).resolve()
    if sha256_path(mass_values_path) != mass_manifest["values_sha256"]:
        raise ValueError("navigation-mass value bytes drifted")
    mass_values = checked_binary_matrix(
        mass_values_path, rows=len(labels), width=1, dtype="<u8"
    ).reshape(-1)
    navigation_hits_path = Path(construction["upper_navigation_hits"]).resolve()
    if sha256_path(navigation_hits_path) != construction["upper_navigation_hits_sha256"]:
        raise ValueError("upper-navigation hit bytes drifted")
    navigation_manifest_path = Path(
        construction["upper_navigation_manifest"]
    ).resolve()
    if sha256_path(navigation_manifest_path) != construction[
        "upper_navigation_manifest_sha256"
    ]:
        raise ValueError("upper-navigation export manifest drifted")
    navigation_manifest = json.loads(
        navigation_manifest_path.read_text(encoding="utf-8")
    )
    expected_navigation = {
        "artifact_sha256": construction["artifact_sha256"],
        "upper_graph_present": True,
        "vectors_sha256": construction["ordered_vectors_sha256"],
        "row_count": len(labels),
        "dimension": int(construction["dimension"]),
        "top_k": UPPER_NAVIGATION_TOP_K,
        "search_ef": int(construction["upper_navigation_search_ef"]),
        "hits_sha256": construction["upper_navigation_hits_sha256"],
    }
    errors = mismatches(navigation_manifest, expected_navigation)
    if errors:
        raise ValueError(f"upper-navigation export mismatch: {errors}")
    navigation_hits = checked_binary_matrix(
        navigation_hits_path,
        rows=len(labels),
        width=UPPER_NAVIGATION_TOP_K,
        dtype="<u8",
    )
    navigation_local = local_hits(navigation_hits, labels, len(labels))
    tie_records, tie_proof_sha256 = duplicate_tie_proof(
        navigation_local, labels, vectors
    )
    self_first_count = int(
        np.count_nonzero(
            navigation_local[:, 0] == np.arange(len(labels), dtype=np.int32)
        )
    )
    expected_self_proof = {
        "self_present_count": len(labels),
        "self_first_count": self_first_count,
        "duplicate_tie_exception_count": len(tie_records),
        "duplicate_tie_proof_sha256": tie_proof_sha256,
    }
    errors = mismatches(mass_info, expected_self_proof)
    errors.update(mismatches(mass_manifest, expected_self_proof))
    if errors:
        raise ValueError(f"upper self-hit proof mismatch: {errors}")
    raw_mass = np.bincount(
        np.asarray(navigation_local).reshape(-1), minlength=len(labels)
    ).astype(np.uint64)
    if mass_info["source"] == MASS_SOURCE:
        expected_mass_values = np.maximum(raw_mass - 1, 1)
    elif mass_info["source"] in {RAW_MASS_SOURCE, REGULARIZED_MASS_SOURCE}:
        expected_mass_values = raw_mass
    else:
        raise ValueError("unknown navigation-mass source")
    if not np.array_equal(mass_values, expected_mass_values):
        raise ValueError("navigation mass does not replay from upper self hits")
    if semantic_mass_sha256(
        mass_info["source"],
        mass_info["transform"],
        mass_values,
        UPPER_NAVIGATION_TOP_K,
    ) != mass_info["sha256"]:
        raise ValueError("navigation-mass semantic checksum mismatch")
    if int(mass_values.sum()) != int(mass_info["total_mass"]):
        raise ValueError("navigation-mass total drifted")

    frozen_candidates: list[tuple[dict[str, Any], np.ndarray, bool]] = []
    for candidate in frozen["candidates"]:
        if candidate["method"] not in MASS_SUPPORTED_ALGORITHMS:
            raise ValueError("candidate method is not a registered mass algorithm")
        expected_candidate = {
            "balance_contract": "navigation_mass_capacity",
            "source_artifact_sha256": construction["artifact_sha256"],
            "partitioner_source_sha256": construction["partitioner_source_sha256"],
            "navigation_mass_manifest_path": str(mass_manifest_path),
            "navigation_mass_manifest_sha256": mass_info["manifest_sha256"],
            "navigation_mass_source": mass_info["source"],
            "navigation_mass_transform": mass_info["transform"],
            "navigation_mass_estimator_version": mass_info["estimator_version"],
            "navigation_mass_sha256": mass_info["sha256"],
            "mass_mode": mass_mode,
            "protocol_amendment": protocol_amendment,
        }
        errors = mismatches(candidate, expected_candidate)
        if errors:
            raise ValueError(f"candidate binding mismatch: {errors}")
        owner_path = Path(candidate["owner_path"]).resolve()
        if sha256_path(owner_path) != candidate["owner_sha256"]:
            raise ValueError("candidate owner checksum drifted")
        owner = checked_binary_matrix(
            owner_path, rows=len(labels), width=1, dtype="<i4"
        ).reshape(-1)
        if np.any(owner < 0) or np.any(owner >= num_partitions):
            raise ValueError("candidate owner contains an invalid partition")
        if is_v2_confirmation:
            if candidate["method"] != V2_METHOD:
                raise ValueError("raw-regularized-v2 has an unexpected method")
            owner_parity_all_pass = validate_owner_parity(
                candidate.get("owner_parity"),
                owner_path=owner_path,
                owner_sha256=candidate["owner_sha256"],
                artifact_sha256=construction["artifact_sha256"],
                method=candidate["method"],
            )
        else:
            if candidate.get("owner_parity") is not None:
                raise ValueError("v1 estimator or raw ablation cannot claim owner parity")
            owner_parity_all_pass = False
        sizes = np.bincount(owner, minlength=num_partitions)
        estimated = np.bincount(
            owner,
            weights=mass_values.astype(np.float64),
            minlength=num_partitions,
        ).astype(np.int64)
        if sizes.tolist() != candidate["partition_sizes"]:
            raise ValueError("candidate partition sizes drifted")
        if estimated.tolist() != candidate["estimated_partition_masses"]:
            raise ValueError("candidate estimated masses drifted")
        if np.any(sizes == 0) or int(np.max(estimated)) > int(
            candidate["estimated_mass_limit"]
        ):
            raise ValueError("candidate violates its weighted capacity contract")
        metrics = topology_metrics(
            owner,
            edge_left,
            edge_right,
            len(owner),
            num_partitions,
        )
        if not canonical_close(metrics, candidate["topology_metrics"]):
            raise ValueError("candidate topology metrics drifted")
        validate_gate_record(
            candidate, "graph_topology_gates", "graph_topology_all_pass"
        )
        if bool(candidate["topology_all_pass"]) or bool(
            candidate["materialization_eligible"]
        ):
            raise ValueError("phase-A candidate was prematurely marked eligible")
        if not bool(candidate["eligibility_pending_post_freeze_query_gates"]):
            raise ValueError("phase-A candidate lacks pending query-gate state")
        frozen_candidates.append((candidate, owner, owner_parity_all_pass))

    # Post-freeze evaluation begins only after the construction audit above.
    attachments_path = Path(args.attachments).expanduser().resolve()
    attachments_manifest_path = Path(
        args.attachments_manifest
    ).expanduser().resolve()
    attachments_manifest = json.loads(
        attachments_manifest_path.read_text(encoding="utf-8")
    )
    expected_attachment = {
        "artifact_sha256": construction["artifact_sha256"],
        "row_count": args.row_count,
        "top_k": args.attachment_k,
        "search_ef": ATTACHMENT_SEARCH_EF,
        "hits_sha256": sha256_path(attachments_path),
    }
    errors = mismatches(attachments_manifest, expected_attachment)
    if errors:
        raise ValueError(f"attachment manifest mismatch: {errors}")
    attachments_sha256 = expected_attachment["hits_sha256"]
    attachment_hits = checked_binary_matrix(
        attachments_path,
        rows=args.row_count,
        width=args.attachment_k,
        dtype="<u8",
    )
    attachment_local = local_hits(attachment_hits, labels, args.row_count)
    query_local, ground_truth, query_manifest = load_query_evaluation(
        args, artifact_path, labels
    )
    source_memberships = [node["shard_membership"] for node in artifact["upper_nodes"]]
    if not all(isinstance(row, list) and len(row) == 1 for row in source_memberships):
        raise ValueError("artifact lacks the singleton source topology reference")
    source_owner = np.asarray(
        [int(row[0]) for row in source_memberships], dtype=np.int32
    )
    source_topology = topology_metrics(
        source_owner,
        edge_left,
        edge_right,
        len(labels),
        num_partitions,
    )
    if not canonical_close(
        source_topology,
        frozen["topology_gate_contract"]["reference_metrics"],
    ):
        raise ValueError("source topology reference drifted")
    source_membership, source_copy_count, source_loads = physical_membership(
        source_owner, attachment_local, num_partitions
    )
    source_unique, source_counts = np.unique(source_copy_count, return_counts=True)
    source_copy_histogram = {
        str(int(value)): int(count)
        for value, count in zip(source_unique.tolist(), source_counts.tolist())
    }
    source_query_metrics = None
    if query_local is not None:
        source_query_metrics = query_metrics(
            source_owner,
            query_local,
            source_membership[labels],
            source_membership,
            ground_truth,
            int(artifact["dynamic_ef_base"]),
            int(artifact["dynamic_ef_factor"]),
        )
    query_gate_inputs_complete = (
        source_query_metrics is not None and ground_truth is not None
    )
    identity_gates = {
        "source_artifact_sha256": True,
        "navigator_sha256": True,
        "upper_graph_sha256": True,
        "ordered_labels_sha256": True,
        "ordered_vectors_sha256": True,
        "upper_navigation_replay": True,
        "duplicate_tie_proof": True,
        "partitioner_source_sha256": True,
        "frozen_owner_sha256": True,
        "protocol_amendment_contract": True,
    }
    identity_all_pass = all(identity_gates.values())

    output_dir.mkdir(parents=True, exist_ok=False)
    evaluation_dir = output_dir / "candidate-evaluation"
    evaluation_dir.mkdir()
    rows: list[dict[str, Any]] = []
    copy_histograms: dict[str, dict[str, int]] = {}
    final_candidates: list[dict[str, Any]] = []
    owner_parity_results: list[bool] = []
    for candidate, owner, owner_parity_all_pass in frozen_candidates:
        owner_parity_results.append(owner_parity_all_pass)
        membership, copy_count, loads = physical_membership(
            owner, attachment_local, num_partitions
        )
        upper_membership = membership[labels]
        load_mean = float(np.mean(loads))
        estimated = np.asarray(
            candidate["estimated_partition_masses"], dtype=np.int64
        )
        estimated_mean = float(np.mean(estimated))
        actual_metrics: dict[str, Any] = {
            "physical_point_count": int(np.sum(loads)),
            "expansion_ratio": float(np.mean(copy_count)),
            "load_min": int(np.min(loads)),
            "load_max": int(np.max(loads)),
            "load_mean": load_mean,
            "load_cv": float(np.std(loads) / load_mean),
            "load_max_over_mean": float(np.max(loads) / load_mean),
            "load_min_over_mean": float(np.min(loads) / load_mean),
            "empty_shards": int(np.count_nonzero(loads == 0)),
        }
        if query_local is not None:
            actual_metrics.update(
                query_metrics(
                    owner,
                    query_local,
                    upper_membership,
                    membership,
                    ground_truth,
                    int(artifact["dynamic_ef_base"]),
                    int(artifact["dynamic_ef_factor"]),
                )
            )
        if query_gate_inputs_complete:
            query_gates = query_gate_results(
                actual_metrics,
                source_query_metrics,
                int(query_manifest["row_count"]),
            )
        else:
            query_gates = {}
        topology_gates = {
            **candidate["graph_topology_gates"],
            **query_gates,
        }
        topology_all_pass = (
            bool(candidate["graph_topology_all_pass"])
            and query_gate_inputs_complete
            and all(bool(gate["pass"]) for gate in query_gates.values())
        )
        candidate_identity_gates = dict(identity_gates)
        if is_v2_confirmation:
            candidate_identity_gates["old_raw_owner_byte_parity"] = (
                owner_parity_all_pass
            )
        candidate_identity_all_pass = all(candidate_identity_gates.values())
        materialization_eligible = (
            is_v2_confirmation
            and bool(mass_info["eligible_as_finalist"])
            and owner_parity_all_pass
            and candidate_identity_all_pass
            and topology_all_pass
        )
        unique, counts = np.unique(copy_count, return_counts=True)
        copy_histograms[candidate["method"]] = {
            str(int(value)): int(count)
            for value, count in zip(unique.tolist(), counts.tolist())
        }
        evaluation_path = evaluation_dir / f"{candidate['method']}.evaluation.npz"
        np.savez_compressed(
            evaluation_path,
            loads=loads,
            estimated_masses=estimated,
            upper_membership=np.packbits(upper_membership, axis=1),
            copy_count_hist_values=unique,
            copy_count_hist_counts=counts,
        )
        copy_histogram = copy_histograms[candidate["method"]]
        materialization_parity = {
            "canonical_format": "orion_numeric_import.assignments.jsonl-v1",
            "assignment_bytes_sha256": membership_assignment_sha256(membership),
            "logical_point_count": int(len(membership)),
            "physical_point_count": int(np.sum(loads)),
            "shard_loads": [int(value) for value in loads.tolist()],
            "copy_count_histogram": copy_histogram,
            "evaluation_npz_path": str(evaluation_path),
            "evaluation_npz_sha256": sha256_path(evaluation_path),
        }
        final_candidate = {
            **candidate,
            "source_artifact_sha256": construction["artifact_sha256"],
            "attachments_sha256": attachments_sha256,
            "upper_navigation_hits_sha256": construction[
                "upper_navigation_hits_sha256"
            ],
            "upper_navigation_manifest_sha256": construction[
                "upper_navigation_manifest_sha256"
            ],
            "actual_metrics": actual_metrics,
            "materialization_parity": materialization_parity,
            "identity_gates": candidate_identity_gates,
            "identity_all_pass": candidate_identity_all_pass,
            "owner_parity_all_pass": owner_parity_all_pass,
            "query_topology_gates": query_gates,
            "topology_gates": topology_gates,
            "topology_gate_inputs_complete": query_gate_inputs_complete,
            "topology_all_pass": topology_all_pass,
            "eligibility_pending_post_freeze_query_gates": False,
            "materialization_eligible": materialization_eligible,
        }
        final_candidates.append(final_candidate)
        row = {
            "method": candidate["method"],
            "materialization_eligible": materialization_eligible,
            "topology_all_pass": topology_all_pass,
            "owner_parity_all_pass": owner_parity_all_pass,
            "l1_min_size": min(candidate["partition_sizes"]),
            "l1_max_size": max(candidate["partition_sizes"]),
            "estimated_mass_min": int(np.min(estimated)),
            "estimated_mass_max": int(np.max(estimated)),
            "estimated_mass_mean": estimated_mean,
            "estimated_mass_cv": float(np.std(estimated) / estimated_mean),
            "estimated_mass_max_over_mean": float(np.max(estimated) / estimated_mean),
            "estimated_mass_limit": candidate["estimated_mass_limit"],
            "partition_seconds": candidate["partition_seconds"],
            "partition_edge_visits": candidate["partition_edge_visits"],
            **candidate["topology_metrics"],
            **actual_metrics,
        }
        rows.append(row)
        print(
            f"method={candidate['method']} "
            f"cut={row['upper_edge_cut_ratio']:.6f} "
            f"load_max_mean={row['load_max_over_mean']:.6f} "
            f"expansion={row['expansion_ratio']:.6f} "
            f"topology_all_pass={topology_all_pass} "
            f"eligible={materialization_eligible}",
            flush=True,
        )

    write_csv(output_dir / "summary.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    owner_parity_all_pass = (
        is_v2_confirmation
        and bool(owner_parity_results)
        and all(owner_parity_results)
    )
    screen_manifest = {
        "format_version": 1,
        "stage": "post_freeze_evaluation",
        "contract": {
            **required_contract,
            "evaluator_runs_in_separate_process": True,
            "owners_changed_after_freeze": False,
            "l0_assignment_after_owner_freeze": True,
            "l0_repair_or_flow": False,
            "multi_assignment": {
                "enabled": True,
                "min_max_vote": 2,
                "vote_delta": 0,
                "max_shards": 0,
            },
        },
        "frozen_candidate_manifest": str(candidate_manifest_path),
        "frozen_candidate_manifest_sha256": candidate_manifest_sha256,
        "mass_estimator": mass_info,
        "protocol_amendment": protocol_amendment,
        "topology_gate_contract": frozen["topology_gate_contract"],
        "search_induced_gate_contract": {
            "version": "l1-search-induced-preregistered-v1",
            "reference": "source_singleton_l1_owner_same_evaluation_inputs",
            "mean_route_metric_max_ratio": 1.05,
            "gt_coverage_mean_max_drop": 0.002,
            "gt_full_coverage_fraction_max_drop": 0.02,
            "gt_zero_coverage_max_increase": 2,
            "gt_zero_coverage_absolute_fraction_max": 0.001,
            "all_gates_required": True,
            "inputs_complete": query_gate_inputs_complete,
        },
        "identity_gates": identity_gates,
        "identity_all_pass": identity_all_pass,
        "owner_parity_all_pass": owner_parity_all_pass,
        "source_reference_evaluation": {
            "topology_metrics": source_topology,
            "query_metrics": source_query_metrics,
            "physical_point_count": int(np.sum(source_loads)),
            "expansion_ratio": float(np.mean(source_copy_count)),
            "shard_loads": [int(value) for value in source_loads.tolist()],
            "load_max_over_mean": float(
                np.max(source_loads) / np.mean(source_loads)
            ),
            "copy_count_histogram": source_copy_histogram,
        },
        "construction_inputs": construction,
        "post_freeze_evaluation_inputs": {
            "attachments": str(attachments_path),
            "attachments_sha256": attachments_sha256,
            "attachments_manifest": str(attachments_manifest_path),
            "attachments_manifest_sha256": sha256_path(attachments_manifest_path),
            "query_hits": str(Path(args.query_hits).resolve())
            if args.query_hits
            else None,
            "query_hits_sha256": sha256_path(Path(args.query_hits).resolve())
            if args.query_hits
            else None,
            "query_hits_manifest": str(Path(args.query_hits_manifest).resolve())
            if args.query_hits_manifest
            else None,
            "query_hits_manifest_sha256": sha256_path(
                Path(args.query_hits_manifest).resolve()
            )
            if args.query_hits_manifest
            else None,
            "query_row_count": int(query_manifest["row_count"])
            if query_manifest
            else None,
            "ground_truth": str(Path(args.ground_truth).resolve())
            if args.ground_truth
            else None,
            "ground_truth_sha256": sha256_path(Path(args.ground_truth).resolve())
            if args.ground_truth
            else None,
        },
        "candidates": final_candidates,
        "copy_count_histograms": copy_histograms,
        "outputs": {
            "summary_csv": "summary.csv",
            "summary_json": "summary.json",
            "candidate_evaluation_dir": "candidate-evaluation",
        },
    }
    screen_manifest_path = output_dir / "screen-manifest.json"
    screen_manifest_path.write_text(
        json.dumps(screen_manifest, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    screen_manifest_path.with_name(screen_manifest_path.name + ".sha256").write_text(
        sha256_path(screen_manifest_path) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
