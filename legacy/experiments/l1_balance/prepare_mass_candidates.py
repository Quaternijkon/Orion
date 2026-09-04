#!/usr/bin/env python3
"""Freeze upper-only navigation-mass L1 candidates, then exit.

This construction process has no command-line option or code path for full L0
attachments, observed shard loads, query ground truth, or multi-assignment.
It writes checksum-bound owner arrays and a candidate manifest.  A separate
process, :mod:`evaluate_frozen_candidates`, may later evaluate those immutable
owners against L0 data without rerunning a partitioner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.l1_balance.l1_mass_partitioner as mass_partitioner  # noqa: E402
from experiments.l1_balance.l1_mass_partitioner import (  # noqa: E402
    MASS_SOURCE,
    MASS_SUPPORTED_ALGORITHMS,
    RAW_MASS_SOURCE,
    REGULARIZED_MASS_SOURCE,
    UPPER_NAVIGATION_TOP_K,
    estimate_raw_upper_navigation_mass,
    estimate_regularized_upper_navigation_mass,
    estimate_upper_navigation_mass,
    partition_l1_by_navigation_mass,
)
from experiments.l1_balance.run_offline_screen import (  # noqa: E402
    canonical_sha256,
    checked_binary_matrix,
    load_upper,
    local_hits,
    sha256_path,
    topology_metrics,
)


UPPER_NAVIGATION_SEARCH_EF = 100
PROTOCOL_PATH = REPO_ROOT / "experiments/l1_balance/PROTOCOL.md"
V2_AMENDMENT_PATH = (
    REPO_ROOT / "experiments/l1_balance/RAW_REGULARIZED_V2_AMENDMENT.json"
)
TOPOLOGY_GATE_VERSION = "l1-topology-preregistered-v1"
EDGE_CUT_MAX_RATIO = 1.03
RETAINED_DEGREE_MIN_RATIO = 0.95
RETAINED_DEGREE_P10_MAX_DROP = 0.025
ISOLATED_FRACTION_MAX_DELTA = 0.01
ISOLATED_FRACTION_ABSOLUTE_MAX = 0.03
LARGEST_COMPONENT_MEAN_MAX_DROP = 0.02
LARGEST_COMPONENT_MIN_FLOOR = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--upper-input-manifest", required=True)
    parser.add_argument("--upper-navigation-hits", required=True)
    parser.add_argument("--upper-navigation-manifest", required=True)
    parser.add_argument("--num-partitions", type=int, default=32)
    parser.add_argument(
        "--mass-mode",
        choices=("self-debiased-floor1", "raw", "raw-regularized-v2"),
        default="self-debiased-floor1",
        help="Locked estimator or its ineligible raw-hit ablation.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(MASS_SUPPORTED_ALGORITHMS),
        choices=MASS_SUPPORTED_ALGORITHMS,
    )
    parser.add_argument("--owner-parity-reference-manifest")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mismatches(
    actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }


def validate_upper_navigation_contract(
    *,
    artifact_path: Path,
    labels: np.ndarray,
    vectors: np.ndarray,
    navigator_sha256: str,
    upper_input_manifest_path: Path,
    navigation_hits_path: Path,
    navigation_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove the navigation rows are ordered production-upper self queries."""

    artifact_sha256 = sha256_path(artifact_path)
    upper_count, dimension = vectors.shape
    upper_input = json.loads(upper_input_manifest_path.read_text(encoding="utf-8"))
    upper_vectors_path = Path(upper_input.get("vectors", "")).expanduser().resolve()
    upper_labels_path = Path(upper_input.get("labels", "")).expanduser().resolve()
    if not upper_vectors_path.is_file() or not upper_labels_path.is_file():
        raise ValueError("upper-input manifest references a missing vector/label file")

    expected_upper = {
        "artifact_sha256": artifact_sha256,
        "row_count": upper_count,
        "dimension": dimension,
        "vectors_sha256": sha256_path(upper_vectors_path),
        "labels_sha256": sha256_path(upper_labels_path),
        "vectors_size_bytes": upper_count * dimension * np.dtype("<f4").itemsize,
        "labels_size_bytes": upper_count * np.dtype("<u8").itemsize,
    }
    errors = mismatches(upper_input, expected_upper)
    if errors:
        raise ValueError(f"upper-input manifest mismatch: {errors}")

    extracted_labels = checked_binary_matrix(
        upper_labels_path, rows=upper_count, width=1, dtype="<u8"
    ).reshape(-1)
    if not np.array_equal(extracted_labels, labels.astype("<u8", copy=False)):
        raise ValueError("upper-input labels differ from artifact upper-node order")
    vector_bytes = vectors.astype("<f4", copy=False).tobytes(order="C")
    if upper_input["vectors_sha256"] != bytes_sha256(vector_bytes):
        raise ValueError("upper-input vectors differ from artifact float32 upper vectors")

    navigation = json.loads(navigation_manifest_path.read_text(encoding="utf-8"))
    expected_navigation = {
        "artifact_sha256": artifact_sha256,
        "upper_graph_present": True,
        "vectors_sha256": upper_input["vectors_sha256"],
        "row_count": upper_count,
        "dimension": dimension,
        "top_k": UPPER_NAVIGATION_TOP_K,
        "search_ef": UPPER_NAVIGATION_SEARCH_EF,
        "hits_sha256": sha256_path(navigation_hits_path),
        "hits_size_bytes": (
            upper_count * UPPER_NAVIGATION_TOP_K * np.dtype("<u8").itemsize
        ),
    }
    errors = mismatches(navigation, expected_navigation)
    if errors:
        raise ValueError(f"upper-navigation manifest mismatch: {errors}")
    if navigation.get("artifact_path") != str(artifact_path):
        raise ValueError("upper-navigation artifact path is not the frozen source")
    if upper_input.get("artifact") != str(artifact_path):
        raise ValueError("upper-input artifact path is not the frozen source")
    if not navigator_sha256:
        raise ValueError("production navigator checksum is empty")
    return upper_input, navigation


def topology_gate_results(
    observed: dict[str, float | int],
    reference: dict[str, float | int],
) -> dict[str, dict[str, Any]]:
    thresholds = {
        "upper_edge_cut_ratio": float(reference["upper_edge_cut_ratio"])
        * EDGE_CUT_MAX_RATIO,
        "retained_degree_mean": float(reference["retained_degree_mean"])
        * RETAINED_DEGREE_MIN_RATIO,
        "retained_degree_p10": float(reference["retained_degree_p10"])
        - RETAINED_DEGREE_P10_MAX_DROP,
        "upper_isolated_fraction": min(
            float(reference["upper_isolated_fraction"])
            + ISOLATED_FRACTION_MAX_DELTA,
            ISOLATED_FRACTION_ABSOLUTE_MAX,
        ),
        "largest_component_fraction_mean": float(
            reference["largest_component_fraction_mean"]
        )
        - LARGEST_COMPONENT_MEAN_MAX_DROP,
        "largest_component_fraction_min": LARGEST_COMPONENT_MIN_FLOOR,
    }
    operators = {
        "upper_edge_cut_ratio": "<=",
        "retained_degree_mean": ">=",
        "retained_degree_p10": ">=",
        "upper_isolated_fraction": "<=",
        "largest_component_fraction_mean": ">=",
        "largest_component_fraction_min": ">=",
    }
    gates: dict[str, dict[str, Any]] = {}
    for metric, threshold in thresholds.items():
        value = float(observed[metric])
        passed = value <= threshold if operators[metric] == "<=" else value >= threshold
        gates[metric] = {
            "observed": value,
            "operator": operators[metric],
            "threshold": threshold,
            "pass": bool(passed),
        }
    return gates


def duplicate_tie_proof(
    navigation_local: np.ndarray,
    labels: np.ndarray,
    vectors: np.ndarray,
) -> tuple[list[dict[str, Any]], str]:
    """Prove every non-rank-zero self hit is preceded only by bitwise duplicates."""

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
                "vector_bits_sha256": bytes_sha256(query_bits),
            }
        )
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return records, bytes_sha256(encoded)


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_owner_parity_reference(
    manifest_path: Path,
    *,
    artifact_sha256: str,
    method: str,
) -> dict[str, Any]:
    manifest_sha256 = sha256_path(manifest_path)
    sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    if sidecar.read_text(encoding="ascii").strip() != manifest_sha256:
        raise ValueError("raw parity-reference manifest checksum sidecar mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "upper_only_candidates_frozen":
        raise ValueError("raw parity reference is not a phase-A manifest")
    if manifest["parameters"].get("mass_mode") != "raw":
        raise ValueError("owner parity reference is not the frozen raw ablation")
    if manifest["construction_inputs"].get("artifact_sha256") != artifact_sha256:
        raise ValueError("owner parity reference uses a different source artifact")
    matches = [
        candidate
        for candidate in manifest["candidates"]
        if candidate.get("method") == method
    ]
    if len(matches) != 1:
        raise ValueError("owner parity reference lacks exactly one matching method")
    candidate = matches[0]
    if candidate.get("navigation_mass_source") != RAW_MASS_SOURCE:
        raise ValueError("owner parity reference does not use raw hit count")
    owner_path = Path(candidate["owner_path"]).resolve()
    if sha256_path(owner_path) != candidate["owner_sha256"]:
        raise ValueError("raw parity-reference owner checksum drifted")
    return {
        "reference_mass_mode": "raw",
        "reference_candidate_manifest_path": str(manifest_path),
        "reference_candidate_manifest_sha256": manifest_sha256,
        "reference_owner_path": str(owner_path),
        "reference_owner_sha256": candidate["owner_sha256"],
    }


def owner_parity_record(
    owner: np.ndarray,
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Prove the v2 owner is exactly the already-frozen raw owner bytes."""

    reference_owner_path = Path(reference["reference_owner_path"]).resolve()
    reference_bytes = reference_owner_path.read_bytes()
    owner_bytes = owner.astype("<i4", copy=False).tobytes(order="C")
    identical = owner_bytes == reference_bytes
    if bytes_sha256(reference_bytes) != reference["reference_owner_sha256"]:
        raise ValueError("raw parity-reference owner bytes drifted")
    if not identical:
        raise ValueError(
            "raw-regularized-v2 owner differs from the frozen raw owner bytes"
        )
    return {
        **reference,
        "owner_bytes_identical": True,
    }


def main() -> None:
    args = parse_args()
    if args.num_partitions <= 0:
        raise ValueError("num-partitions must be positive")
    is_v2_confirmation = args.mass_mode == "raw-regularized-v2"
    if is_v2_confirmation:
        if args.methods != ["mass-balanced-kmeans"]:
            raise ValueError("raw-regularized-v2 freezes only mass-balanced-kmeans")
        if not args.owner_parity_reference_manifest:
            raise ValueError("raw-regularized-v2 requires the frozen raw owner reference")
    elif args.owner_parity_reference_manifest:
        raise ValueError("owner parity reference is only valid for raw-regularized-v2")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    artifact_path = Path(args.artifact).expanduser().resolve()
    (
        artifact,
        adjacency,
        vectors,
        labels,
        entry_point,
        edge_left,
        edge_right,
        navigator_sha256,
    ) = load_upper(artifact_path)
    upper_graph_sha256 = canonical_sha256(artifact["upper_graph"])
    if int(artifact["shard_count"]) != args.num_partitions:
        raise ValueError("artifact shard_count differs from num-partitions")
    upper_input_manifest_path = Path(
        args.upper_input_manifest
    ).expanduser().resolve()
    navigation_hits_path = Path(
        args.upper_navigation_hits
    ).expanduser().resolve()
    navigation_manifest_path = Path(
        args.upper_navigation_manifest
    ).expanduser().resolve()
    upper_input, navigation = validate_upper_navigation_contract(
        artifact_path=artifact_path,
        labels=labels,
        vectors=vectors,
        navigator_sha256=navigator_sha256,
        upper_input_manifest_path=upper_input_manifest_path,
        navigation_hits_path=navigation_hits_path,
        navigation_manifest_path=navigation_manifest_path,
    )
    navigation_hits = checked_binary_matrix(
        navigation_hits_path,
        rows=len(labels),
        width=UPPER_NAVIGATION_TOP_K,
        dtype="<u8",
    )
    navigation_local = local_hits(navigation_hits, labels, len(labels))
    self_hits = navigation_local == np.arange(len(labels), dtype=np.int32)[:, None]
    if not np.all(np.any(self_hits, axis=1)):
        raise ValueError("a production upper self query lacks its own node in top-10")
    self_first_count = int(np.count_nonzero(self_hits[:, 0]))
    tie_records, tie_proof_sha256 = duplicate_tie_proof(
        navigation_local, labels, vectors
    )
    if args.mass_mode == "self-debiased-floor1":
        navigation_mass = estimate_upper_navigation_mass(
            navigation_local, len(labels)
        )
    elif args.mass_mode == "raw":
        navigation_mass = estimate_raw_upper_navigation_mass(
            navigation_local, len(labels)
        )
    else:
        navigation_mass = estimate_regularized_upper_navigation_mass(
            navigation_local, len(labels)
        )

    protocol_amendment = None
    parity_reference = None
    if is_v2_confirmation:
        amendment = json.loads(V2_AMENDMENT_PATH.read_text(encoding="utf-8"))
        required_amendment = {
            "id": "post-exploratory-raw-regularized-v2-confirmation",
            "status": "post_exploratory_promotion_not_preregistered",
            "mass_mode": "raw-regularized-v2",
            "source": REGULARIZED_MASS_SOURCE,
            "transform": navigation_mass.transform,
            "estimator_version": 2,
            "method": "mass-balanced-kmeans",
            "topology_gates_unchanged": True,
            "old_raw_owner_byte_parity_required": True,
            "online_qps_only_confirmation": True,
        }
        errors = mismatches(amendment, required_amendment)
        if errors:
            raise ValueError(f"v2 protocol amendment mismatch: {errors}")
        protocol_amendment = {
            **amendment,
            "amendment_path": str(V2_AMENDMENT_PATH.resolve()),
            "amendment_sha256": sha256_path(V2_AMENDMENT_PATH),
            "protocol_path": str(PROTOCOL_PATH.resolve()),
            "protocol_sha256": sha256_path(PROTOCOL_PATH),
        }
        parity_reference = load_owner_parity_reference(
            Path(args.owner_parity_reference_manifest).expanduser().resolve(),
            artifact_sha256=sha256_path(artifact_path),
            method="mass-balanced-kmeans",
        )

    partitioner_path = Path(mass_partitioner.__file__).resolve()
    partitioner_sha256 = sha256_path(partitioner_path)
    source_memberships = [node["shard_membership"] for node in artifact["upper_nodes"]]
    if not all(isinstance(row, list) and len(row) == 1 for row in source_memberships):
        raise ValueError("artifact lacks a singleton source L1 owner for topology gates")
    reference_owner = np.asarray(
        [int(row[0]) for row in source_memberships], dtype=np.int32
    )
    reference_topology = topology_metrics(
        reference_owner,
        edge_left,
        edge_right,
        len(labels),
        args.num_partitions,
    )

    results = []
    for method in args.methods:
        result = partition_l1_by_navigation_mass(
            adjacency,
            args.num_partitions,
            algorithm=method,
            navigation_mass=navigation_mass,
            vectors=vectors if method == "mass-balanced-kmeans" else None,
            entry_point=entry_point,
        )
        owner = np.asarray(result.owner, dtype=np.int32)
        metrics = topology_metrics(
            owner,
            edge_left,
            edge_right,
            len(owner),
            args.num_partitions,
        )
        gates = topology_gate_results(metrics, reference_topology)
        parity = (
            owner_parity_record(owner, parity_reference)
            if parity_reference is not None
            else None
        )
        results.append((method, owner, result, metrics, gates, parity))

    output_dir.mkdir(parents=True, exist_ok=False)
    owners_dir = output_dir / "owners"
    owners_dir.mkdir()
    mass_values_path = output_dir / "navigation-mass.u64le"
    np.asarray(navigation_mass.values, dtype="<u8").tofile(mass_values_path)
    mass_manifest_path = output_dir / "navigation-mass-manifest.json"
    mass_manifest = {
        "format_version": 1,
        "mass_mode": args.mass_mode,
        "source": navigation_mass.source,
        "transform": navigation_mass.transform,
        "estimator_version": navigation_mass.estimator_version,
        "semantic_sha256": navigation_mass.sha256,
        "values_file": str(mass_values_path),
        "values_sha256": sha256_path(mass_values_path),
        "values_size_bytes": mass_values_path.stat().st_size,
        "total_mass": navigation_mass.total_mass,
        "source_artifact": str(artifact_path),
        "source_artifact_sha256": sha256_path(artifact_path),
        "navigator_sha256": navigator_sha256,
        "upper_graph_sha256": upper_graph_sha256,
        "ordered_labels_sha256": upper_input["labels_sha256"],
        "ordered_vectors_sha256": upper_input["vectors_sha256"],
        "upper_navigation_hits": str(navigation_hits_path),
        "upper_navigation_hits_sha256": navigation["hits_sha256"],
        "upper_navigation_manifest": str(navigation_manifest_path),
        "upper_navigation_manifest_sha256": sha256_path(navigation_manifest_path),
        "upper_node_count": len(labels),
        "dimension": int(vectors.shape[1]),
        "top_k": UPPER_NAVIGATION_TOP_K,
        "search_ef": UPPER_NAVIGATION_SEARCH_EF,
        "self_present_count": len(labels),
        "self_first_count": self_first_count,
        "self_first_fraction": self_first_count / len(labels),
        "duplicate_tie_exception_count": len(tie_records),
        "duplicate_tie_exceptions": tie_records,
        "duplicate_tie_proof_sha256": tie_proof_sha256,
        "partitioner_source": str(partitioner_path),
        "partitioner_source_sha256": partitioner_sha256,
        "partitioner_config": {
            "algorithms": list(MASS_SUPPORTED_ALGORITHMS),
            "deterministic_seed": mass_partitioner.MASS_DETERMINISTIC_SEED,
            "kmeans_iterations": mass_partitioner.MASS_KMEANS_ITERATIONS,
            "capacity_rule": "ceil(total_mass / P) + max_vertex_mass - 1",
            "graph_normalization": "deterministic_undirected_simple",
        },
        "protocol_amendment": protocol_amendment,
    }
    write_json_new(mass_manifest_path, mass_manifest)
    mass_manifest_sha256 = sha256_path(mass_manifest_path)

    candidates: list[dict[str, Any]] = []
    for method, owner, result, metrics, gates, parity in results:
        owner_path = owners_dir / f"{method}.owner.i32le"
        owner.astype("<i4", copy=False).tofile(owner_path)
        graph_all_pass = all(bool(gate["pass"]) for gate in gates.values())
        candidate = {
            "method": method,
            "balance_contract": "navigation_mass_capacity",
            "source_artifact_sha256": sha256_path(artifact_path),
            "owner_path": str(owner_path),
            "owner_sha256": sha256_path(owner_path),
            "partitioner_source_sha256": partitioner_sha256,
            "navigation_mass_manifest_path": str(mass_manifest_path),
            "navigation_mass_manifest_sha256": mass_manifest_sha256,
            "navigation_mass_source": navigation_mass.source,
            "navigation_mass_transform": navigation_mass.transform,
            "navigation_mass_estimator_version": (
                navigation_mass.estimator_version
            ),
            "navigation_mass_sha256": navigation_mass.sha256,
            "partition_sizes": list(result.partition_sizes),
            "estimated_partition_masses": list(
                result.estimated_partition_masses
            ),
            "estimated_mass_target": result.mass_target,
            "estimated_mass_limit": result.mass_limit,
            "partition_seconds": result.elapsed_seconds,
            "partition_edge_visits": result.edge_visits,
            "topology_metrics": metrics,
            "graph_topology_gates": gates,
            "graph_topology_all_pass": graph_all_pass,
            "topology_all_pass": False,
            "mass_mode": args.mass_mode,
            "protocol_amendment": protocol_amendment,
            "owner_parity": parity,
            "eligibility_pending_post_freeze_query_gates": True,
            "materialization_eligible": False,
        }
        candidates.append(candidate)
        print(
            f"method={method} cut={metrics['upper_edge_cut_ratio']:.6f} "
            f"mass_max_mean={max(result.estimated_partition_masses) / result.mass_target:.6f} "
            f"graph_topology_all_pass={graph_all_pass} eligible=False",
            flush=True,
        )

    mass_array = np.asarray(navigation_mass.values, dtype=np.int64)
    candidate_manifest_path = output_dir / "candidate-manifest.json"
    candidate_manifest = {
        "format_version": 1,
        "stage": "upper_only_candidates_frozen",
        "contract": {
            "partition_input_scope": "production_upper_graph_and_upper_navigation_mass_only",
            "construction_process_accepts_l0_inputs": False,
            "construction_process_accepts_query_ground_truth": False,
            "partitioner_reads_full_attachments": False,
            "partitioner_reads_l0_load": False,
            "partitioner_reads_multi_assignment_state": False,
            "separate_evaluator_process_required": True,
            "l0_repair_or_flow": False,
        },
        "mass_estimator": {
            "format_version": 1,
            "mass_mode": args.mass_mode,
            "source": navigation_mass.source,
            "transform": navigation_mass.transform,
            "estimator_version": navigation_mass.estimator_version,
            "sha256": navigation_mass.sha256,
            "manifest_path": str(mass_manifest_path),
            "manifest_sha256": mass_manifest_sha256,
            "query_count": navigation_mass.query_count,
            "top_k": navigation_mass.top_k,
            "total_mass": navigation_mass.total_mass,
            "self_present_count": len(labels),
            "self_first_count": self_first_count,
            "self_first_fraction": self_first_count / len(labels),
            "duplicate_tie_exception_count": len(tie_records),
            "duplicate_tie_proof_sha256": tie_proof_sha256,
            "vertex_mass_min": int(np.min(mass_array)),
            "vertex_mass_max": int(np.max(mass_array)),
            "vertex_mass_mean": float(np.mean(mass_array)),
            "vertex_mass_cv": float(np.std(mass_array) / np.mean(mass_array)),
            "rank_weighting": False,
            "eligible_as_finalist": is_v2_confirmation,
        },
        "protocol_amendment": protocol_amendment,
        "topology_gate_contract": {
            "version": TOPOLOGY_GATE_VERSION,
            "preregistered_before_mass_candidate_results": True,
            "reference": "source_singleton_l1_owner",
            "reference_metrics": reference_topology,
            "edge_cut_max_ratio": EDGE_CUT_MAX_RATIO,
            "retained_degree_min_ratio": RETAINED_DEGREE_MIN_RATIO,
            "retained_degree_p10_max_drop": RETAINED_DEGREE_P10_MAX_DROP,
            "isolated_fraction_max_delta": ISOLATED_FRACTION_MAX_DELTA,
            "isolated_fraction_absolute_max": ISOLATED_FRACTION_ABSOLUTE_MAX,
            "largest_component_mean_max_drop": LARGEST_COMPONENT_MEAN_MAX_DROP,
            "largest_component_min_floor": LARGEST_COMPONENT_MIN_FLOOR,
            "all_gates_required": True,
        },
        "construction_inputs": {
            "artifact": str(artifact_path),
            "artifact_sha256": sha256_path(artifact_path),
            "navigator_sha256": navigator_sha256,
            "upper_graph_sha256": upper_graph_sha256,
            "logical_point_count": int(artifact["logical_point_count"]),
            "upper_node_count": len(labels),
            "dimension": int(vectors.shape[1]),
            "ordered_labels_sha256": upper_input["labels_sha256"],
            "ordered_vectors_sha256": upper_input["vectors_sha256"],
            "upper_input_manifest": str(upper_input_manifest_path),
            "upper_input_manifest_sha256": sha256_path(upper_input_manifest_path),
            "upper_navigation_hits": str(navigation_hits_path),
            "upper_navigation_hits_sha256": navigation["hits_sha256"],
            "upper_navigation_manifest": str(navigation_manifest_path),
            "upper_navigation_manifest_sha256": sha256_path(
                navigation_manifest_path
            ),
            "upper_navigation_top_k": UPPER_NAVIGATION_TOP_K,
            "upper_navigation_search_ef": UPPER_NAVIGATION_SEARCH_EF,
            "partitioner_source": str(partitioner_path),
            "partitioner_source_sha256": partitioner_sha256,
            "partitioner_config": mass_manifest["partitioner_config"],
        },
        "parameters": {
            "num_partitions": args.num_partitions,
            "mass_mode": args.mass_mode,
            "methods": args.methods,
        },
        "candidates": candidates,
        "outputs": {
            "navigation_mass": str(mass_values_path),
            "navigation_mass_manifest": str(mass_manifest_path),
            "owners_dir": str(owners_dir),
        },
    }
    write_json_new(candidate_manifest_path, candidate_manifest)
    candidate_manifest_sha256 = sha256_path(candidate_manifest_path)
    checksum_path = candidate_manifest_path.with_name(
        candidate_manifest_path.name + ".sha256"
    )
    checksum_path.write_text(candidate_manifest_sha256 + "\n", encoding="ascii")

    # Construction artifacts are immutable inputs to the later evaluator.
    for path in (
        mass_values_path,
        mass_manifest_path,
        candidate_manifest_path,
        checksum_path,
        *(Path(candidate["owner_path"]) for candidate in candidates),
    ):
        os.chmod(path, 0o444)
    print(f"candidate_manifest={candidate_manifest_path}")
    print(f"candidate_manifest_sha256={candidate_manifest_sha256}")


if __name__ == "__main__":
    main()
