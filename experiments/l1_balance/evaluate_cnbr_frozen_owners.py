#!/usr/bin/env python3
"""Evaluate frozen N_native and C_CNBR owners without rebuilding either owner.

Phase A is audited completely before this process opens full L0 attachments,
query hits, or ground truth.  The evaluator never invokes KMeans, CNBR, a
partitioner, or an owner-repair routine.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance.run_offline_screen import (  # noqa: E402
    canonical_sha256,
    checked_binary_matrix,
    load_upper,
    local_hits,
    physical_membership,
    query_metrics,
    sha256_path,
    topology_metrics as screen_topology_metrics,
)
from experiments.l1_balance import prepare_native_cnbr_phase_a as phase_a_contract  # noqa: E402
from experiments.l1_balance import freeze_phase_b_source_binding as source_binding_contract  # noqa: E402


FORMAL_PHASE_A_STAGE = phase_a_contract.FORMAL_STAGE
GRID_PHASE_A_STAGE = phase_a_contract.GRID_STAGE
PHASE_B_STAGE = "post_freeze_evaluation"
REFERENCE_NAME = "N_native"
CANDIDATE_NAME = "C_CNBR"
FORMAL_RATIOS = {CANDIDATE_NAME: (9, 4)}
GRID_RATIOS = {
    phase_a_contract.GRID_OWNER_KEYS[trigger_id]: ratio
    for trigger_id, ratio in phase_a_contract.core.CNBR_FROZEN_TRIGGER_FAMILY.items()
}

UPPER_NAVIGATION_TOP_K = phase_a_contract.core.UPPER_NAVIGATION_TOP_K
UPPER_NAVIGATION_SEARCH_EF = phase_a_contract.UPPER_NAVIGATION_SEARCH_EF
ATTACHMENT_SEARCH_EF = phase_a_contract.UPPER_NAVIGATION_SEARCH_EF
MASS_MODE = phase_a_contract.core.PROXY_MASS_CONTRACT_ID
MASS_SOURCE = phase_a_contract.core.MASS_SOURCE
MASS_TRANSFORM = phase_a_contract.core.MASS_TRANSFORM
MASS_ESTIMATOR_VERSION = phase_a_contract.core.MASS_ESTIMATOR_VERSION

EDGE_CUT_MAX_RATIO = phase_a_contract.core.EDGE_CUT_MAX_RATIO
RETAINED_DEGREE_MIN_RATIO = phase_a_contract.core.RETAINED_DEGREE_MIN_RATIO
RETAINED_DEGREE_P10_MAX_DROP = phase_a_contract.core.RETAINED_DEGREE_P10_MAX_DROP
ISOLATED_FRACTION_MAX_DELTA = phase_a_contract.core.ISOLATED_FRACTION_MAX_DELTA
ISOLATED_FRACTION_ABSOLUTE_MAX = (
    phase_a_contract.core.ISOLATED_FRACTION_ABSOLUTE_MAX
)
LARGEST_COMPONENT_MEAN_MAX_DROP = (
    phase_a_contract.core.LARGEST_COMPONENT_MEAN_MAX_DROP
)
LARGEST_COMPONENT_MIN_FLOOR = phase_a_contract.core.LARGEST_COMPONENT_MIN_FLOOR

MULTI_ASSIGNMENT_CONTRACT = {
    "enabled": True,
    "min_max_vote": 2,
    "vote_delta": 0,
    "max_shards": 0,
}


@dataclass(frozen=True)
class FrozenFile:
    label: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class FrozenOwner:
    name: str
    role: str
    record: dict[str, Any]
    path: Path
    sha256: str
    values: np.ndarray
    owner_record_path: Path
    owner_record_sha256: str
    owner_record: dict[str, Any]


@dataclass(frozen=True)
class FrozenPhaseA:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    artifact_path: Path
    artifact: dict[str, Any]
    labels: np.ndarray
    vectors: np.ndarray
    edge_left: np.ndarray
    edge_right: np.ndarray
    num_partitions: int
    candidate_set: str
    candidate_ratios: dict[str, tuple[int, int]]
    selected_candidate_name: str
    mass_values: np.ndarray
    owners: dict[str, FrozenOwner]
    performance_path: Path
    performance_sha256: str
    performance: dict[str, Any]
    frozen_files: tuple[FrozenFile, ...]


@dataclass(frozen=True)
class PhaseBInputs:
    dataset_manifest_path: Path
    dataset_manifest_sha256: str
    dataset_sha256: str
    source_build_manifest_path: Path
    source_build_manifest_sha256: str
    attachments_path: Path
    attachments_manifest_path: Path
    attachments_sha256: str
    attachment_local: np.ndarray
    query_hits_path: Path
    query_manifest_path: Path
    query_hits_sha256: str
    query_manifest: dict[str, Any]
    query_local: np.ndarray
    ground_truth_path: Path
    ground_truth_manifest_path: Path
    ground_truth_manifest_sha256: str
    ground_truth_sha256: str
    ground_truth: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--source-build-manifest", required=True)
    parser.add_argument("--attachments", required=True)
    parser.add_argument("--attachments-manifest", required=True)
    parser.add_argument("--row-count", required=True, type=int)
    parser.add_argument("--attachment-k", default=10, type=int)
    parser.add_argument("--query-hits", required=True)
    parser.add_argument("--query-hits-manifest", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--ground-truth-manifest", required=True)
    parser.add_argument("--ground-truth-width", default=10, type=int)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _mismatches(
    actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }


def _canonical_close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _canonical_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _canonical_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))
    return left == right


def _require_read_only(path: Path, label: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o222:
        raise ValueError(f"{label} is not frozen read-only: mode={mode:o}")


def _resolve_record_path(
    base_dir: Path,
    record: dict[str, Any],
    *,
    label: str,
    require_relative: bool,
) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label}.path is missing")
    supplied = Path(raw).expanduser()
    if require_relative and supplied.is_absolute():
        raise ValueError(f"{label}.path must be relative to the Phase-A manifest")
    if not require_relative and not supplied.is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    return (base_dir / supplied).resolve() if require_relative else supplied.resolve()


def _validate_file_record(
    base_dir: Path,
    value: Any,
    *,
    label: str,
    require_relative: bool,
    require_read_only: bool,
    frozen_files: list[FrozenFile],
) -> tuple[dict[str, Any], Path]:
    record = _mapping(value, label)
    path = _resolve_record_path(
        base_dir, record, label=label, require_relative=require_relative
    )
    if not path.is_file():
        raise ValueError(f"{label} does not reference a regular file: {path}")
    actual_size = path.stat().st_size
    if "size_bytes" in record and record.get("size_bytes") != actual_size:
        raise ValueError(
            f"{label} size drifted: expected {record.get('size_bytes')}, "
            f"actual {actual_size}"
        )
    actual_sha256 = sha256_path(path)
    if record.get("sha256") != actual_sha256:
        raise ValueError(f"{label} checksum drifted")
    if require_read_only:
        _require_read_only(path, label)
    frozen_files.append(FrozenFile(label, path, actual_sha256))
    return record, path


def _semantic_mass_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(MASS_SOURCE.encode("ascii"))
    digest.update(b"\0")
    digest.update(MASS_TRANSFORM.encode("ascii"))
    digest.update(struct.pack("<QQ", len(values), UPPER_NAVIGATION_TOP_K))
    for value in values:
        digest.update(struct.pack("<Q", int(value)))
    return digest.hexdigest()


def _assignment_bytes(point_id: int, shards: np.ndarray) -> bytes:
    return (
        f'{{"id":{point_id},"shards":['
        + ",".join(str(int(shard)) for shard in shards)
        + "]}\n"
    ).encode("ascii")


def membership_assignment_sha256(membership: np.ndarray) -> str:
    """Hash the exact JSONL bytes accepted by Orion's numeric importer."""

    digest = hashlib.sha256()
    for point_id, row in enumerate(membership):
        digest.update(_assignment_bytes(point_id, np.flatnonzero(row)))
    return digest.hexdigest()


def _duplicate_tie_proof(
    navigation_local: np.ndarray,
    labels: np.ndarray,
    vectors: np.ndarray,
) -> tuple[list[dict[str, Any]], int, str]:
    records: list[dict[str, Any]] = []
    self_first_count = 0
    for row in range(len(navigation_local)):
        values = np.asarray(navigation_local[row])
        if len(np.unique(values)) != UPPER_NAVIGATION_TOP_K:
            raise ValueError(f"upper self-navigation row {row} repeats a hit")
        positions = np.flatnonzero(values == row)
        if len(positions) != 1:
            raise ValueError(
                f"upper self-navigation row {row} has {len(positions)} self hits"
            )
        self_rank = int(positions[0])
        if self_rank == 0:
            self_first_count += 1
            continue
        query_bits = np.ascontiguousarray(vectors[row]).tobytes()
        preceding = values[:self_rank]
        for hit in preceding:
            if np.ascontiguousarray(vectors[int(hit)]).tobytes() != query_bits:
                raise ValueError(
                    f"upper self-navigation row {row} has a non-duplicate hit before self"
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
    return records, self_first_count, hashlib.sha256(encoded).hexdigest()


def _phase_a_topology_metrics(
    owner: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    node_count: int,
    num_partitions: int,
) -> dict[str, float | int]:
    metrics = screen_topology_metrics(
        owner, edge_left, edge_right, node_count, num_partitions
    )
    return {
        "upper_edge_count": int(metrics["upper_edge_count"]),
        "upper_edge_cut_count": int(
            np.count_nonzero(owner[edge_left] != owner[edge_right])
        ),
        "upper_edge_cut_ratio": float(metrics["upper_edge_cut_ratio"]),
        "retained_degree_mean": float(metrics["retained_degree_mean"]),
        "retained_degree_p10": float(metrics["retained_degree_p10"]),
        "upper_isolated_fraction": float(metrics["upper_isolated_fraction"]),
        "largest_component_fraction_mean": float(
            metrics["largest_component_fraction_mean"]
        ),
        "largest_component_fraction_min": float(
            metrics["largest_component_fraction_min"]
        ),
    }


def graph_gate_results(
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
        operator = operators[metric]
        passed = value <= threshold if operator == "<=" else value >= threshold
        gates[metric] = {
            "reference": float(reference[metric]),
            "observed": value,
            "operator": operator,
            "threshold": float(threshold),
            "pass": bool(passed),
        }
    return gates


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
            "reference": float(reference[metric]),
            "observed": value,
            "operator": operator,
            "threshold": float(threshold),
            "pass": bool(passed),
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
    observed_zero = int(observed["gt_queries_zero_coverage"])
    reference_zero = int(reference["gt_queries_zero_coverage"])
    threshold = min(reference_zero + 2, int(math.floor(query_count * 0.001)))
    gates["gt_queries_zero_coverage"] = {
        "reference": reference_zero,
        "observed": observed_zero,
        "operator": "<=",
        "threshold": threshold,
        "pass": observed_zero <= threshold,
    }
    return gates


def _validate_graph_gate_contract(value: Any) -> None:
    contract = _mapping(value, "graph_gate_contract")
    expected = phase_a_contract.phase_a_graph_gate_contract()
    if contract != expected:
        raise ValueError("graph gate contract differs from the Phase-A generator")


def _validate_fixed_parameters(
    value: Any, num_partitions: int, candidate_set: str
) -> None:
    fixed = _mapping(value, "fixed_parameters")
    if fixed.get("num_partitions") != num_partitions:
        raise ValueError("fixed_parameters.num_partitions drifted")
    expected = phase_a_contract.phase_a_fixed_parameters(candidate_set)
    if fixed != expected:
        raise ValueError("fixed_parameters differ from the Phase-A generator contract")
    encoded = json.dumps(fixed, sort_keys=True, allow_nan=False)
    if '2.25' in encoded:
        raise ValueError("fixed_parameters contain a floating trigger representation")


def _validate_self_navigation_manifest(
    manifest: dict[str, Any],
    *,
    artifact_path: Path,
    artifact_sha256: str,
    vectors_sha256: str,
    hits_path: Path,
    hits_sha256: str,
    upper_count: int,
    dimension: int,
) -> None:
    expected = {
        "artifact_sha256": artifact_sha256,
        "row_count": upper_count,
        "dimension": dimension,
        "top_k": UPPER_NAVIGATION_TOP_K,
        "search_ef": UPPER_NAVIGATION_SEARCH_EF,
        "hits_sha256": hits_sha256,
    }
    errors = _mismatches(manifest, expected)
    if errors:
        raise ValueError(f"upper self-navigation manifest drifted: {errors}")
    if "artifact_path" in manifest and Path(manifest["artifact_path"]).resolve() != artifact_path:
        raise ValueError("upper self-navigation manifest artifact path drifted")
    if "vectors_sha256" in manifest and manifest["vectors_sha256"] != vectors_sha256:
        raise ValueError("upper self-navigation manifest vector identity drifted")
    if "hits_path" in manifest and Path(manifest["hits_path"]).resolve() != hits_path:
        raise ValueError("upper self-navigation manifest hit path drifted")


def _validate_owner_record(
    *,
    name: str,
    expected_role: str,
    owner_dtype: str,
    owner_encoding: str,
    estimated_mass_field: str,
    value: Any,
    manifest_dir: Path,
    upper_count: int,
    num_partitions: int,
    mass_values: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    frozen_files: list[FrozenFile],
) -> FrozenOwner:
    record = _mapping(value, f"owners.{name}")
    if record.get("role") != expected_role:
        raise ValueError(f"owners.{name}.role must be {expected_role}")
    owner_record, owner_path = _validate_file_record(
        manifest_dir,
        record.get("owner"),
        label=f"owners.{name}.owner",
        require_relative=True,
        require_read_only=True,
        frozen_files=frozen_files,
    )
    if (
        owner_record.get("dtype") != owner_dtype
        or owner_record.get("encoding") != owner_encoding
        or owner_record.get("row_count") != upper_count
        or owner_record.get("count") != upper_count
    ):
        raise ValueError(f"owners.{name}.owner shape or dtype drifted")
    owner = checked_binary_matrix(
        owner_path, rows=upper_count, width=1, dtype=owner_dtype
    ).reshape(-1)
    owner = np.asarray(owner, dtype=np.int32).copy()
    if np.any(owner < 0) or np.any(owner >= num_partitions):
        raise ValueError(f"owners.{name} contains an invalid partition")
    sizes = np.bincount(owner, minlength=num_partitions)
    if np.any(sizes == 0):
        raise ValueError(f"owners.{name} contains an empty partition")
    if sizes.tolist() != record.get("partition_sizes"):
        raise ValueError(f"owners.{name} partition sizes drifted")
    masses = np.zeros(num_partitions, dtype=np.int64)
    np.add.at(masses, owner, mass_values.astype(np.int64, copy=False))
    if masses.tolist() != record.get(estimated_mass_field):
        raise ValueError(f"owners.{name} estimated masses drifted")
    metrics = _phase_a_topology_metrics(
        owner, edge_left, edge_right, upper_count, num_partitions
    )
    if not _canonical_close(metrics, record.get("topology_metrics")):
        raise ValueError(f"owners.{name} topology metrics drifted")
    owner_record_binding, owner_record_path = _validate_file_record(
        manifest_dir,
        record.get("owner_record"),
        label=f"owners.{name}.owner_record",
        require_relative=True,
        require_read_only=True,
        frozen_files=frozen_files,
    )
    owner_record_sha256 = owner_record_binding["sha256"]
    if record.get("phase_a_owner_record_sha256") != owner_record_sha256:
        raise ValueError(f"owners.{name} owner-record checksum binding drifted")
    owner_record_value = json.loads(owner_record_path.read_text(encoding="utf-8"))
    common_expected = {
        "format_version": 1,
        "record_type": "phase_a_owner",
        "candidate_id": record.get("candidate_id"),
        "role": expected_role,
        "owner": record.get("owner"),
        "partition_sizes": record.get("partition_sizes"),
        estimated_mass_field: record.get(estimated_mass_field),
        "topology_metrics": record.get("topology_metrics"),
        "topology_gates": record.get("topology_gates"),
        "topology_all_pass": record.get("topology_all_pass"),
    }
    errors = _mismatches(owner_record_value, common_expected)
    if errors:
        raise ValueError(f"owners.{name} owner-record content drifted: {errors}")
    return FrozenOwner(
        name=name,
        role=expected_role,
        record=record,
        path=owner_path,
        sha256=owner_record["sha256"],
        values=owner,
        owner_record_path=owner_record_path,
        owner_record_sha256=owner_record_sha256,
        owner_record=owner_record_value,
    )


def _owner_sha256(owner: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(owner, dtype="<i4").tobytes(order="C")
    ).hexdigest()


def _load_predicate_evidence(
    *,
    source_load: int,
    target_load: int,
    node_mass: int,
    trigger_numerator: int,
    trigger_denominator: int,
) -> dict[str, dict[str, int | str | bool]]:
    return {
        "source_above_trigger": {
            "lhs": source_load * trigger_denominator,
            "operator": ">",
            "rhs": trigger_numerator,
            "pass": source_load * trigger_denominator > trigger_numerator,
        },
        "target_below_trigger": {
            "lhs": target_load * trigger_denominator,
            "operator": "<",
            "rhs": trigger_numerator,
            "pass": target_load * trigger_denominator < trigger_numerator,
        },
        "source_target_gap_gt_node_mass": {
            "lhs": source_load - target_load,
            "operator": ">",
            "rhs": node_mass,
            "pass": source_load - target_load > node_mass,
        },
        "post_move_target_at_or_below_trigger": {
            "lhs": (target_load + node_mass) * trigger_denominator,
            "operator": "<=",
            "rhs": trigger_numerator,
            "pass": (target_load + node_mass) * trigger_denominator
            <= trigger_numerator,
        },
    }


def _validate_cnbr_trace(
    *,
    name: str,
    trace: dict[str, Any],
    rounds_record: dict[str, Any],
    reference_owner: np.ndarray,
    candidate_owner: np.ndarray,
    mass_values: np.ndarray,
    adjacency: list[list[int]],
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    num_partitions: int,
    ratio_numerator: int,
    ratio_denominator: int,
    reference_metrics: dict[str, Any],
) -> None:
    total_mass = int(np.sum(mass_values))
    trigger_numerator = ratio_numerator * total_mass
    trigger_denominator = ratio_denominator * num_partitions
    expected_header = {
        "format_version": 1,
        "record_type": "cnbr_round_trace",
        "candidate_id": "C_CNBR" if ratio_numerator == 9 and ratio_denominator == 4 else name,
        "family_member_id": name,
        "trigger_ratio": {
            "numerator": ratio_numerator,
            "denominator": ratio_denominator,
        },
        "base_owner_sha256": _owner_sha256(reference_owner),
    }
    errors = _mismatches(trace, expected_header)
    if errors:
        raise ValueError(f"{name} trace header drifted: {errors}")
    rounds = trace.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError(f"{name} trace rounds must be a list")
    owner = reference_owner.copy()
    moved: set[int] = set()
    loads = np.zeros(num_partitions, dtype=np.int64)
    np.add.at(loads, owner, mass_values.astype(np.int64, copy=False))
    proposal_count = 0
    committed_count = 0
    rolled_back_count = 0
    for expected_round, round_record in enumerate(rounds, start=1):
        if not isinstance(round_record, dict) or round_record.get("round") != expected_round:
            raise ValueError(f"{name} trace round numbering drifted")
        snapshot_owner = owner.copy()
        snapshot_loads = loads.copy()
        snapshot_moved = set(moved)
        if round_record.get("pre_owner_sha256") != _owner_sha256(owner):
            raise ValueError(f"{name} round {expected_round} pre-owner SHA drifted")
        if round_record.get("pre_partition_masses") != loads.tolist():
            raise ValueError(f"{name} round {expected_round} pre-loads drifted")
        proposals = round_record.get("proposals")
        decisions = round_record.get("commit_decisions")
        accepted = round_record.get("accepted_moves")
        if not all(isinstance(value, list) for value in (proposals, decisions, accepted)):
            raise ValueError(f"{name} round {expected_round} trace lists are malformed")
        if round_record.get("proposal_count") != len(proposals):
            raise ValueError(f"{name} round {expected_round} proposal count drifted")
        proposal_count += len(proposals)
        accepted_decisions = [
            decision for decision in decisions if decision.get("status") == "accepted"
        ]
        if accepted_decisions != accepted or len(accepted) != round_record.get(
            "accepted_move_count"
        ):
            raise ValueError(f"{name} round {expected_round} accepted trace drifted")
        for move in accepted:
            node = int(move["node"])
            source = int(move["source"])
            target = int(move["target"])
            node_mass = int(mass_values[node])
            if node in moved or int(owner[node]) != source or int(move["node_mass"]) != node_mass:
                raise ValueError(f"{name} round {expected_round} replays an invalid move")
            source_load = int(loads[source])
            target_load = int(loads[target])
            evidence = _load_predicate_evidence(
                source_load=source_load,
                target_load=target_load,
                node_mass=node_mass,
                trigger_numerator=trigger_numerator,
                trigger_denominator=trigger_denominator,
            )
            if move.get("commit_load_predicates") != evidence or not all(
                bool(item["pass"]) for item in evidence.values()
            ):
                raise ValueError(f"{name} round {expected_round} load proof drifted")
            source_neighbors = sum(
                1 for neighbor in adjacency[node] if int(owner[neighbor]) == source
            )
            target_neighbors = sum(
                1 for neighbor in adjacency[node] if int(owner[neighbor]) == target
            )
            if move.get("commit_edge_delta") != source_neighbors - target_neighbors:
                raise ValueError(f"{name} round {expected_round} edge proof drifted")
            owner[node] = target
            loads[source] -= node_mass
            loads[target] += node_mass
            moved.add(node)
        if round_record.get("post_partition_masses") != loads.tolist():
            raise ValueError(f"{name} round {expected_round} post-loads drifted")
        if round_record.get("post_owner_sha256_before_round_gate") != _owner_sha256(owner):
            raise ValueError(f"{name} round {expected_round} post-owner SHA drifted")
        observed_metrics = _phase_a_topology_metrics(
            owner, edge_left, edge_right, len(owner), num_partitions
        )
        expected_gates = graph_gate_results(observed_metrics, reference_metrics)
        if not _canonical_close(observed_metrics, round_record.get("post_topology_metrics")):
            raise ValueError(f"{name} round {expected_round} topology replay drifted")
        if not _canonical_close(expected_gates, round_record.get("graph_gates")):
            raise ValueError(f"{name} round {expected_round} gate replay drifted")
        gates_pass = all(bool(gate["pass"]) for gate in expected_gates.values())
        if round_record.get("graph_gates_all_pass") is not gates_pass:
            raise ValueError(f"{name} round {expected_round} gate pass bit drifted")
        if bool(round_record.get("rolled_back")):
            rolled_back_count += 1
            owner = snapshot_owner
            loads = snapshot_loads
            moved = snapshot_moved
            if round_record.get("post_partition_masses_after_rollback") != loads.tolist():
                raise ValueError(f"{name} round {expected_round} rollback loads drifted")
            if round_record.get("post_owner_sha256_after_rollback") != _owner_sha256(owner):
                raise ValueError(f"{name} round {expected_round} rollback SHA drifted")
        else:
            committed_count += len(accepted)
    if not np.array_equal(owner, candidate_owner):
        raise ValueError(f"{name} trace does not replay to the frozen owner")
    final_moved = sorted(int(value) for value in trace.get("final_moved_nodes", []))
    if final_moved != sorted(moved):
        raise ValueError(f"{name} final moved-node trace drifted")
    final_loads = np.zeros(num_partitions, dtype=np.int64)
    np.add.at(final_loads, owner, mass_values.astype(np.int64, copy=False))
    final_metrics = _phase_a_topology_metrics(
        owner, edge_left, edge_right, len(owner), num_partitions
    )
    final_gates = graph_gate_results(final_metrics, reference_metrics)
    if trace.get("final_partition_proxy_masses") != final_loads.tolist():
        raise ValueError(f"{name} final proxy loads drifted")
    if not _canonical_close(trace.get("final_topology_metrics"), final_metrics):
        raise ValueError(f"{name} final topology trace drifted")
    if not _canonical_close(trace.get("final_topology_gates"), final_gates):
        raise ValueError(f"{name} final topology gates drifted")
    aggregate_expected = {
        "count": len(rounds),
        "proposal_count": proposal_count,
        "committed_move_count": committed_count,
        "rolled_back_round_count": rolled_back_count,
    }
    errors = _mismatches(rounds_record, aggregate_expected)
    if errors:
        raise ValueError(f"{name} trace aggregate drifted: {errors}")


def validate_phase_a(args: argparse.Namespace) -> FrozenPhaseA:
    """Audit every frozen Phase-A byte before any Phase-B input is opened."""

    manifest_path = Path(args.phase_a_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Phase-A manifest does not exist: {manifest_path}")
    sidecar_path = manifest_path.with_name(manifest_path.name + ".sha256")
    manifest_sha256 = sha256_path(manifest_path)
    if sidecar_path.read_text(encoding="ascii").strip() != manifest_sha256:
        raise ValueError("Phase-A manifest checksum sidecar mismatch")
    _require_read_only(manifest_path, "Phase-A manifest")
    _require_read_only(sidecar_path, "Phase-A manifest sidecar")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = manifest.get("stage")
    if manifest.get("format_version") != phase_a_contract.FORMAT_VERSION or stage not in {
        FORMAL_PHASE_A_STAGE,
        GRID_PHASE_A_STAGE,
    }:
        raise ValueError("input is not the frozen N_native/C_CNBR Phase-A manifest")
    if stage == FORMAL_PHASE_A_STAGE:
        candidate_set = "formal"
        candidate_ratios = FORMAL_RATIOS
        selected_candidate_name = CANDIDATE_NAME
    else:
        candidate_set = "frozen-grid"
        candidate_ratios = GRID_RATIOS
        selected_candidate_name = "CNBR_9_4"
    schema = phase_a_contract.phase_a_schema(candidate_set)
    if stage != schema["stage"] or manifest.get("candidate_set") != candidate_set:
        raise ValueError("Phase-A candidate_set differs from its stage")

    contract = _mapping(manifest.get("contract"), "contract")
    if contract != schema["contract"]:
        raise ValueError("Phase-A architecture contract differs from its generator")
    forbidden = _mapping(
        manifest.get("forbidden_stage_invocations"),
        "forbidden_stage_invocations",
    )
    expected_forbidden = schema["forbidden_stage_invocations"]
    if forbidden != expected_forbidden:
        raise ValueError("Phase-A forbidden-stage invocation counters drifted")

    manifest_dir = manifest_path.parent
    frozen_files: list[FrozenFile] = [
        FrozenFile("Phase-A manifest", manifest_path, manifest_sha256),
        FrozenFile(
            "Phase-A manifest sidecar", sidecar_path, sha256_path(sidecar_path)
        ),
    ]
    construction = _mapping(manifest.get("construction_inputs"), "construction_inputs")
    artifact_record, artifact_path = _validate_file_record(
        manifest_dir,
        construction.get("artifact"),
        label="construction_inputs.artifact",
        require_relative=False,
        require_read_only=False,
        frozen_files=frozen_files,
    )
    artifact_sha256 = artifact_record["sha256"]
    if construction.get("artifact_sha256") != artifact_sha256:
        raise ValueError("source artifact checksum drifted")
    (
        artifact,
        directed_adjacency,
        vectors,
        labels,
        _entry_point,
        edge_left,
        edge_right,
        navigator_sha256,
    ) = load_upper(artifact_path)
    if (
        artifact.get("layout_sha256") != phase_a_contract.NEUTRAL_LAYOUT_SHA256
        or artifact.get("physical_point_count") != artifact.get("logical_point_count")
    ):
        raise ValueError("Phase-A artifact is not a neutral upper-only projection")
    upper_nodes = artifact.get("upper_nodes")
    if not isinstance(upper_nodes, list) or any(
        not isinstance(node, dict)
        or node.get("shard_membership")
        != phase_a_contract.NEUTRAL_SHARD_MEMBERSHIP
        for node in upper_nodes
    ):
        raise ValueError("Phase-A artifact exposes a historical shard membership")
    upper_count = len(labels)
    dimension = int(vectors.shape[1])
    normalized_sets = [set(row) for row in directed_adjacency]
    for node, row in enumerate(directed_adjacency):
        for neighbor in row:
            normalized_sets[neighbor].add(node)
    adjacency = [sorted(row) for row in normalized_sets]
    construction_expected = {
        "generation": int(artifact["generation"]),
        "logical_point_count": int(artifact["logical_point_count"]),
        "upper_node_count": upper_count,
        "dimension": dimension,
        "source_upper_k": int(artifact["upper_k"]),
        "source_upper_ef_search": int(artifact["upper_ef_search"]),
        "upper_graph_sha256": canonical_sha256(artifact["upper_graph"]),
        "navigator_sha256": navigator_sha256,
        "upper_navigation_top_k": UPPER_NAVIGATION_TOP_K,
        "upper_navigation_search_ef": UPPER_NAVIGATION_SEARCH_EF,
    }
    errors = _mismatches(construction, construction_expected)
    if errors:
        raise ValueError(f"Phase-A construction identity drifted: {errors}")
    artifact_record_expected = {
        "generation": int(artifact["generation"]),
        "logical_point_count": int(artifact["logical_point_count"]),
        "shard_count": int(artifact["shard_count"]),
    }
    errors = _mismatches(artifact_record, artifact_record_expected)
    if errors:
        raise ValueError(f"Phase-A artifact record drifted: {errors}")

    projection_binding = _mapping(
        construction.get("upper_only_projection"),
        "construction_inputs.upper_only_projection",
    )
    projection_manifest_path = Path(
        projection_binding.get("manifest_path", "")
    ).expanduser().resolve()
    if not projection_manifest_path.is_file():
        raise ValueError("upper-only projection manifest is missing")
    projection_manifest_sha256 = sha256_path(projection_manifest_path)
    if (
        projection_binding.get("manifest_sha256")
        != projection_manifest_sha256
        or projection_binding.get("manifest_size_bytes")
        != projection_manifest_path.stat().st_size
    ):
        raise ValueError("upper-only projection manifest binding drifted")
    _require_read_only(projection_manifest_path, "upper-only projection manifest")
    projection_sidecar = projection_manifest_path.with_name(
        projection_manifest_path.name + ".sha256"
    )
    if (
        not projection_sidecar.is_file()
        or projection_sidecar.read_text(encoding="ascii").strip()
        != projection_manifest_sha256
    ):
        raise ValueError("upper-only projection manifest sidecar mismatch")
    _require_read_only(projection_sidecar, "upper-only projection manifest sidecar")
    frozen_files.extend(
        (
            FrozenFile(
                "upper-only projection manifest",
                projection_manifest_path,
                projection_manifest_sha256,
            ),
            FrozenFile(
                "upper-only projection manifest sidecar",
                projection_sidecar,
                sha256_path(projection_sidecar),
            ),
        )
    )
    projection_manifest = json.loads(
        projection_manifest_path.read_text(encoding="utf-8")
    )
    projection_expected = {
        "format_version": 1,
        "record_type": phase_a_contract.UPPER_ONLY_PROJECTION_RECORD_TYPE,
        "contract": projection_binding.get("contract"),
        "source_artifact": projection_binding.get("source_artifact"),
        "upper_build_provenance": projection_binding.get(
            "upper_build_provenance"
        ),
        "projection_source": projection_binding.get("projection_source"),
        "identity": projection_binding.get("identity"),
        "redaction_proof": projection_binding.get("redaction_proof"),
    }
    errors = _mismatches(projection_manifest, projection_expected)
    if errors:
        raise ValueError(f"upper-only projection manifest drifted: {errors}")
    projection_output = _mapping(
        projection_manifest.get("output_artifact"),
        "upper-only projection output artifact",
    )
    if (
        Path(projection_output.get("path", "")).expanduser().resolve()
        != artifact_path
        or projection_output.get("sha256") != artifact_sha256
        or projection_output.get("size_bytes") != artifact_path.stat().st_size
    ):
        raise ValueError("upper-only projection output binding drifted")
    projection_source = _mapping(
        projection_manifest.get("projection_source"),
        "upper-only projection source",
    )
    current_projection_source = phase_a_contract.UPPER_ONLY_PROJECTION_SCRIPT.resolve()
    if (
        projection_source.get("path") != str(current_projection_source)
        or projection_source.get("sha256") != sha256_path(current_projection_source)
        or projection_source.get("size_bytes") != current_projection_source.stat().st_size
    ):
        raise ValueError("upper-only projection source drifted after freeze")
    if int(artifact["logical_point_count"]) != int(args.row_count):
        raise ValueError("row-count differs from the frozen source artifact")

    labels_record, labels_path = _validate_file_record(
        manifest_dir,
        construction.get("ordered_labels"),
        label="construction_inputs.ordered_labels",
        require_relative=False,
        require_read_only=False,
        frozen_files=frozen_files,
    )
    vectors_record, vectors_path = _validate_file_record(
        manifest_dir,
        construction.get("ordered_vectors"),
        label="construction_inputs.ordered_vectors",
        require_relative=False,
        require_read_only=False,
        frozen_files=frozen_files,
    )
    labels_sha256 = labels_record["sha256"]
    vectors_sha256 = vectors_record["sha256"]
    if construction.get("ordered_labels_sha256") != labels_sha256:
        raise ValueError("ordered upper-label checksum drifted")
    if construction.get("ordered_vectors_sha256") != vectors_sha256:
        raise ValueError("ordered upper-vector checksum drifted")
    if (
        labels_record.get("dtype") != phase_a_contract.MASS_DTYPE
        or labels_record.get("row_count") != upper_count
        or vectors_record.get("dtype") != "<f4"
        or vectors_record.get("row_count") != upper_count
        or vectors_record.get("dimension") != dimension
    ):
        raise ValueError("ordered upper input shape or dtype drifted")
    frozen_labels = checked_binary_matrix(
        labels_path, rows=upper_count, width=1, dtype="<u8"
    ).reshape(-1)
    frozen_vectors = checked_binary_matrix(
        vectors_path, rows=upper_count, width=dimension, dtype="<f4"
    )
    if not np.array_equal(frozen_labels, labels.astype("<u8", copy=False)):
        raise ValueError("ordered upper labels differ from artifact bytes")
    if not np.array_equal(frozen_vectors, vectors.astype("<f4", copy=False)):
        raise ValueError("ordered upper vectors differ from artifact bytes")

    upper_input_record, upper_input_path = _validate_file_record(
        manifest_dir,
        construction.get("upper_input_manifest"),
        label="construction_inputs.upper_input_manifest",
        require_relative=False,
        require_read_only=False,
        frozen_files=frozen_files,
    )
    upper_input_sha256 = upper_input_record["sha256"]
    if construction.get("upper_input_manifest_sha256") != upper_input_sha256:
        raise ValueError("upper-input manifest checksum drifted")
    upper_input = json.loads(upper_input_path.read_text(encoding="utf-8"))
    upper_input_expected = {
        "format_version": 1,
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "generation": int(artifact["generation"]),
        "row_count": upper_count,
        "dimension": dimension,
        "vectors": str(vectors_path),
        "vectors_sha256": vectors_sha256,
        "vectors_size_bytes": upper_count * dimension * 4,
        "labels": str(labels_path),
        "labels_sha256": labels_sha256,
        "labels_size_bytes": upper_count * 8,
    }
    errors = _mismatches(upper_input, upper_input_expected)
    if errors:
        raise ValueError(f"upper-input manifest drifted: {errors}")

    self_record, self_path = _validate_file_record(
        manifest_dir,
        construction.get("self_navigation"),
        label="construction_inputs.self_navigation",
        require_relative=False,
        require_read_only=False,
        frozen_files=frozen_files,
    )
    self_sha256 = self_record["sha256"]
    if Path(construction.get("upper_navigation_hits", "")).expanduser().resolve() != self_path:
        raise ValueError("upper self-navigation hit path alias drifted")
    if construction.get("upper_navigation_hits_sha256") != self_sha256:
        raise ValueError("upper self-navigation hit checksum drifted")
    self_manifest_path = Path(self_record.get("manifest_path", "")).expanduser().resolve()
    self_manifest_sha256 = sha256_path(self_manifest_path)
    if Path(construction.get("upper_navigation_manifest", "")).expanduser().resolve() != self_manifest_path:
        raise ValueError("upper self-navigation manifest path alias drifted")
    if construction.get("upper_navigation_manifest_sha256") != self_manifest_sha256:
        raise ValueError("upper self-navigation manifest checksum drifted")
    if self_record.get("manifest_sha256") != self_manifest_sha256:
        raise ValueError("upper self-navigation nested manifest checksum drifted")
    if (
        self_record.get("dtype") != phase_a_contract.MASS_DTYPE
        or self_record.get("row_count") != upper_count
        or self_record.get("top_k") != UPPER_NAVIGATION_TOP_K
        or self_record.get("search_ef") != UPPER_NAVIGATION_SEARCH_EF
    ):
        raise ValueError("upper self-navigation shape or dtype drifted")
    frozen_files.append(
        FrozenFile(
            "upper self-navigation manifest",
            self_manifest_path,
            self_manifest_sha256,
        )
    )
    self_manifest = json.loads(self_manifest_path.read_text(encoding="utf-8"))
    _validate_self_navigation_manifest(
        self_manifest,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        vectors_sha256=vectors_sha256,
        hits_path=self_path,
        hits_sha256=self_sha256,
        upper_count=upper_count,
        dimension=dimension,
    )
    self_hits = checked_binary_matrix(
        self_path, rows=upper_count, width=UPPER_NAVIGATION_TOP_K, dtype="<u8"
    )
    self_local = local_hits(self_hits, labels, upper_count)
    tie_records, self_first_count, tie_sha256 = _duplicate_tie_proof(
        self_local, labels, vectors
    )

    source_code = _mapping(manifest.get("source_code"), "source_code")
    if set(source_code) != {
        "core",
        "generator",
        "record",
        "supporting_files",
        "protocol_addendum_external_sources",
    }:
        raise ValueError("Phase-A source-code schema drifted")
    source_record_binding, source_record_path = _validate_file_record(
        manifest_dir,
        source_code["record"],
        label="source_code.record",
        require_relative=True,
        require_read_only=True,
        frozen_files=frozen_files,
    )
    supporting_files = _mapping(
        source_code.get("supporting_files"), "source_code.supporting_files"
    )
    source_files = {
        "core": _mapping(source_code.get("core"), "source_code.core"),
        "generator": _mapping(source_code.get("generator"), "source_code.generator"),
        **supporting_files,
    }
    expected_source_names = {
        "core",
        "generator",
        "upper_only_projection",
        "protocol_addendum_json",
        "protocol_addendum_markdown",
    }
    if set(source_files) != expected_source_names:
        raise ValueError("Phase-A copied source set drifted")
    current_sources = {
        "core": Path(phase_a_contract.core.__file__).resolve(),
        "generator": Path(phase_a_contract.__file__).resolve(),
        "upper_only_projection": phase_a_contract.UPPER_ONLY_PROJECTION_SCRIPT.resolve(),
        "protocol_addendum_json": phase_a_contract.PROTOCOL_ADDENDUM_JSON.resolve(),
        "protocol_addendum_markdown": phase_a_contract.PROTOCOL_ADDENDUM_MD.resolve(),
    }
    for name, current_path in current_sources.items():
        copied_record, _copied_path = _validate_file_record(
            manifest_dir,
            source_files[name],
            label=f"source_code.{name}",
            require_relative=True,
            require_read_only=True,
            frozen_files=frozen_files,
        )
        if copied_record["sha256"] != sha256_path(current_path):
            raise ValueError(f"Phase-A producer/source {name} drifted after freeze")
    source_record = json.loads(source_record_path.read_text(encoding="utf-8"))
    source_record_expected = {
        "format_version": 1,
        "record_type": "phase_a_source_code",
        "files": source_files,
        "protocol_contract_id": "cnbr-replacement-correction-v1",
        "proxy_mass_contract_id": MASS_MODE,
    }
    errors = _mismatches(source_record, source_record_expected)
    if errors:
        raise ValueError(f"Phase-A source-code record drifted: {errors}")
    protocol_sources = _mapping(
        source_code.get("protocol_addendum_external_sources"),
        "source_code.protocol_addendum_external_sources",
    )
    protocol_expected = {
        "json_path": str(phase_a_contract.PROTOCOL_ADDENDUM_JSON.resolve()),
        "json_sha256": sha256_path(phase_a_contract.PROTOCOL_ADDENDUM_JSON),
        "markdown_path": str(phase_a_contract.PROTOCOL_ADDENDUM_MD.resolve()),
        "markdown_sha256": sha256_path(phase_a_contract.PROTOCOL_ADDENDUM_MD),
    }
    if protocol_sources != protocol_expected:
        raise ValueError("Phase-A protocol source binding drifted")

    mass = _mapping(manifest.get("mass"), "mass")
    mass_expected = {
        "proxy_mass_contract_id": MASS_MODE,
        "source": MASS_SOURCE,
        "transform": MASS_TRANSFORM,
        "estimator_version": MASS_ESTIMATOR_VERSION,
        "query_count": upper_count,
        "top_k": UPPER_NAVIGATION_TOP_K,
        "total_mass": upper_count * UPPER_NAVIGATION_TOP_K,
        "duplicate_tie_exception_count": len(tie_records),
        "duplicate_tie_proof_sha256": tie_sha256,
    }
    errors = _mismatches(mass, mass_expected)
    if errors:
        raise ValueError(f"navigation proxy-mass contract drifted: {errors}")
    mass_record_binding, mass_record_path = _validate_file_record(
        manifest_dir,
        mass.get("record"),
        label="mass.record",
        require_relative=True,
        require_read_only=True,
        frozen_files=frozen_files,
    )
    mass_value_record, mass_path = _validate_file_record(
        manifest_dir,
        mass.get("values"),
        label="mass.values",
        require_relative=True,
        require_read_only=True,
        frozen_files=frozen_files,
    )
    if (
        mass_value_record.get("dtype") != schema["mass_dtype"]
        or mass_value_record.get("encoding") != schema["mass_encoding"]
        or mass_value_record.get("row_count") != upper_count
        or mass_value_record.get("count") != upper_count
    ):
        raise ValueError("navigation proxy-mass shape or dtype drifted")
    mass_values = checked_binary_matrix(
        mass_path, rows=upper_count, width=1, dtype="<u8"
    ).reshape(-1)
    mass_values = np.asarray(mass_values, dtype=np.uint64).copy()
    replayed_mass = np.bincount(
        np.asarray(self_local).reshape(-1), minlength=upper_count
    ).astype(np.uint64)
    if not np.array_equal(mass_values, replayed_mass):
        raise ValueError("navigation mass does not replay from frozen self hits")
    if mass.get("semantic_sha256") != _semantic_mass_sha256(mass_values):
        raise ValueError("navigation proxy-mass semantic checksum drifted")
    mass_record = json.loads(mass_record_path.read_text(encoding="utf-8"))
    mass_record_expected = {
        "format_version": 1,
        "record_type": "upper_self_navigation_proxy_mass",
        "proxy_mass_contract_id": MASS_MODE,
        "source": MASS_SOURCE,
        "transform": MASS_TRANSFORM,
        "estimator_version": MASS_ESTIMATOR_VERSION,
        "semantic_sha256": mass["semantic_sha256"],
        "values": mass["values"],
        "query_count": upper_count,
        "top_k": UPPER_NAVIGATION_TOP_K,
        "search_ef": UPPER_NAVIGATION_SEARCH_EF,
        "total_mass": int(mass_values.sum()),
        "vertex_mass_min": int(mass_values.min()),
        "vertex_mass_max": int(mass_values.max()),
        "vertex_mass_mean": float(mass_values.mean()),
        "self_present_count": upper_count,
        "self_first_count": self_first_count,
        "duplicate_tie_exception_count": len(tie_records),
        "duplicate_tie_exceptions": tie_records,
        "duplicate_tie_proof_sha256": tie_sha256,
        "upper_navigation_hits": str(self_path),
        "upper_navigation_hits_sha256": self_sha256,
        "upper_navigation_manifest": str(self_manifest_path),
        "upper_navigation_manifest_sha256": self_manifest_sha256,
        "source_code_record_sha256": source_record_binding["sha256"],
    }
    errors = _mismatches(mass_record, mass_record_expected)
    if errors:
        raise ValueError(f"navigation proxy-mass record drifted: {errors}")

    num_partitions = int(
        _mapping(manifest.get("fixed_parameters"), "fixed_parameters").get(
            "num_partitions", 0
        )
    )
    if num_partitions != int(artifact["shard_count"]):
        raise ValueError("Phase-A partition count differs from the source artifact")
    _validate_fixed_parameters(
        manifest["fixed_parameters"], num_partitions, candidate_set
    )
    _validate_graph_gate_contract(manifest.get("graph_gate_contract"))

    owner_records = _mapping(manifest.get("owners"), "owners")
    expected_owner_names = {REFERENCE_NAME, *candidate_ratios}
    if set(owner_records) != expected_owner_names:
        raise ValueError("Phase-A owners differ from the exact frozen candidate set")
    owners = {
        name: _validate_owner_record(
            name=name,
            expected_role=schema["owner_roles"][name],
            owner_dtype=schema["owner_dtype"],
            owner_encoding=schema["owner_encoding"],
            estimated_mass_field=schema["estimated_partition_mass_field"],
            value=owner_records[name],
            manifest_dir=manifest_dir,
            upper_count=upper_count,
            num_partitions=num_partitions,
            mass_values=mass_values,
            edge_left=edge_left,
            edge_right=edge_right,
            frozen_files=frozen_files,
        )
        for name in (REFERENCE_NAME, *candidate_ratios)
    }
    reference = owners[REFERENCE_NAME]
    reference_gates = graph_gate_results(
        reference.record["topology_metrics"], reference.record["topology_metrics"]
    )
    if not _canonical_close(reference.record.get("topology_gates"), reference_gates):
        raise ValueError("N_native topology reference gates drifted")
    n_owner_record_expected = {
        "source_code_record_sha256": source_record_binding["sha256"],
        "proxy_mass_record_sha256": mass_record_binding["sha256"],
        "algorithm": manifest["fixed_parameters"]["N_native"],
    }
    errors = _mismatches(reference.owner_record, n_owner_record_expected)
    if errors:
        raise ValueError(f"N_native owner-record binding drifted: {errors}")

    for candidate_name, (ratio_numerator, ratio_denominator) in candidate_ratios.items():
        candidate = owners[candidate_name]
        candidate_record = candidate.record
        candidate_id = (
            CANDIDATE_NAME
            if (ratio_numerator, ratio_denominator) == (9, 4)
            else candidate_name
        )
        candidate_expected = {
            "candidate_id": candidate_id,
            "family_member_id": candidate_name,
            "selected_adoption_candidate": candidate_name
            == selected_candidate_name,
            "trigger_ratio": {
                "numerator": ratio_numerator,
                "denominator": ratio_denominator,
            },
            "base_owner_role": REFERENCE_NAME,
            "base_owner_sha256": reference.sha256,
            "parent_n_owner_record_sha256": reference.owner_record_sha256,
            "source_code_record_sha256": source_record_binding["sha256"],
            "proxy_mass_record_sha256": mass_record_binding["sha256"],
        }
        errors = _mismatches(candidate.owner_record, candidate_expected)
        if errors:
            raise ValueError(f"{candidate_name} owner-record binding drifted: {errors}")
        for field in (
            "candidate_id",
            "family_member_id",
            "selected_adoption_candidate",
            "trigger_ratio",
            "base_owner_role",
            "base_owner_sha256",
            "parent_n_owner_record_sha256",
            "round_trace",
            "moved_node_count",
        ):
            if candidate_record.get(field) != candidate.owner_record.get(field):
                raise ValueError(f"{candidate_name} manifest/owner-record {field} drifted")
        moved_nodes = np.flatnonzero(candidate.values != reference.values).tolist()
        if candidate_record.get("moved_node_count") != len(moved_nodes):
            raise ValueError(f"{candidate_name} moved-node count drifted")
        rounds_binding, trace_path = _validate_file_record(
            manifest_dir,
            candidate_record.get("round_trace"),
            label=f"owners.{candidate_name}.round_trace",
            require_relative=True,
            require_read_only=True,
            frozen_files=frozen_files,
        )
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        _validate_cnbr_trace(
            name=candidate_name,
            trace=trace,
            rounds_record=rounds_binding,
            reference_owner=reference.values,
            candidate_owner=candidate.values,
            mass_values=mass_values,
            adjacency=adjacency,
            edge_left=edge_left,
            edge_right=edge_right,
            num_partitions=num_partitions,
            ratio_numerator=ratio_numerator,
            ratio_denominator=ratio_denominator,
            reference_metrics=reference.record["topology_metrics"],
        )
        expected_gates = graph_gate_results(
            candidate_record["topology_metrics"],
            reference.record["topology_metrics"],
        )
        if not _canonical_close(expected_gates, candidate_record.get("topology_gates")):
            raise ValueError(f"{candidate_name} graph topology gates drifted")
        if candidate_record.get("topology_all_pass") is not all(
            bool(gate["pass"]) for gate in expected_gates.values()
        ):
            raise ValueError(f"{candidate_name} topology_all_pass drifted")

    outputs = _mapping(manifest.get("outputs"), "outputs")
    expected_outputs = {
        "mass_values": mass["values"]["path"],
        "N_native_owner": owner_records[REFERENCE_NAME]["owner"]["path"],
        "source_core": source_files["core"]["path"],
        "source_generator": source_files["generator"]["path"],
        **{
            f"{name}_owner": owner_records[name]["owner"]["path"]
            for name in candidate_ratios
        },
        "owners_dir": "owners",
        "records_dir": "records",
        "traces_dir": "traces",
        "source_dir": "source",
        "performance_record": {"path": phase_a_contract.PERFORMANCE_NAME},
    }
    if set(outputs) != set(schema["output_keys"]) or outputs != expected_outputs:
        raise ValueError("Phase-A outputs contract drifted")
    performance_path = (
        manifest_dir / outputs["performance_record"]["path"]
    ).resolve()
    _require_read_only(performance_path, "Phase-A performance record")
    performance_sha256 = sha256_path(performance_path)
    frozen_files.append(
        FrozenFile("Phase-A performance record", performance_path, performance_sha256)
    )
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    performance_expected = {
        "format_version": 1,
        "record_type": "phase_a_construction_performance",
        "phase_a_manifest": manifest_path.name,
        "phase_a_manifest_sha256": manifest_sha256,
        "candidate_set": candidate_set,
        "source_artifact_sha256": artifact_sha256,
    }
    errors = _mismatches(performance, performance_expected)
    if errors:
        raise ValueError(f"Phase-A performance binding drifted: {errors}")
    performance_owners = _mapping(
        performance.get("owners"), "performance.owners"
    )
    if set(performance_owners) != expected_owner_names:
        raise ValueError("Phase-A performance owner set drifted")
    for name, owner in owners.items():
        perf = _mapping(performance_owners[name], f"performance.owners.{name}")
        if perf.get("owner_sha256") != owner.sha256 or perf.get(
            "owner_record_sha256"
        ) != owner.owner_record_sha256:
            raise ValueError(f"Phase-A performance binding for {name} drifted")

    return FrozenPhaseA(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        artifact_path=artifact_path,
        artifact=artifact,
        labels=labels,
        vectors=vectors,
        edge_left=edge_left,
        edge_right=edge_right,
        num_partitions=num_partitions,
        candidate_set=candidate_set,
        candidate_ratios=dict(candidate_ratios),
        selected_candidate_name=selected_candidate_name,
        mass_values=mass_values,
        owners=owners,
        performance_path=performance_path,
        performance_sha256=performance_sha256,
        performance=performance,
        frozen_files=tuple(frozen_files),
    )


def open_phase_b_inputs(
    args: argparse.Namespace, frozen: FrozenPhaseA
) -> PhaseBInputs:
    """Open L0/query inputs only after :func:`validate_phase_a` succeeds."""

    dataset_manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    dataset_manifest_sha256 = sha256_path(dataset_manifest_path)
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset = _mapping(dataset_manifest.get("dataset"), "dataset_manifest.dataset")
    source_dataset = _mapping(
        dataset_manifest.get("source_dataset"), "dataset_manifest.source_dataset"
    )
    source_dataset_record = _mapping(
        source_dataset.get("dataset"), "dataset_manifest.source_dataset.dataset"
    )
    dataset_sha256 = dataset.get("sha256")
    if (
        not isinstance(dataset_sha256, str)
        or source_dataset.get("sha256") != dataset_sha256
        or source_dataset_record.get("sha256") != dataset_sha256
    ):
        raise ValueError("dataset manifest has inconsistent source dataset checksums")
    source_dataset_path = Path(source_dataset.get("path", "")).expanduser().resolve()
    if sha256_path(source_dataset_path) != dataset_sha256:
        raise ValueError("source dataset checksum drifted")
    dimension = int(dataset.get("dimension", 0))
    expected_train_rows = int(dataset.get("expected_train_rows", 0))
    expected_query_rows = int(dataset.get("expected_query_rows", 0))
    if (
        dimension != int(frozen.vectors.shape[1])
        or expected_train_rows != int(args.row_count)
        or expected_query_rows <= 0
    ):
        raise ValueError("dataset manifest shape differs from the frozen artifact")
    dataset_files = _mapping(dataset_manifest.get("files"), "dataset_manifest.files")
    for name in ("vectors", "queries", "ground_truth"):
        record = _mapping(dataset_files.get(name), f"dataset_manifest.files.{name}")
        path = Path(record.get("path", "")).expanduser().resolve()
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"dataset manifest {name} file size drifted")
        if name != "vectors" and sha256_path(path) != record.get("sha256"):
            raise ValueError(f"dataset manifest {name} file checksum drifted")

    source_build_manifest_path = Path(
        args.source_build_manifest
    ).expanduser().resolve()
    source_build_manifest_sha256 = sha256_path(source_build_manifest_path)
    source_build = json.loads(
        source_build_manifest_path.read_text(encoding="utf-8")
    )
    source_is_binding = (
        source_build.get("record_type") == source_binding_contract.RECORD_TYPE
    )
    if not source_is_binding:
        raise ValueError(
            "Phase B requires a frozen source-build checksum binding with "
            "replayable vector lineage"
        )
    if source_is_binding:
        sidecar_path = source_build_manifest_path.with_name(
            source_build_manifest_path.name + ".sha256"
        )
        if (
            source_build.get("format_version")
            != source_binding_contract.FORMAT_VERSION
            or source_build.get("contract")
            != source_binding_contract.binding_contract()
            or not sidecar_path.is_file()
            or sidecar_path.read_text(encoding="ascii").strip()
            != source_build_manifest_sha256
        ):
            raise ValueError("source-build checksum binding contract drifted")
        _require_read_only(source_build_manifest_path, "source-build binding")
        _require_read_only(sidecar_path, "source-build binding sidecar")
        binding_dataset = _mapping(
            source_build.get("dataset"), "source build binding dataset"
        )
        binding_expected = {
            "manifest_path": str(dataset_manifest_path),
            "manifest_sha256": dataset_manifest_sha256,
            "sha256": dataset_sha256,
            "dimension": dimension,
            "train_rows_used": int(args.row_count),
        }
        errors = _mismatches(binding_dataset, binding_expected)
        if errors:
            raise ValueError(f"source-build binding dataset drifted: {errors}")
        binding_outputs = _mapping(
            source_build.get("outputs"), "source build binding outputs"
        )
        binding_files = _mapping(
            binding_outputs.get("files"), "source build binding output files"
        )
        binding_vectors_name = binding_outputs.get("vectors")
        if not isinstance(binding_vectors_name, str):
            raise ValueError("source-build binding does not name full L0 vectors")
        binding_vectors_record = _mapping(
            binding_files.get(binding_vectors_name),
            "source-build binding full-vector output",
        )
        binding_vectors_path = Path(
            str(binding_vectors_record.get("path", ""))
        ).expanduser().resolve()
        lineage = _mapping(
            source_build.get("vector_lineage"), "source-build binding vector lineage"
        )
        lineage_source_build = _mapping(
            lineage.get("source_build_manifest"),
            "source-build binding lineage source manifest",
        )
        lineage_projection = _mapping(
            lineage.get("upper_only_projection_manifest"),
            "source-build binding lineage projection manifest",
        )
        lineage_source_build_path = Path(
            str(lineage_source_build.get("path", ""))
        ).expanduser().resolve()
        lineage_projection_path = Path(
            str(lineage_projection.get("path", ""))
        ).expanduser().resolve()
        expected_lineage = source_binding_contract.validate_source_build_vector_lineage(
            dataset_manifest_path=dataset_manifest_path,
            dataset_manifest=dataset_manifest,
            artifact_path=frozen.artifact_path,
            artifact=frozen.artifact,
            vectors_path=binding_vectors_path,
            source_build_manifest_path=lineage_source_build_path,
            projection_manifest_path=lineage_projection_path,
        )
        if lineage != expected_lineage:
            raise ValueError("source-build binding vector lineage drifted")
        phase_a_projection = _mapping(
            _mapping(
                frozen.manifest.get("construction_inputs"),
                "Phase-A construction inputs",
            ).get("upper_only_projection"),
            "Phase-A upper-only projection binding",
        )
        if (
            Path(
                str(phase_a_projection.get("manifest_path", ""))
            ).expanduser().resolve()
            != lineage_projection_path
            or phase_a_projection.get("manifest_sha256")
            != lineage_projection.get("sha256")
            or phase_a_projection.get("manifest_size_bytes")
            != lineage_projection.get("size_bytes")
        ):
            raise ValueError(
                "source-build binding projection differs from frozen Phase-A"
            )
    if (_mapping(source_build.get("dataset"), "source build dataset")).get(
        "sha256"
    ) != dataset_sha256:
        raise ValueError("source build manifest uses another dataset checksum")
    build_outputs = _mapping(
        _mapping(source_build.get("outputs"), "source build outputs").get("files"),
        "source build output files",
    )
    artifact_output = _mapping(
        build_outputs.get(frozen.artifact_path.name),
        "source build artifact output",
    )
    if artifact_output.get("sha256") != sha256_path(frozen.artifact_path):
        raise ValueError("source build manifest does not bind the Phase-A artifact")
    if source_is_binding and Path(
        artifact_output.get("path", "")
    ).expanduser().resolve() != frozen.artifact_path:
        raise ValueError("source-build binding artifact path drifted")

    attachments_path = Path(args.attachments).expanduser().resolve()
    attachments_manifest_path = Path(args.attachments_manifest).expanduser().resolve()
    attachments_manifest = json.loads(
        attachments_manifest_path.read_text(encoding="utf-8")
    )
    attachments_sha256 = sha256_path(attachments_path)
    attachment_expected = {
        "artifact_sha256": sha256_path(frozen.artifact_path),
        "row_count": int(args.row_count),
        "top_k": int(args.attachment_k),
        "search_ef": ATTACHMENT_SEARCH_EF,
        "hits_sha256": attachments_sha256,
    }
    errors = _mismatches(attachments_manifest, attachment_expected)
    if errors:
        raise ValueError(f"attachment manifest mismatch: {errors}")
    if Path(attachments_manifest.get("artifact_path", "")).resolve() != frozen.artifact_path:
        raise ValueError("attachment manifest artifact path drifted")
    if int(attachments_manifest.get("dimension", 0)) != dimension:
        raise ValueError("attachment manifest dimension drifted")
    attachment_vectors_path = Path(
        attachments_manifest.get("vectors_path", "")
    ).expanduser().resolve()
    attachment_vectors_sha256 = attachments_manifest.get("vectors_sha256")
    vector_output = _mapping(
        build_outputs.get(attachment_vectors_path.name),
        "source build vector output",
    )
    if vector_output.get("sha256") != attachment_vectors_sha256:
        raise ValueError("source build and attachment vector checksums differ")
    if source_is_binding and Path(
        vector_output.get("path", "")
    ).expanduser().resolve() != attachment_vectors_path:
        raise ValueError("source-build binding vector path drifted")
    if sha256_path(attachment_vectors_path) != attachment_vectors_sha256:
        raise ValueError("attachment source vector checksum drifted")
    attachment_hits = checked_binary_matrix(
        attachments_path,
        rows=int(args.row_count),
        width=int(args.attachment_k),
        dtype="<u8",
    )
    attachment_local = local_hits(
        attachment_hits, frozen.labels, int(args.row_count)
    )

    query_hits_path = Path(args.query_hits).expanduser().resolve()
    query_manifest_path = Path(args.query_hits_manifest).expanduser().resolve()
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    query_rows = int(query_manifest.get("row_count", 0))
    query_k = int(query_manifest.get("top_k", 0))
    if query_rows <= 0 or query_k <= 0:
        raise ValueError("query-hit manifest has an invalid shape")
    query_hits_sha256 = sha256_path(query_hits_path)
    query_expected = {
        "artifact_sha256": sha256_path(frozen.artifact_path),
        "hits_sha256": query_hits_sha256,
    }
    errors = _mismatches(query_manifest, query_expected)
    if errors:
        raise ValueError(f"query-hit manifest mismatch: {errors}")
    if Path(query_manifest.get("artifact_path", "")).resolve() != frozen.artifact_path:
        raise ValueError("query-hit manifest artifact path drifted")
    if query_rows != expected_query_rows or int(query_manifest.get("dimension", 0)) != dimension:
        raise ValueError("query-hit manifest differs from the dataset query shape")
    query_vectors = _mapping(
        dataset_files.get("queries"), "dataset_manifest.files.queries"
    )
    if (
        Path(query_manifest.get("vectors_path", "")).resolve()
        != Path(query_vectors["path"]).resolve()
        or query_manifest.get("vectors_sha256") != query_vectors.get("sha256")
    ):
        raise ValueError("query-hit vectors are not the checksum-bound dataset queries")
    query_hits = checked_binary_matrix(
        query_hits_path, rows=query_rows, width=query_k, dtype="<u8"
    )
    query_local = local_hits(query_hits, frozen.labels, query_rows)

    ground_truth_path = Path(args.ground_truth).expanduser().resolve()
    ground_truth_manifest_path = Path(
        args.ground_truth_manifest
    ).expanduser().resolve()
    ground_truth_manifest_sha256 = sha256_path(ground_truth_manifest_path)
    ground_truth_manifest = json.loads(
        ground_truth_manifest_path.read_text(encoding="utf-8")
    )
    ground_truth_sha256 = sha256_path(ground_truth_path)
    dataset_ground_truth = _mapping(
        dataset_files.get("ground_truth"), "dataset_manifest.files.ground_truth"
    )
    ground_truth_expected = {
        "dtype": "<u4",
        "row_count": query_rows,
        "width": int(args.ground_truth_width),
        "sha256": ground_truth_sha256,
        "size_bytes": query_rows * int(args.ground_truth_width) * 4,
        "source_dataset": "neighbors",
    }
    errors = _mismatches(ground_truth_manifest, ground_truth_expected)
    if errors:
        raise ValueError(f"ground-truth manifest mismatch: {errors}")
    if (
        Path(ground_truth_manifest.get("output", "")).resolve() != ground_truth_path
        or dataset_ground_truth.get("sha256") != ground_truth_sha256
        or dataset_ground_truth.get("size_bytes") != ground_truth_path.stat().st_size
    ):
        raise ValueError("ground truth is not the checksum-bound dataset reference")
    if Path(ground_truth_manifest.get("source_hdf5", "")).resolve() != source_dataset_path:
        raise ValueError("ground truth was extracted from another source dataset")
    ground_truth = checked_binary_matrix(
        ground_truth_path,
        rows=query_rows,
        width=int(args.ground_truth_width),
        dtype="<u4",
    )
    if np.any(np.asarray(ground_truth) >= int(args.row_count)):
        raise ValueError("ground truth contains a point outside the full dataset")
    return PhaseBInputs(
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=dataset_manifest_sha256,
        dataset_sha256=dataset_sha256,
        source_build_manifest_path=source_build_manifest_path,
        source_build_manifest_sha256=source_build_manifest_sha256,
        attachments_path=attachments_path,
        attachments_manifest_path=attachments_manifest_path,
        attachments_sha256=attachments_sha256,
        attachment_local=attachment_local,
        query_hits_path=query_hits_path,
        query_manifest_path=query_manifest_path,
        query_hits_sha256=query_hits_sha256,
        query_manifest=query_manifest,
        query_local=query_local,
        ground_truth_path=ground_truth_path,
        ground_truth_manifest_path=ground_truth_manifest_path,
        ground_truth_manifest_sha256=ground_truth_manifest_sha256,
        ground_truth_sha256=ground_truth_sha256,
        ground_truth=ground_truth,
    )


def assignment_views(
    owner: np.ndarray,
    attachment_local: np.ndarray,
    num_partitions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay ordered primary selection and unchanged physical multi-assignment."""

    hit_owner = owner[attachment_local]
    rows = len(hit_owner)
    votes = np.zeros((rows, num_partitions), dtype=np.uint8)
    row_ids = np.arange(rows, dtype=np.int64)
    for column in range(hit_owner.shape[1]):
        np.add.at(votes, (row_ids, hit_owner[:, column]), 1)
    maximum = votes.max(axis=1)
    primary_shards = np.argmax(votes, axis=1).astype(np.int32, copy=False)
    fallback = maximum < 2
    primary_shards[fallback] = hit_owner[fallback, 0]
    primary_loads = np.bincount(
        primary_shards, minlength=num_partitions
    ).astype(np.int64, copy=False)

    membership, copy_count, physical_loads = physical_membership(
        owner, attachment_local, num_partitions
    )
    replay_membership = votes == maximum[:, None]
    replay_membership[fallback] = False
    replay_membership[row_ids[fallback], primary_shards[fallback]] = True
    if not np.array_equal(membership, replay_membership):
        raise AssertionError("multi-assignment replay differs from physical_membership")
    if not np.array_equal(
        primary_shards, np.asarray([np.flatnonzero(row)[0] for row in membership])
    ):
        raise AssertionError("primary shard differs from ordered target-shard semantics")
    return membership, copy_count, physical_loads, primary_shards, primary_loads


def _load_distribution(loads: np.ndarray, prefix: str) -> dict[str, float | int]:
    mean = float(np.mean(loads))
    return {
        f"{prefix}_min": int(np.min(loads)),
        f"{prefix}_max": int(np.max(loads)),
        f"{prefix}_mean": mean,
        f"{prefix}_cv": float(np.std(loads) / mean),
        f"{prefix}_max_over_mean": float(np.max(loads) / mean),
        f"{prefix}_min_over_mean": float(np.min(loads) / mean),
        f"{prefix}_empty_shards": int(np.count_nonzero(loads == 0)),
    }


def _copy_histogram(copy_count: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(copy_count, return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(values.tolist(), counts.tolist(), strict=True)
    }


def _verify_frozen_files_unchanged(frozen: FrozenPhaseA) -> None:
    for record in frozen.frozen_files:
        if sha256_path(record.path) != record.sha256:
            raise ValueError(f"{record.label} changed after the Phase-A audit")
    for owner in frozen.owners.values():
        _require_read_only(owner.path, f"owners.{owner.name}.owner")


def evaluate_phase_b(
    frozen: FrozenPhaseA, inputs: PhaseBInputs
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluations: dict[str, dict[str, Any]] = {}
    owner_names = (REFERENCE_NAME, *frozen.candidate_ratios)
    for name in owner_names:
        owner_info = frozen.owners[name]
        (
            membership,
            copy_count,
            physical_loads,
            primary_shards,
            primary_loads,
        ) = assignment_views(
            owner_info.values, inputs.attachment_local, frozen.num_partitions
        )
        proxy_loads = np.asarray(
            owner_info.record[phase_a_contract.ESTIMATED_PARTITION_MASS_FIELD],
            dtype=np.int64,
        )
        actual_metrics: dict[str, Any] = {
            "logical_point_count": int(len(membership)),
            "physical_point_count": int(np.sum(physical_loads)),
            "expansion_ratio": float(np.mean(copy_count)),
            **_load_distribution(proxy_loads, "proxy_mass_load"),
            **_load_distribution(primary_loads, "primary_load"),
            **_load_distribution(physical_loads, "physical_copy_load"),
        }
        actual_metrics.update(
            query_metrics(
                owner_info.values,
                inputs.query_local,
                membership[frozen.labels],
                membership,
                inputs.ground_truth,
                int(frozen.artifact["dynamic_ef_base"]),
                int(frozen.artifact["dynamic_ef_factor"]),
            )
        )
        copy_histogram = _copy_histogram(copy_count)
        if name == REFERENCE_NAME:
            evaluation_role = "reference"
        elif frozen.candidate_set == "formal":
            evaluation_role = "candidate"
        else:
            evaluation_role = "frozen_family_member"
        evaluations[name] = {
            "name": name,
            "role": evaluation_role,
            "phase_a_role": owner_info.role,
            "owner_sha256": owner_info.sha256,
            "owner_record_sha256": owner_info.owner_record_sha256,
            "selected_adoption_candidate": name
            == frozen.selected_candidate_name,
            "topology_metrics": owner_info.record["topology_metrics"],
            "actual_metrics": actual_metrics,
            "construction_performance": frozen.performance["owners"][name],
            "materialization_parity": {
                "canonical_format": "orion_numeric_import.assignments.jsonl-v1",
                "assignment_bytes_sha256": membership_assignment_sha256(membership),
                "logical_point_count": int(len(membership)),
                "physical_point_count": int(np.sum(physical_loads)),
                "primary_shards_sha256": hashlib.sha256(
                    np.ascontiguousarray(primary_shards, dtype="<i4").tobytes()
                ).hexdigest(),
                "primary_shard_loads": [
                    int(value) for value in primary_loads.tolist()
                ],
                "physical_copy_shard_loads": [
                    int(value) for value in physical_loads.tolist()
                ],
                "copy_count_histogram": copy_histogram,
            },
        }

    reference = evaluations[REFERENCE_NAME]
    reference["graph_topology_gates"] = {}
    reference["query_topology_gates"] = {}
    reference["identity_all_pass"] = True
    reference["graph_topology_all_pass"] = True
    reference["query_topology_all_pass"] = True
    reference["topology_all_pass"] = True
    reference["physical_copy_load_improves_over_reference"] = True
    reference["screen_eligible"] = True
    reference["materialization_eligible"] = True
    reference_physical_max_mean = float(
        reference["actual_metrics"]["physical_copy_load_max_over_mean"]
    )
    for name in frozen.candidate_ratios:
        candidate = evaluations[name]
        graph_gates = graph_gate_results(
            candidate["topology_metrics"], reference["topology_metrics"]
        )
        query_gates = query_gate_results(
            candidate["actual_metrics"],
            reference["actual_metrics"],
            int(inputs.query_manifest["row_count"]),
        )
        graph_all_pass = all(bool(gate["pass"]) for gate in graph_gates.values())
        query_all_pass = all(bool(gate["pass"]) for gate in query_gates.values())
        candidate_physical_max_mean = float(
            candidate["actual_metrics"]["physical_copy_load_max_over_mean"]
        )
        physical_load_gate = {
            "observed": candidate_physical_max_mean,
            "reference": reference_physical_max_mean,
            "operator": "<",
            "threshold": reference_physical_max_mean,
            "pass": candidate_physical_max_mean < reference_physical_max_mean,
        }
        candidate["base_owner_sha256"] = frozen.owners[REFERENCE_NAME].sha256
        candidate["identity_all_pass"] = True
        candidate["graph_topology_gates"] = graph_gates
        candidate["query_topology_gates"] = query_gates
        candidate["graph_topology_all_pass"] = graph_all_pass
        candidate["query_topology_all_pass"] = query_all_pass
        candidate["topology_all_pass"] = graph_all_pass and query_all_pass
        candidate["physical_copy_load_gate"] = physical_load_gate
        candidate["physical_copy_load_improves_over_reference"] = bool(
            physical_load_gate["pass"]
        )
        candidate["screen_eligible"] = bool(
            candidate["identity_all_pass"]
            and candidate["graph_topology_all_pass"]
            and candidate["query_topology_all_pass"]
            and candidate["physical_copy_load_improves_over_reference"]
        )
        candidate["materialization_eligible"] = bool(
            candidate["screen_eligible"]
            and candidate["selected_adoption_candidate"]
        )

    _verify_frozen_files_unchanged(frozen)
    rows: list[dict[str, Any]] = []
    for name in owner_names:
        owner_record = frozen.owners[name].record
        evaluation = evaluations[name]
        masses = np.asarray(
            owner_record[phase_a_contract.ESTIMATED_PARTITION_MASS_FIELD],
            dtype=np.int64,
        )
        mass_mean = float(np.mean(masses))
        performance_fields = {
            f"construction_{key}": value
            for key, value in evaluation["construction_performance"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        rows.append(
            {
                "name": name,
                "role": evaluation["role"],
                "selected_adoption_candidate": evaluation[
                    "selected_adoption_candidate"
                ],
                "materialization_eligible": evaluation["materialization_eligible"],
                "identity_all_pass": evaluation["identity_all_pass"],
                "graph_topology_all_pass": evaluation["graph_topology_all_pass"],
                "query_topology_all_pass": evaluation["query_topology_all_pass"],
                "topology_all_pass": evaluation["topology_all_pass"],
                "physical_copy_load_improves_over_reference": evaluation[
                    "physical_copy_load_improves_over_reference"
                ],
                "screen_eligible": evaluation["screen_eligible"],
                "l1_min_size": int(min(owner_record["partition_sizes"])),
                "l1_max_size": int(max(owner_record["partition_sizes"])),
                "estimated_mass_min": int(np.min(masses)),
                "estimated_mass_max": int(np.max(masses)),
                "estimated_mass_mean": mass_mean,
                "estimated_mass_cv": float(np.std(masses) / mass_mean),
                "estimated_mass_max_over_mean": float(np.max(masses) / mass_mean),
                **evaluation["topology_metrics"],
                **evaluation["actual_metrics"],
                **performance_fields,
            }
        )

    candidate_records = {
        name: evaluations[name] for name in frozen.candidate_ratios
    }
    screen_manifest = {
        "format_version": 1,
        "stage": PHASE_B_STAGE,
        "candidate_set": frozen.candidate_set,
        "comparison": (
            "N_native_vs_C_CNBR"
            if frozen.candidate_set == "formal"
            else "N_native_vs_frozen_CNBR_ratio_family"
        ),
        "contract": {
            **frozen.manifest["contract"],
            "phase_a_fully_verified_before_l0_open": True,
            "evaluator_runs_in_separate_process": True,
            "owners_changed_after_freeze": False,
            "l0_assignment_after_owner_freeze": True,
            "l0_repair_or_flow": False,
            "multi_assignment": MULTI_ASSIGNMENT_CONTRACT,
            "load_balance_cannot_override_topology_failure": True,
        },
        "phase_a_manifest": str(frozen.manifest_path),
        "phase_a_manifest_sha256": frozen.manifest_sha256,
        "identity_gates": {
            "phase_a_manifest_sidecar": True,
            "phase_a_source_code_sha256": True,
            "source_artifact_sha256": True,
            "navigator_sha256": True,
            "upper_graph_sha256": True,
            "ordered_labels_sha256": True,
            "ordered_vectors_sha256": True,
            "upper_self_navigation_replay": True,
            "navigation_mass_replay": True,
            "N_native_owner_sha256": True,
            "all_candidate_owner_sha256": True,
            "all_candidate_base_owner_sha256": True,
            "owners_read_only": True,
            "owners_unchanged_during_evaluation": True,
            "dataset_sha256": True,
            "source_build_artifact_and_vectors_sha256": True,
            "attachment_vector_sha256": True,
            "query_vector_sha256": True,
            "ground_truth_sha256": True,
        },
        "identity_all_pass": True,
        "phase_a_performance_record": str(frozen.performance_path),
        "phase_a_performance_record_sha256": frozen.performance_sha256,
        "phase_a_shared_construction_performance": frozen.performance["shared"],
        "graph_gate_contract": frozen.manifest["graph_gate_contract"],
        "search_induced_gate_contract": {
            "version": "l1-search-induced-preregistered-v1",
            "reference": REFERENCE_NAME,
            "mean_route_metric_max_ratio": 1.05,
            "gt_coverage_mean_max_drop": 0.002,
            "gt_full_coverage_fraction_max_drop": 0.02,
            "gt_zero_coverage_max_increase": 2,
            "gt_zero_coverage_absolute_fraction_max": 0.001,
            "all_gates_required": True,
        },
        "post_freeze_evaluation_inputs": {
            "dataset_manifest": str(inputs.dataset_manifest_path),
            "dataset_manifest_sha256": inputs.dataset_manifest_sha256,
            "dataset_sha256": inputs.dataset_sha256,
            "source_build_manifest": str(inputs.source_build_manifest_path),
            "source_build_manifest_sha256": inputs.source_build_manifest_sha256,
            "attachments": str(inputs.attachments_path),
            "attachments_sha256": inputs.attachments_sha256,
            "attachments_manifest": str(inputs.attachments_manifest_path),
            "attachments_manifest_sha256": sha256_path(
                inputs.attachments_manifest_path
            ),
            "query_hits": str(inputs.query_hits_path),
            "query_hits_sha256": inputs.query_hits_sha256,
            "query_hits_manifest": str(inputs.query_manifest_path),
            "query_hits_manifest_sha256": sha256_path(inputs.query_manifest_path),
            "query_row_count": int(inputs.query_manifest["row_count"]),
            "ground_truth": str(inputs.ground_truth_path),
            "ground_truth_sha256": inputs.ground_truth_sha256,
            "ground_truth_manifest": str(inputs.ground_truth_manifest_path),
            "ground_truth_manifest_sha256": (
                inputs.ground_truth_manifest_sha256
            ),
        },
        "reference": reference,
        "candidate": (
            candidate_records[frozen.selected_candidate_name]
            if frozen.candidate_set == "formal"
            else None
        ),
        "candidates": candidate_records,
        "outputs": {
            "summary_csv": "summary.csv",
            "summary_json": "summary.json",
            "screen_manifest": "screen-manifest.json",
            "screen_manifest_sidecar": "screen-manifest.json.sha256",
        },
    }
    return rows, screen_manifest


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _phase_b_source_paths() -> dict[str, Path]:
    return {
        "evaluator": Path(__file__).resolve(),
        "offline_screen": (
            REPO_ROOT / "experiments/l1_balance/run_offline_screen.py"
        ).resolve(),
        "phase_a_generator": Path(phase_a_contract.__file__).resolve(),
        "phase_a_core": Path(phase_a_contract.core.__file__).resolve(),
        "source_binding": Path(source_binding_contract.__file__).resolve(),
    }


def _capture_phase_b_source_bundle() -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    for name, path in _phase_b_source_paths().items():
        data = path.read_bytes()
        bundle[name] = {
            "path": path,
            "bytes": data,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    return bundle


def _verify_phase_b_source_bundle(
    bundle: dict[str, dict[str, Any]],
) -> None:
    if set(bundle) != set(_phase_b_source_paths()):
        raise ValueError("Phase-B source bundle membership drifted")
    for name, record in bundle.items():
        path = Path(record["path"])
        if (
            path != _phase_b_source_paths()[name]
            or sha256_path(path) != record["sha256"]
            or path.stat().st_size != record["size_bytes"]
        ):
            raise ValueError(f"Phase-B source changed during evaluation: {name}")


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    screen_manifest: dict[str, Any],
    source_bundle: dict[str, dict[str, Any]],
) -> None:
    _verify_phase_b_source_bundle(source_bundle)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_dir = output_dir / "source"
    source_dir.mkdir()
    source_files: dict[str, dict[str, Any]] = {}
    source_paths: list[Path] = []
    for name, snapshot in source_bundle.items():
        original = Path(snapshot["path"])
        copied = source_dir / original.name
        with copied.open("xb") as handle:
            handle.write(snapshot["bytes"])
        copied_sha256 = sha256_path(copied)
        if copied_sha256 != snapshot["sha256"]:
            raise ValueError(f"Phase-B source copy changed while freezing {name}")
        source_files[name] = {
            "path": copied.relative_to(output_dir).as_posix(),
            "sha256": copied_sha256,
            "size_bytes": copied.stat().st_size,
        }
        source_paths.append(copied)
    source_record_path = output_dir / "phase-b-source-code.record.json"
    source_record = {
        "format_version": 1,
        "record_type": "phase_b_evaluator_source_code",
        "files": source_files,
        "runtime": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
        },
    }
    source_record_path.write_text(
        json.dumps(source_record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_record_sha256 = sha256_path(source_record_path)
    screen_manifest["evaluator_source_code"] = {
        "record": {
            "path": source_record_path.relative_to(output_dir).as_posix(),
            "sha256": source_record_sha256,
            "size_bytes": source_record_path.stat().st_size,
        },
        "files": source_files,
    }
    screen_manifest["identity_gates"]["phase_b_evaluator_source_code_sha256"] = True
    screen_manifest["outputs"].update(
        {
            "source_dir": source_dir.name,
            "source_record": source_record_path.name,
        }
    )
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"
    screen_path = output_dir / "screen-manifest.json"
    sidecar_path = output_dir / "screen-manifest.json.sha256"
    _write_csv(summary_csv_path, rows)
    summary_json_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    screen_path.write_text(
        json.dumps(screen_manifest, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    sidecar_path.write_text(sha256_path(screen_path) + "\n", encoding="ascii")
    for path in (
        *source_paths,
        source_record_path,
        summary_csv_path,
        summary_json_path,
        screen_path,
        sidecar_path,
    ):
        os.chmod(path, 0o444)


def run(args: argparse.Namespace) -> None:
    if (
        int(args.row_count) <= 0
        or int(args.attachment_k) != UPPER_NAVIGATION_TOP_K
        or int(args.ground_truth_width) <= 0
    ):
        raise ValueError(
            "row-count must be positive, attachment-k must equal 10, and "
            "ground-truth-width must be positive"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    frozen = validate_phase_a(args)
    source_bundle = _capture_phase_b_source_bundle()
    inputs = open_phase_b_inputs(args, frozen)
    rows, screen_manifest = evaluate_phase_b(frozen, inputs)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    _write_outputs(output_dir, rows, screen_manifest, source_bundle)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
