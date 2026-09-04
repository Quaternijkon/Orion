#!/usr/bin/env python3
"""Materialize the frozen formal N_native or C_CNBR owner.

This adapter is deliberately independent from the historical raw-v2
materializer.  It accepts only the checksum-bound formal Phase-A/Phase-B
N_native versus C_CNBR experiment, replays Orion's unchanged enabled/2/0/0
multi-assignment after the owner freeze, and changes only shard membership.
The production upper graph, ordered upper labels/vector bits, search contract,
and full vector row order are verified unchanged before a bundle is published.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from experiments.l1_balance import evaluate_cnbr_frozen_owners as phase_b
from experiments.l1_balance import prepare_native_cnbr_phase_a as phase_a
from experiments.l1_balance import cnbr_construction_cost_gate as cost_gate


FORMAT_VERSION = 1
TOOL_PATH = "experiments/l1_balance/materialize_native_cnbr_candidate.py"
FORMAL_CANDIDATE_SET = "formal"
GRID_CANDIDATE_SET = "frozen-grid"
ASSIGNMENT_FORMAT = "orion_numeric_import.assignments.jsonl-v1"
ATTACHMENT_TOP_K = phase_b.UPPER_NAVIGATION_TOP_K
ATTACHMENT_SEARCH_EF = phase_b.ATTACHMENT_SEARCH_EF
MULTI_ASSIGNMENT_CONTRACT = dict(phase_b.MULTI_ASSIGNMENT_CONTRACT)
MATERIALIZER_INVOCATION_COUNTS = {
    "full_l0_weighting": 0,
    "topology_refinement": 0,
    "fission": 0,
    "l0_repair": 0,
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
GRID_NAMES = (
    "CNBR_9_4",
    "CNBR_2_1",
    "CNBR_7_4",
    "CNBR_8_5",
    "CNBR_3_2",
    "CNBR_7_5",
)
SELECTED_GRID_NAME = "CNBR_9_4"
SELECTION_STAGE = "cnbr_dual_dataset_frozen_family_selection"


@dataclass(frozen=True)
class ArmContract:
    arm: str
    balance_variant: str
    phase_a_role: str
    phase_b_role: str
    selected_adoption_candidate: bool


ARM_CONTRACTS = {
    "N_native": ArmContract(
        arm="N_native",
        balance_variant="natural_orion",
        phase_a_role="balance_free_orion_core",
        phase_b_role="reference",
        selected_adoption_candidate=False,
    ),
    "C_CNBR": ArmContract(
        arm="C_CNBR",
        balance_variant="cnbr",
        phase_a_role="replacement_candidate",
        phase_b_role="candidate",
        selected_adoption_candidate=True,
    ),
}


@dataclass(frozen=True)
class MaterializationBinding:
    arm: ArmContract
    screen_path: Path
    screen_sha256: str
    screen: dict[str, Any]
    record: dict[str, Any]
    frozen: phase_b.FrozenPhaseA
    inputs: phase_b.PhaseBInputs
    owner: phase_b.FrozenOwner
    source_artifact_path: Path
    source_artifact_sha256: str
    vectors_path: Path
    vectors_sha256: str
    source_build_manifest_path: Path
    source_build_manifest_sha256: str
    dataset_manifest_path: Path
    dataset_manifest_sha256: str
    dataset_path: Path
    dataset_sha256: str
    replay_queries_path: Path
    replay_query_rows: int
    forbidden_stage_invocations: dict[str, int]
    evaluator_source_code_record_sha256: str
    selection_path: Path | None
    selection_sha256: str | None
    selection_source_code_record_sha256: str | None
    construction_cost_path: Path | None
    construction_cost_sha256: str | None
    construction_cost_gate: dict[str, Any] | None


@dataclass(frozen=True)
class AssignmentResult:
    path: Path
    layout_sha256: str
    physical_point_count: int
    shard_counts: np.ndarray
    copy_histogram: Counter[int]
    primary_shards_sha256: str
    primary_shard_loads: np.ndarray
    upper_memberships: list[list[int]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-b-screen", required=True)
    parser.add_argument(
        "--selection-manifest",
        help=(
            "Required only for C_CNBR. N_native is the independent fallback "
            "and is validated solely against its frozen formal Phase-A/Phase-B "
            "reference chain."
        ),
    )
    parser.add_argument(
        "--construction-cost-audit",
        help=(
            "Required only for C_CNBR. Must be the frozen aggregate "
            "construction-cost-v4 PASS audit bound to both selected formal "
            "Phase-B screens."
        ),
    )
    parser.add_argument("--arm", required=True, choices=tuple(ARM_CONTRACTS))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Audit the frozen Phase-A/Phase-B chain and exit before any writes.",
    )
    parser.add_argument("--generation", type=int)
    parser.add_argument("--rebind-binary")
    parser.add_argument("--replay-verifier")
    parser.add_argument("--output-dir")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args(argv)


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a valid SHA-256 string")
    return normalized


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def require_file(raw: Any, label: str) -> Path:
    if not isinstance(raw, (str, os.PathLike)) or not str(raw):
        raise ValueError(f"{label} path is missing")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def require_read_only(path: Path, label: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o222:
        raise ValueError(f"{label} is not frozen read-only: mode={mode:o}")


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay Orion's unchanged enabled/2/0/0 max-vote rule."""

    hit_owner = owner[local_hits]
    row_count = len(hit_owner)
    row_ids = np.arange(row_count, dtype=np.int64)
    votes = np.zeros((row_count, num_partitions), dtype=np.uint8)
    for column in range(hit_owner.shape[1]):
        np.add.at(votes, (row_ids, hit_owner[:, column]), 1)
    maximum = votes.max(axis=1)
    membership = votes == maximum[:, None]
    fallback = maximum < MULTI_ASSIGNMENT_CONTRACT["min_max_vote"]
    membership[fallback] = False
    membership[row_ids[fallback], hit_owner[fallback, 0]] = True
    copy_count = membership.sum(axis=1, dtype=np.int16)
    primary = np.asarray(
        [np.flatnonzero(row)[0] for row in membership], dtype="<i4"
    )
    return membership, copy_count, primary


def expected_phase_b_contract() -> dict[str, Any]:
    return {
        **phase_a.phase_a_contract(),
        "phase_a_fully_verified_before_l0_open": True,
        "evaluator_runs_in_separate_process": True,
        "owners_changed_after_freeze": False,
        "l0_assignment_after_owner_freeze": True,
        "l0_repair_or_flow": False,
        "multi_assignment": MULTI_ASSIGNMENT_CONTRACT,
        "load_balance_cannot_override_topology_failure": True,
    }


def _validate_screen_file(
    inputs: Mapping[str, Any], path_key: str, sha_key: str
) -> tuple[Path, str]:
    path = require_file(inputs.get(path_key), f"Phase-B {path_key}")
    expected_sha256 = normalized_sha256(
        inputs.get(sha_key), f"Phase-B {sha_key}"
    )
    actual_sha256 = sha256_path(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Phase-B {path_key} checksum drifted")
    return path, actual_sha256


def _validate_record_common(
    record: Mapping[str, Any], contract: ArmContract, owner: phase_b.FrozenOwner
) -> None:
    expected = {
        "name": contract.arm,
        "role": contract.phase_b_role,
        "phase_a_role": contract.phase_a_role,
        "owner_sha256": owner.sha256,
        "owner_record_sha256": owner.owner_record_sha256,
        "selected_adoption_candidate": contract.selected_adoption_candidate,
        "identity_all_pass": True,
        "graph_topology_all_pass": True,
        "query_topology_all_pass": True,
        "topology_all_pass": True,
        "screen_eligible": True,
        "materialization_eligible": True,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": record.get(key)}
        for key, expected_value in expected.items()
        if record.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Phase-B {contract.arm} record drifted: {mismatches}")
    if record.get("topology_metrics") != owner.record.get("topology_metrics"):
        raise ValueError(f"Phase-B {contract.arm} topology metrics drifted")
    parity = record.get("materialization_parity")
    if not isinstance(parity, Mapping):
        raise ValueError(f"Phase-B {contract.arm} lacks materialization parity")
    if parity.get("canonical_format") != ASSIGNMENT_FORMAT:
        raise ValueError(f"Phase-B {contract.arm} assignment format drifted")


def _validate_candidate_gates(
    screen: Mapping[str, Any], reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    graph_gates = phase_b.graph_gate_results(
        dict(candidate["topology_metrics"]), dict(reference["topology_metrics"])
    )
    query_count = int(
        screen["post_freeze_evaluation_inputs"]["query_row_count"]
    )
    query_gates = phase_b.query_gate_results(
        dict(candidate["actual_metrics"]),
        dict(reference["actual_metrics"]),
        query_count,
    )
    if candidate.get("graph_topology_gates") != graph_gates:
        raise ValueError("Phase-B C_CNBR graph topology gates do not replay")
    if candidate.get("query_topology_gates") != query_gates:
        raise ValueError("Phase-B C_CNBR query topology gates do not replay")
    if not all(bool(gate.get("pass")) for gate in graph_gates.values()):
        raise ValueError("Phase-B C_CNBR failed a graph topology gate")
    if not all(bool(gate.get("pass")) for gate in query_gates.values()):
        raise ValueError("Phase-B C_CNBR failed a query topology gate")
    reference_ratio = float(
        reference["actual_metrics"]["physical_copy_load_max_over_mean"]
    )
    candidate_ratio = float(
        candidate["actual_metrics"]["physical_copy_load_max_over_mean"]
    )
    expected_load_gate = {
        "observed": candidate_ratio,
        "reference": reference_ratio,
        "operator": "<",
        "threshold": reference_ratio,
        "pass": candidate_ratio < reference_ratio,
    }
    if candidate.get("physical_copy_load_gate") != expected_load_gate:
        raise ValueError("Phase-B C_CNBR physical-load gate drifted")
    if candidate_ratio >= reference_ratio:
        raise ValueError("Phase-B C_CNBR does not improve physical-copy imbalance")


def _relative_frozen_file(base_dir: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ValueError(f"{label} must be a relative frozen path")
    path = (base_dir / raw).resolve()
    try:
        path.relative_to(base_dir.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its frozen output directory") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _validate_source_snapshot(
    *,
    base_dir: Path,
    value: Any,
    label: str,
    record_type: str,
    expected_current_sources: Mapping[str, Path],
    runtime_bound: bool,
) -> str:
    if not isinstance(value, Mapping) or set(value) != {"record", "files"}:
        raise ValueError(f"{label} source-code binding schema drifted")
    files = value.get("files")
    record_binding = value.get("record")
    if not isinstance(files, Mapping) or set(files) != set(expected_current_sources):
        raise ValueError(f"{label} frozen source file set drifted")
    if not isinstance(record_binding, Mapping):
        raise ValueError(f"{label} source-code record binding is missing")
    record_path = _relative_frozen_file(
        base_dir, record_binding.get("path"), f"{label} source-code record"
    )
    record_sha256 = sha256_path(record_path)
    if (
        record_binding.get("sha256") != record_sha256
        or record_binding.get("size_bytes") != record_path.stat().st_size
    ):
        raise ValueError(f"{label} source-code record checksum drifted")
    require_read_only(record_path, f"{label} source-code record")
    record = load_json_object(record_path, f"{label} source-code record")
    expected_record_keys = {"format_version", "record_type", "files"}
    if runtime_bound:
        expected_record_keys.add("runtime")
    if (
        set(record) != expected_record_keys
        or record.get("format_version") != FORMAT_VERSION
        or record.get("record_type") != record_type
        or record.get("files") != files
    ):
        raise ValueError(f"{label} source-code record content drifted")
    if runtime_bound:
        runtime = record.get("runtime")
        if runtime != {
            "numpy_version": np.__version__,
            "python_version": sys.version,
        }:
            raise ValueError(f"{label} evaluator runtime binding drifted")
    for name, current_path in expected_current_sources.items():
        current_path = current_path.resolve()
        binding = files.get(name)
        if not isinstance(binding, Mapping) or set(binding) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"{label} source binding {name} is malformed")
        frozen_path = _relative_frozen_file(
            base_dir, binding.get("path"), f"{label} source {name}"
        )
        frozen_sha256 = sha256_path(frozen_path)
        if (
            binding.get("sha256") != frozen_sha256
            or binding.get("size_bytes") != frozen_path.stat().st_size
        ):
            raise ValueError(f"{label} frozen source {name} checksum drifted")
        require_read_only(frozen_path, f"{label} frozen source {name}")
        if sha256_path(current_path) != frozen_sha256:
            raise ValueError(
                f"current {label} source {name} differs from the frozen evaluator"
            )
    return record_sha256


def _validate_evaluator_source_code(
    screen_path: Path, screen: Mapping[str, Any], label: str
) -> str:
    expected_sources = {
        "evaluator": Path(phase_b.__file__).resolve(),
        "offline_screen": (
            REPO_ROOT / "experiments/l1_balance/run_offline_screen.py"
        ).resolve(),
        "phase_a_generator": Path(phase_a.__file__).resolve(),
        "phase_a_core": Path(phase_a.core.__file__).resolve(),
        "source_binding": Path(phase_b.source_binding_contract.__file__).resolve(),
    }
    return _validate_source_snapshot(
        base_dir=screen_path.parent,
        value=screen.get("evaluator_source_code"),
        label=label,
        record_type="phase_b_evaluator_source_code",
        expected_current_sources=expected_sources,
        runtime_bound=True,
    )


def _load_selection_screen(
    raw_path: Any,
    expected_sha256: Any,
    *,
    candidate_set: str,
    label: str,
) -> tuple[Path, str, dict[str, Any], str]:
    path = require_file(raw_path, label)
    digest = normalized_sha256(expected_sha256, f"{label} SHA-256")
    if sha256_path(path) != digest:
        raise ValueError(f"{label} checksum drifted")
    sidecar = path.with_name(path.name + ".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip() != digest
    ):
        raise ValueError(f"{label} checksum sidecar mismatch")
    require_read_only(path, label)
    require_read_only(sidecar, f"{label} sidecar")
    screen = load_json_object(path, label)
    if (
        screen.get("format_version") != FORMAT_VERSION
        or screen.get("stage") != phase_b.PHASE_B_STAGE
        or screen.get("candidate_set") != candidate_set
        or screen.get("identity_all_pass") is not True
    ):
        raise ValueError(f"{label} is not a canonical Phase-B v2 screen")
    source_record_sha256 = _validate_evaluator_source_code(path, screen, label)
    return path, digest, screen, source_record_sha256


def _selection_row(name: str, screens: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name}
    passes: list[bool] = []
    for dataset in ("sift1m", "glove-200-angular"):
        grid = screens[dataset]["grid"]
        candidates = grid.get("candidates")
        if not isinstance(candidates, Mapping) or not isinstance(
            candidates.get(name), Mapping
        ):
            raise ValueError(f"selection grid lacks {dataset} {name}")
        candidate = candidates[name]
        metrics = candidate.get("actual_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"selection grid lacks {dataset} {name} metrics")
        passed = candidate.get("screen_eligible") is True
        passes.append(passed)
        prefix = "sift" if dataset == "sift1m" else "glove"
        row.update(
            {
                f"{prefix}_screen_eligible": passed,
                f"{prefix}_graph_topology_all_pass": candidate.get(
                    "graph_topology_all_pass"
                ),
                f"{prefix}_query_topology_all_pass": candidate.get(
                    "query_topology_all_pass"
                ),
                f"{prefix}_physical_copy_load_max_over_mean": metrics.get(
                    "physical_copy_load_max_over_mean"
                ),
                f"{prefix}_routed_shards_mean": metrics.get(
                    "routed_shards_mean"
                ),
                f"{prefix}_gt_routing_coverage_mean": metrics.get(
                    "gt_routing_coverage_mean"
                ),
                f"{prefix}_expansion_ratio": metrics.get("expansion_ratio"),
            }
        )
    row["dual_dataset_all_gate_pass"] = all(passes)
    return row


def validate_selection_manifest(
    selection_path: Path | str,
    formal_screen_path: Path,
    formal_screen_sha256: str,
) -> tuple[Path, str, str, str]:
    selection_path = require_file(selection_path, "CNBR selection manifest")
    selection_sidecar = selection_path.with_name(
        selection_path.name + ".sha256"
    )
    selection_sha256 = sha256_path(selection_path)
    if (
        not selection_sidecar.is_file()
        or selection_sidecar.read_text(encoding="ascii").strip()
        != selection_sha256
    ):
        raise ValueError("CNBR selection manifest checksum sidecar mismatch")
    require_read_only(selection_path, "CNBR selection manifest")
    require_read_only(selection_sidecar, "CNBR selection manifest sidecar")
    selection = load_json_object(selection_path, "CNBR selection manifest")
    selection_expected = {
        "format_version": FORMAT_VERSION,
        "stage": SELECTION_STAGE,
        "selection_status": "post_exploratory_two_dataset_family_selection",
        "strict_cross_dataset_holdout_claim": False,
        "candidate_family": list(GRID_NAMES),
        "dual_dataset_all_gate_pass": [SELECTED_GRID_NAME],
        "selected_grid_candidate": SELECTED_GRID_NAME,
        "selected_formal_candidate": "C_CNBR",
        "selection_basis": (
            "unique_all_gate_pass_on_both_sift1m_and_glove_200_angular"
        ),
        "new_ratio_or_retuning_after_selection_forbidden": True,
    }
    mismatches = {
        key: {"expected": value, "actual": selection.get(key)}
        for key, value in selection_expected.items()
        if selection.get(key) != value
    }
    if mismatches:
        raise ValueError(f"CNBR selection contract drifted: {mismatches}")
    selection_source_record_sha256 = _validate_source_snapshot(
        base_dir=selection_path.parent,
        value=selection.get("synthesizer_source_code"),
        label="CNBR selection",
        record_type="cnbr_selection_synthesizer_source_code",
        expected_current_sources={
            "synthesizer": (
                REPO_ROOT
                / "experiments/l1_balance/synthesize_cnbr_dual_dataset_selection.py"
            ).resolve()
        },
        runtime_bound=False,
    )
    inputs = selection.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "sift1m",
        "glove-200-angular",
    }:
        raise ValueError("CNBR selection input dataset set drifted")
    screens: dict[str, dict[str, dict[str, Any]]] = {}
    matched_dataset = ""
    for dataset in ("sift1m", "glove-200-angular"):
        record = inputs.get(dataset)
        if not isinstance(record, Mapping) or set(record) != {
            "formal_phase_a_manifest_sha256",
            "formal_screen",
            "formal_screen_sha256",
            "grid_phase_a_manifest_sha256",
            "grid_screen",
            "grid_screen_sha256",
        }:
            raise ValueError(f"CNBR selection {dataset} input schema drifted")
        formal_path, formal_sha, formal, _formal_source_sha = (
            _load_selection_screen(
                record.get("formal_screen"),
                record.get("formal_screen_sha256"),
                candidate_set=FORMAL_CANDIDATE_SET,
                label=f"{dataset} formal Phase-B screen",
            )
        )
        grid_path, grid_sha, grid, _grid_source_sha = _load_selection_screen(
            record.get("grid_screen"),
            record.get("grid_screen_sha256"),
            candidate_set=GRID_CANDIDATE_SET,
            label=f"{dataset} grid Phase-B screen",
        )
        if (
            record.get("formal_phase_a_manifest_sha256")
            != formal.get("phase_a_manifest_sha256")
            or record.get("grid_phase_a_manifest_sha256")
            != grid.get("phase_a_manifest_sha256")
        ):
            raise ValueError(f"CNBR selection {dataset} Phase-A binding drifted")
        formal_reference = formal.get("reference")
        formal_candidate = formal.get("candidate")
        grid_reference = grid.get("reference")
        grid_candidates = grid.get("candidates")
        if not all(
            isinstance(value, Mapping)
            for value in (
                formal_reference,
                formal_candidate,
                grid_reference,
                grid_candidates,
            )
        ) or set(grid_candidates) != set(GRID_NAMES):
            raise ValueError(f"CNBR selection {dataset} owner records drifted")
        grid_candidate = grid_candidates[SELECTED_GRID_NAME]
        parity_fields = (
            "owner_sha256",
            "topology_metrics",
            "actual_metrics",
        )
        if any(
            formal_reference.get(field) != grid_reference.get(field)
            for field in ("owner_sha256", "owner_record_sha256")
        ) or any(
            formal_candidate.get(field) != grid_candidate.get(field)
            for field in parity_fields
        ):
            raise ValueError(f"CNBR selection {dataset} formal/grid 9/4 parity failed")
        if (
            formal_candidate.get("materialization_eligible") is not True
            or grid_candidate.get("screen_eligible") is not True
        ):
            raise ValueError(f"CNBR selection {dataset} 9/4 is not eligible")
        screens[dataset] = {"formal": formal, "grid": grid}
        if formal_path == formal_screen_path and formal_sha == formal_screen_sha256:
            matched_dataset = dataset
        # Resolve both paths during validation even though only the formal path
        # is used below; their exact bytes are selection inputs.
        _ = grid_path, grid_sha
    if not matched_dataset:
        raise ValueError("requested formal screen is not an input to the CNBR selection")
    expected_rows = [_selection_row(name, screens) for name in GRID_NAMES]
    if selection.get("rows") != expected_rows:
        raise ValueError("CNBR selection rows do not replay from the four v2 screens")
    dual_pass = [
        row["name"]
        for row in expected_rows
        if row["dual_dataset_all_gate_pass"]
    ]
    if dual_pass != [SELECTED_GRID_NAME]:
        raise ValueError("CNBR selection no longer has unique dual-dataset 9/4 pass")
    return (
        selection_path,
        selection_sha256,
        selection_source_record_sha256,
        matched_dataset,
    )


def validate_phase_b_screen(
    screen_path: Path | str,
    arm_name: str,
    selection_path: Path | str | None,
    construction_cost_path: Path | str | None,
) -> MaterializationBinding:
    """Re-audit the frozen chain without rebuilding either owner."""

    if arm_name not in ARM_CONTRACTS:
        raise ValueError("arm must be exactly N_native or C_CNBR")
    arm = ARM_CONTRACTS[arm_name]
    screen_path = require_file(screen_path, "Phase-B screen manifest")
    screen_sidecar = screen_path.with_name(screen_path.name + ".sha256")
    if not screen_sidecar.is_file():
        raise ValueError("Phase-B screen checksum sidecar is missing")
    screen_sha256 = sha256_path(screen_path)
    if screen_sidecar.read_text(encoding="ascii").strip() != screen_sha256:
        raise ValueError("Phase-B screen checksum sidecar mismatch")
    require_read_only(screen_path, "Phase-B screen manifest")
    require_read_only(screen_sidecar, "Phase-B screen checksum sidecar")
    screen = load_json_object(screen_path, "Phase-B screen manifest")
    if (
        screen.get("format_version") != FORMAT_VERSION
        or screen.get("stage") != phase_b.PHASE_B_STAGE
        or screen.get("candidate_set") != FORMAL_CANDIDATE_SET
        or screen.get("comparison") != "N_native_vs_C_CNBR"
    ):
        raise ValueError("materializer accepts only the formal N_native/C_CNBR Phase-B screen")
    expected_outputs = {
        "screen_manifest": "screen-manifest.json",
        "screen_manifest_sidecar": "screen-manifest.json.sha256",
        "source_dir": "source",
        "source_record": "phase-b-source-code.record.json",
        "summary_csv": "summary.csv",
        "summary_json": "summary.json",
    }
    if screen.get("outputs") != expected_outputs:
        raise ValueError("Phase-B v2 output/source schema drifted")
    evaluator_source_code_record_sha256 = _validate_evaluator_source_code(
        screen_path, screen, "requested formal Phase-B screen"
    )
    validated_selection_path: Path | None = None
    selection_sha256: str | None = None
    selection_source_code_record_sha256: str | None = None
    construction_cost_binding: dict[str, Any] | None = None
    if arm_name == "C_CNBR":
        if selection_path in (None, ""):
            raise ValueError("C_CNBR requires --selection-manifest")
        if construction_cost_path in (None, ""):
            raise ValueError("C_CNBR requires --construction-cost-audit")
        (
            validated_selection_path,
            selection_sha256,
            selection_source_code_record_sha256,
            _selection_dataset,
        ) = validate_selection_manifest(
            selection_path, screen_path, screen_sha256
        )
        selection_value = load_json_object(
            validated_selection_path, "CNBR selection manifest"
        )
        selection_inputs = selection_value.get("inputs")
        if not isinstance(selection_inputs, Mapping):
            raise ValueError("CNBR selection input binding disappeared")
        expected_formal_screens = {
            dataset: selection_inputs[dataset]["formal_screen_sha256"]
            for dataset in ("sift1m", "glove-200-angular")
        }
        construction_cost_binding = cost_gate.validate_construction_cost_v4_pass(
            construction_cost_path,
            expected_formal_screen_sha256_by_dataset=expected_formal_screens,
        )
    if screen.get("contract") != expected_phase_b_contract():
        raise ValueError("Phase-B architecture contract drifted")
    identity_gates = screen.get("identity_gates")
    if (
        screen.get("identity_all_pass") is not True
        or not isinstance(identity_gates, Mapping)
        or not identity_gates
        or any(value is not True for value in identity_gates.values())
    ):
        raise ValueError("Phase-B identity gates did not all pass")

    phase_a_manifest_path = require_file(
        screen.get("phase_a_manifest"), "Phase-A manifest"
    )
    phase_a_manifest_sha256 = normalized_sha256(
        screen.get("phase_a_manifest_sha256"), "Phase-A manifest SHA-256"
    )
    if sha256_path(phase_a_manifest_path) != phase_a_manifest_sha256:
        raise ValueError("Phase-B binds a drifted Phase-A manifest")
    reference = screen.get("reference")
    candidate = screen.get("candidate")
    candidates = screen.get("candidates")
    if not all(isinstance(value, Mapping) for value in (reference, candidate, candidates)):
        raise ValueError("Phase-B formal owner records are incomplete")
    if set(candidates) != {"C_CNBR"} or candidate != candidates["C_CNBR"]:
        raise ValueError("Phase-B formal candidate set drifted")
    reference_parity = reference.get("materialization_parity")
    if not isinstance(reference_parity, Mapping):
        raise ValueError("Phase-B N_native parity is missing")
    row_count = int(reference_parity.get("logical_point_count", 0))
    if row_count <= 0:
        raise ValueError("Phase-B logical point count is invalid")

    frozen = phase_b.validate_phase_a(
        argparse.Namespace(
            phase_a_manifest=str(phase_a_manifest_path), row_count=row_count
        )
    )
    if (
        frozen.candidate_set != FORMAL_CANDIDATE_SET
        or set(frozen.owners) != {"N_native", "C_CNBR"}
        or frozen.selected_candidate_name != "C_CNBR"
        or frozen.manifest_sha256 != phase_a_manifest_sha256
    ):
        raise ValueError("Phase-A is not the exact formal N_native/C_CNBR pair")
    performance_path = require_file(
        screen.get("phase_a_performance_record"), "Phase-A performance record"
    )
    if (
        performance_path != frozen.performance_path
        or normalized_sha256(
            screen.get("phase_a_performance_record_sha256"),
            "Phase-A performance SHA-256",
        )
        != frozen.performance_sha256
        or screen.get("phase_a_shared_construction_performance")
        != frozen.performance.get("shared")
    ):
        raise ValueError("Phase-B performance binding drifted")

    _validate_record_common(
        reference,
        ARM_CONTRACTS["N_native"],
        frozen.owners["N_native"],
    )
    if arm_name == "C_CNBR":
        _validate_record_common(
            candidate,
            ARM_CONTRACTS["C_CNBR"],
            frozen.owners["C_CNBR"],
        )
        if candidate.get("base_owner_sha256") != frozen.owners["N_native"].sha256:
            raise ValueError("Phase-B C_CNBR parent N owner drifted")
        _validate_candidate_gates(screen, reference, candidate)
        record = dict(candidate)
    else:
        record = dict(reference)

    post_freeze = screen.get("post_freeze_evaluation_inputs")
    if not isinstance(post_freeze, Mapping):
        raise ValueError("Phase-B post-freeze input binding is missing")
    expected_paths = (
        ("dataset_manifest", "dataset_manifest_sha256"),
        ("source_build_manifest", "source_build_manifest_sha256"),
        ("attachments", "attachments_sha256"),
        ("attachments_manifest", "attachments_manifest_sha256"),
        ("query_hits", "query_hits_sha256"),
        ("query_hits_manifest", "query_hits_manifest_sha256"),
        ("ground_truth", "ground_truth_sha256"),
        ("ground_truth_manifest", "ground_truth_manifest_sha256"),
    )
    bound_files = {
        path_key: _validate_screen_file(post_freeze, path_key, sha_key)[0]
        for path_key, sha_key in expected_paths
    }
    ground_truth_manifest = load_json_object(
        bound_files["ground_truth_manifest"], "ground-truth manifest"
    )
    ground_truth_width = int(ground_truth_manifest.get("width", 0))
    if ground_truth_width <= 0:
        raise ValueError("ground-truth width is invalid")
    open_args = argparse.Namespace(
        dataset_manifest=str(bound_files["dataset_manifest"]),
        source_build_manifest=str(bound_files["source_build_manifest"]),
        attachments=str(bound_files["attachments"]),
        attachments_manifest=str(bound_files["attachments_manifest"]),
        row_count=row_count,
        attachment_k=ATTACHMENT_TOP_K,
        query_hits=str(bound_files["query_hits"]),
        query_hits_manifest=str(bound_files["query_hits_manifest"]),
        ground_truth=str(bound_files["ground_truth"]),
        ground_truth_manifest=str(bound_files["ground_truth_manifest"]),
        ground_truth_width=ground_truth_width,
    )
    inputs = phase_b.open_phase_b_inputs(open_args, frozen)
    if int(post_freeze.get("query_row_count", 0)) != int(
        inputs.query_manifest["row_count"]
    ):
        raise ValueError("Phase-B query-row count drifted")

    attachment_manifest = load_json_object(
        bound_files["attachments_manifest"], "attachment manifest"
    )
    vectors_path = require_file(
        attachment_manifest.get("vectors_path"), "attachment vector source"
    )
    vectors_sha256 = normalized_sha256(
        attachment_manifest.get("vectors_sha256"), "attachment vector SHA-256"
    )
    if sha256_path(vectors_path) != vectors_sha256:
        raise ValueError("attachment vector source checksum drifted")
    expected_vector_size = (
        row_count * int(frozen.artifact["vector_schema"]["dimension"]) * 4
    )
    if vectors_path.stat().st_size != expected_vector_size:
        raise ValueError("attachment vector source has an invalid shape")

    dataset_manifest = load_json_object(
        bound_files["dataset_manifest"], "dataset manifest"
    )
    source_dataset = dataset_manifest.get("source_dataset")
    dataset = dataset_manifest.get("dataset")
    files = dataset_manifest.get("files")
    if not all(isinstance(value, Mapping) for value in (source_dataset, dataset, files)):
        raise ValueError("dataset manifest provenance is incomplete")
    dataset_path = require_file(source_dataset.get("path"), "source dataset")
    dataset_sha256 = normalized_sha256(dataset.get("sha256"), "dataset SHA-256")
    if (
        source_dataset.get("sha256") != dataset_sha256
        or sha256_path(dataset_path) != dataset_sha256
    ):
        raise ValueError("source dataset checksum drifted")
    query_record = files.get("queries")
    if not isinstance(query_record, Mapping):
        raise ValueError("dataset query record is missing")
    replay_queries_path = require_file(
        query_record.get("path"), "upper replay query corpus"
    )
    replay_query_rows = int(dataset.get("expected_query_rows", 0))
    replay_query_sha256 = normalized_sha256(
        query_record.get("sha256"), "query corpus SHA-256"
    )
    if (
        replay_query_rows <= 0
        or replay_query_rows != int(inputs.query_manifest["row_count"])
        or sha256_path(replay_queries_path) != replay_query_sha256
    ):
        raise ValueError("upper replay query corpus drifted")

    forbidden = frozen.manifest.get("forbidden_stage_invocations")
    expected_forbidden = phase_a.phase_a_forbidden_stage_invocations()
    if forbidden != expected_forbidden:
        raise ValueError("Phase-A forbidden-stage invocation counters drifted")
    if any(value != 0 for value in forbidden.values()):
        raise ValueError("a forbidden Phase-A stage was invoked")

    owner = frozen.owners[arm_name]
    require_read_only(owner.path, f"{arm_name} owner")
    require_read_only(owner.owner_record_path, f"{arm_name} owner record")
    return MaterializationBinding(
        arm=arm,
        screen_path=screen_path,
        screen_sha256=screen_sha256,
        screen=screen,
        record=record,
        frozen=frozen,
        inputs=inputs,
        owner=owner,
        source_artifact_path=frozen.artifact_path,
        source_artifact_sha256=sha256_path(frozen.artifact_path),
        vectors_path=vectors_path,
        vectors_sha256=vectors_sha256,
        source_build_manifest_path=bound_files["source_build_manifest"],
        source_build_manifest_sha256=sha256_path(
            bound_files["source_build_manifest"]
        ),
        dataset_manifest_path=bound_files["dataset_manifest"],
        dataset_manifest_sha256=sha256_path(bound_files["dataset_manifest"]),
        dataset_path=dataset_path,
        dataset_sha256=dataset_sha256,
        replay_queries_path=replay_queries_path,
        replay_query_rows=replay_query_rows,
        forbidden_stage_invocations=dict(forbidden),
        evaluator_source_code_record_sha256=(
            evaluator_source_code_record_sha256
        ),
        selection_path=validated_selection_path,
        selection_sha256=selection_sha256,
        selection_source_code_record_sha256=(
            selection_source_code_record_sha256
        ),
        construction_cost_path=(
            Path(construction_cost_binding["audit"])
            if construction_cost_binding is not None
            else None
        ),
        construction_cost_sha256=(
            construction_cost_binding["audit_sha256"]
            if construction_cost_binding is not None
            else None
        ),
        construction_cost_gate=construction_cost_binding,
    )


def validate_materialization_parity(
    binding: MaterializationBinding,
    *,
    layout_sha256: str,
    physical_point_count: int,
    shard_counts: np.ndarray,
    copy_histogram: Counter[int],
    primary_shards_sha256: str,
    primary_shard_loads: np.ndarray,
) -> None:
    parity = binding.record["materialization_parity"]
    expected = {
        "canonical_format": ASSIGNMENT_FORMAT,
        "assignment_bytes_sha256": layout_sha256,
        "logical_point_count": int(binding.frozen.artifact["logical_point_count"]),
        "physical_point_count": physical_point_count,
        "primary_shards_sha256": primary_shards_sha256,
        "primary_shard_loads": [int(value) for value in primary_shard_loads],
        "physical_copy_shard_loads": [int(value) for value in shard_counts],
        "copy_count_histogram": {
            str(key): int(value) for key, value in sorted(copy_histogram.items())
        },
    }
    if parity != expected:
        mismatches = {
            key: {"expected": value, "actual": parity.get(key)}
            for key, value in expected.items()
            if parity.get(key) != value
        }
        extras = sorted(set(parity) - set(expected))
        raise ValueError(
            "golden Phase-B multi-assignment parity mismatch: "
            f"mismatches={mismatches}, extra={extras}"
        )


def materialize_assignments(
    binding: MaterializationBinding,
    output_dir: Path,
    chunk_size: int,
) -> AssignmentResult:
    row_count = int(binding.frozen.artifact["logical_point_count"])
    partitions = int(binding.frozen.num_partitions)
    upper_labels = np.asarray(binding.frozen.labels, dtype=np.int64)
    if (
        np.any(upper_labels < 0)
        or np.any(upper_labels >= row_count)
        or len(np.unique(upper_labels)) != len(upper_labels)
    ):
        raise ValueError("upper labels must be unique numeric logical point IDs")
    upper_position = np.full(row_count, -1, dtype=np.int32)
    upper_position[upper_labels] = np.arange(len(upper_labels), dtype=np.int32)
    upper_memberships: list[list[int] | None] = [None] * len(upper_labels)
    assignment_path = output_dir / "orion_numeric_import.assignments.jsonl"
    temporary_path = assignment_path.with_name(assignment_path.name + ".tmp")
    assignment_digest = hashlib.sha256()
    primary_digest = hashlib.sha256()
    shard_counts = np.zeros(partitions, dtype=np.int64)
    primary_shard_loads = np.zeros(partitions, dtype=np.int64)
    copy_histogram: Counter[int] = Counter()
    physical_point_count = 0
    try:
        with temporary_path.open("xb") as handle:
            for start in range(0, row_count, chunk_size):
                stop = min(row_count, start + chunk_size)
                membership, copy_count, primary = compact_membership(
                    binding.owner.values,
                    np.asarray(binding.inputs.attachment_local[start:stop]),
                    partitions,
                )
                shard_counts += membership.sum(axis=0, dtype=np.int64)
                primary_shard_loads += np.bincount(
                    primary, minlength=partitions
                ).astype(np.int64, copy=False)
                primary_digest.update(
                    np.ascontiguousarray(primary, dtype="<i4").tobytes(order="C")
                )
                values, counts = np.unique(copy_count, return_counts=True)
                copy_histogram.update(
                    {
                        int(value): int(count)
                        for value, count in zip(
                            values.tolist(), counts.tolist(), strict=True
                        )
                    }
                )
                physical_point_count += int(copy_count.sum())
                for offset, row in enumerate(membership):
                    point_id = start + offset
                    shards = np.flatnonzero(row).astype(int).tolist()
                    encoded = assignment_bytes(point_id, shards)
                    handle.write(encoded)
                    assignment_digest.update(encoded)
                    upper_index = int(upper_position[point_id])
                    if upper_index >= 0:
                        upper_memberships[upper_index] = shards
                print(f"materialized={stop}/{row_count}", flush=True)
        layout_sha256 = assignment_digest.hexdigest()
        primary_shards_sha256 = primary_digest.hexdigest()
        validate_materialization_parity(
            binding,
            layout_sha256=layout_sha256,
            physical_point_count=physical_point_count,
            shard_counts=shard_counts,
            copy_histogram=copy_histogram,
            primary_shards_sha256=primary_shards_sha256,
            primary_shard_loads=primary_shard_loads,
        )
        os.replace(temporary_path, assignment_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    if any(row is None for row in upper_memberships):
        raise RuntimeError("not every upper point received its final L0 membership")
    if np.any(shard_counts <= 0):
        raise RuntimeError("materialization produced an empty logical shard")
    if sha256_path(assignment_path) != layout_sha256:
        raise RuntimeError("published assignment checksum differs from Phase-B parity")
    return AssignmentResult(
        path=assignment_path,
        layout_sha256=layout_sha256,
        physical_point_count=physical_point_count,
        shard_counts=shard_counts,
        copy_histogram=copy_histogram,
        primary_shards_sha256=primary_shards_sha256,
        primary_shard_loads=primary_shard_loads,
        upper_memberships=[list(row) for row in upper_memberships if row is not None],
    )


def _validate_rebound_artifact(
    source: Mapping[str, Any],
    rebound: Mapping[str, Any],
    assignment: AssignmentResult,
    generation: int,
    *,
    allow_shard_count_change: bool = False,
) -> None:
    immutable_fields = [
        "format_version",
        "logical_point_count",
        "upper_k",
        "upper_ef_search",
        "dynamic_ef_base",
        "dynamic_ef_factor",
        "vector_schema",
        "upper_graph",
    ]
    if not allow_shard_count_change:
        immutable_fields.append("shard_count")
    changed = {
        field: {"source": source.get(field), "rebound": rebound.get(field)}
        for field in immutable_fields
        if source.get(field) != rebound.get(field)
    }
    if changed:
        raise RuntimeError(f"membership rebind changed immutable fields: {changed}")
    if allow_shard_count_change and rebound.get("shard_count") != len(
        assignment.shard_counts
    ):
        raise RuntimeError("rebound artifact shard count differs from the assignment")
    if (
        rebound.get("generation") != generation
        or rebound.get("layout_sha256") != assignment.layout_sha256
        or rebound.get("physical_point_count") != assignment.physical_point_count
    ):
        raise RuntimeError("rebound artifact generation/layout/count binding drifted")
    source_nodes = source.get("upper_nodes")
    rebound_nodes = rebound.get("upper_nodes")
    if (
        not isinstance(source_nodes, list)
        or not isinstance(rebound_nodes, list)
        or len(source_nodes) != len(rebound_nodes)
        or len(rebound_nodes) != len(assignment.upper_memberships)
    ):
        raise RuntimeError("rebind changed the ordered upper-node count")
    for index, (source_node, rebound_node, memberships) in enumerate(
        zip(source_nodes, rebound_nodes, assignment.upper_memberships, strict=True)
    ):
        if not isinstance(source_node, Mapping) or not isinstance(rebound_node, Mapping):
            raise RuntimeError(f"upper node {index} is malformed")
        for field in ("label", "vector"):
            if source_node.get(field) != rebound_node.get(field):
                raise RuntimeError(
                    f"rebind changed ordered upper-node {field} at index {index}"
                )
        if rebound_node.get("shard_membership") != memberships:
            raise RuntimeError(
                f"rebind membership differs from replay at upper index {index}"
            )


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
    source_shard_count: int | None = None,
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
    stdout_path = manifest_path.parent / "upper-replay.stdout.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path = manifest_path.parent / "upper-replay.stderr.log"
    if completed.stderr:
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    manifest = load_json_object(manifest_path, "upper replay manifest")
    source = manifest.get("source")
    rebound = manifest.get("rebound")
    query = manifest.get("query_corpus")
    replay = manifest.get("replay")
    gates = manifest.get("gates")
    if not all(isinstance(value, Mapping) for value in (source, rebound, query, replay, gates)):
        raise ValueError("upper replay manifest is incomplete")
    expected = {
        "format_version": FORMAT_VERSION,
        "verdict": "PASS",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("upper replay manifest did not publish PASS")
    source_expected = {
        "path": str(source_artifact),
        "file_sha256": source_sha256,
        "generation": source_generation,
        "layout_sha256": source_layout_sha256,
        "shard_count": (
            shard_count if source_shard_count is None else source_shard_count
        ),
    }
    rebound_expected = {
        "path": str(rebound_artifact),
        "file_sha256": rebound_sha256,
        "generation": rebound_generation,
        "layout_sha256": rebound_layout_sha256,
        "shard_count": shard_count,
        "physical_point_count": rebound_physical_count,
    }
    query_sha256 = sha256_path(queries)
    query_expected = {
        "path": str(queries),
        "sha256": query_sha256,
        "row_count": query_rows,
        "dimension": dimension,
        "size_bytes": query_rows * dimension * 4,
    }
    replay_sha256 = sha256_path(replay_path)
    replay_expected = {
        "path": str(replay_path),
        "sha256": replay_sha256,
        "ordered_label_and_distance_bits_sha256": replay_sha256,
        "size_bytes": replay_path.stat().st_size,
    }
    for label, actual, expected_values in (
        ("source", source, source_expected),
        ("rebound", rebound, rebound_expected),
        ("query", query, query_expected),
        ("replay", replay, replay_expected),
    ):
        mismatches = {
            key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected_values.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise ValueError(f"upper replay {label} binding mismatch: {mismatches}")
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
    for gate in REPLAY_GATES:
        if gates.get(gate) is not True:
            raise ValueError(f"upper replay gate {gate} did not pass")
    return {
        "manifest_sha256": sha256_path(manifest_path),
        "replay_sha256": replay_sha256,
        "query_sha256": query_sha256,
        "canonical_upper_graph_sha256": source["canonical_upper_graph_sha256"],
        "ordered_upper_nodes_identity_sha256": source[
            "ordered_upper_nodes_identity_sha256"
        ],
    }


def _upper_build_provenance(
    binding: MaterializationBinding,
) -> dict[str, Any]:
    construction = binding.frozen.manifest.get("construction_inputs")
    if not isinstance(construction, Mapping):
        raise ValueError("Phase-A construction inputs are missing")
    projection = construction.get("upper_only_projection")
    if not isinstance(projection, Mapping):
        raise ValueError("Phase-A upper-only projection binding is missing")
    provenance = projection.get("upper_build_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "source_manifest_sha256",
        "source_manifest_size_bytes",
        "parameters",
    }:
        raise ValueError("upper-build provenance schema drifted")
    normalized_sha256(
        provenance.get("source_manifest_sha256"),
        "upper-build source manifest SHA-256",
    )
    source_size = provenance.get("source_manifest_size_bytes")
    if (
        isinstance(source_size, bool)
        or not isinstance(source_size, int)
        or source_size <= 0
    ):
        raise ValueError("upper-build source manifest size is invalid")
    parameters = provenance.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != set(
        phase_a.UPPER_BUILD_PARAMETER_KEYS
    ):
        raise ValueError("upper-build parameter whitelist drifted")
    safe_parameters: dict[str, int] = {}
    for key in phase_a.UPPER_BUILD_PARAMETER_KEYS:
        value = parameters[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"upper-build parameter {key} must be a positive integer"
            )
        safe_parameters[key] = value
    return {
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "source_manifest_size_bytes": source_size,
        "parameters": safe_parameters,
    }


def _build_parameters(
    binding: MaterializationBinding, generation: int
) -> dict[str, Any]:
    artifact = binding.frozen.artifact
    upper_build = _upper_build_provenance(binding)["parameters"]
    parameters = {
        "balance_mode": binding.arm.balance_variant,
        "l1_partitioner": binding.arm.arm,
        "initial_num_shards": int(binding.frozen.num_partitions),
        "enable_fission": False,
        "enable_topology_refinement": False,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
        "sample_denominator": int(
            binding.frozen.manifest["fixed_parameters"]["upper_subset_denominator"]
        ),
        "upper_sample_seed": upper_build["upper_sample_seed"],
        "upper_m": upper_build["upper_m"],
        "upper_ef_construction": upper_build["upper_ef_construction"],
        "upper_graph_seed": upper_build["upper_graph_seed"],
        "attachment_search_ef": ATTACHMENT_SEARCH_EF,
        "k_overlap": ATTACHMENT_TOP_K,
        "upper_k": int(artifact["upper_k"]),
        "upper_search_ef": int(artifact["upper_ef_search"]),
        "dynamic_ef_base": int(artifact["dynamic_ef_base"]),
        "dynamic_ef_factor": int(artifact["dynamic_ef_factor"]),
        "allow_decoupled_runtime_upper_search": False,
        "generation": generation,
        "l0_repair": False,
    }
    integer_fields = (
        "sample_denominator",
        "upper_sample_seed",
        "upper_m",
        "upper_ef_construction",
        "upper_graph_seed",
    )
    if any(
        isinstance(parameters[field], bool)
        or not isinstance(parameters[field], int)
        or parameters[field] <= 0
        for field in integer_fields
    ):
        raise ValueError("source upper-build parameters are not positive integers")
    return parameters


def file_record(path: Path, known_sha256: str | None = None) -> dict[str, Any]:
    return {
        "sha256": known_sha256 or sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def materialize_bundle(
    binding: MaterializationBinding,
    *,
    generation: int,
    rebind_binary: Path,
    replay_verifier: Path,
    output_dir: Path,
    chunk_size: int,
    tool_path: str = TOOL_PATH,
    parameter_overrides: Mapping[str, Any] | None = None,
    extra_diagnostics: Mapping[str, Any] | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
    assignment_builder: Callable[
        [MaterializationBinding, Path, int], AssignmentResult
    ]
    | None = None,
) -> dict[str, Any]:
    if generation <= int(binding.frozen.artifact["generation"]):
        raise ValueError("generation must be greater than the source generation")
    if chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output_dir.parent}")
    for path, label in (
        (rebind_binary, "rebind binary"),
        (replay_verifier, "upper replay verifier"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"{label} is not executable: {path}")

    output_dir.mkdir(parents=False, exist_ok=False)
    assignment = (assignment_builder or materialize_assignments)(
        binding, output_dir, chunk_size
    )
    source_artifact = binding.frozen.artifact
    source_generation = int(source_artifact["generation"])
    membership_sidecar_path = output_dir / "rebind-memberships.json"
    write_json_new(
        membership_sidecar_path,
        {
            "format_version": FORMAT_VERSION,
            "source_artifact_sha256": binding.source_artifact_sha256,
            "source_generation": source_generation,
            "generation": generation,
            "layout_sha256": assignment.layout_sha256,
            "shard_count": int(binding.frozen.num_partitions),
            "physical_point_count": assignment.physical_point_count,
            "shard_memberships": assignment.upper_memberships,
        },
    )
    rebound_path = output_dir / f"generation-{generation}.json"
    completed = subprocess.run(
        [
            str(rebind_binary),
            str(binding.source_artifact_path),
            str(membership_sidecar_path),
            str(rebound_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rebind_stdout_path = output_dir / "rebind.stdout.log"
    rebind_stdout_path.write_text(completed.stdout, encoding="utf-8")
    rebind_stderr_path = output_dir / "rebind.stderr.log"
    if completed.stderr:
        rebind_stderr_path.write_text(completed.stderr, encoding="utf-8")
    if not rebound_path.is_file():
        raise RuntimeError("rebind binary did not publish an artifact")
    rebound_sha256 = sha256_path(rebound_path)
    rebound_sidecar_path = rebound_path.with_name(rebound_path.name + ".sha256")
    if (
        not rebound_sidecar_path.is_file()
        or rebound_sidecar_path.read_text(encoding="ascii").strip()
        != rebound_sha256
    ):
        raise RuntimeError("rebound artifact checksum sidecar mismatch")
    rebound_artifact = load_json_object(rebound_path, "rebound artifact")
    _validate_rebound_artifact(
        source_artifact,
        rebound_artifact,
        assignment,
        generation,
        allow_shard_count_change=(
            tool_path
            == "experiments/c1/scripts/c1_orion_bmr10_scaling_materialize.py"
        ),
    )

    dimension = int(source_artifact["vector_schema"]["dimension"])
    replay_path = output_dir / "upper-replay.bin"
    replay_manifest_path = output_dir / "upper-replay-manifest.json"
    replay_gate = run_upper_replay_gate(
        verifier=replay_verifier,
        source_artifact=binding.source_artifact_path,
        rebound_artifact=rebound_path,
        queries=binding.replay_queries_path,
        query_rows=binding.replay_query_rows,
        dimension=dimension,
        replay_path=replay_path,
        manifest_path=replay_manifest_path,
        source_sha256=binding.source_artifact_sha256,
        rebound_sha256=rebound_sha256,
        source_generation=source_generation,
        rebound_generation=generation,
        source_layout_sha256=str(source_artifact["layout_sha256"]),
        rebound_layout_sha256=assignment.layout_sha256,
        shard_count=int(binding.frozen.num_partitions),
        rebound_physical_count=assignment.physical_point_count,
        source_shard_count=int(source_artifact["shard_count"]),
    )

    vector_link_path = output_dir / "orion_numeric_import.f32le"
    source_vector_mode = stat.S_IMODE(binding.vectors_path.stat().st_mode)
    os.link(binding.vectors_path, vector_link_path)
    if not os.path.samefile(binding.vectors_path, vector_link_path):
        raise RuntimeError("materialized vectors are not a hardlink to source rows")
    if stat.S_IMODE(binding.vectors_path.stat().st_mode) != source_vector_mode:
        raise RuntimeError("hardlink creation changed source vector permissions")

    import_manifest_path = output_dir / "orion_numeric_import.manifest.json"
    write_json_new(
        import_manifest_path,
        {
            "format_version": FORMAT_VERSION,
            "orion_generation": generation,
            "orion_artifact_file": rebound_path.name,
            "orion_artifact_sha256": rebound_sha256,
            "dimension": dimension,
            "point_count": int(source_artifact["logical_point_count"]),
            "shard_count": int(binding.frozen.num_partitions),
            "vector_name": str(source_artifact["vector_schema"]["vector_name"]),
            "vectors_file": vector_link_path.name,
            "vectors_sha256": binding.vectors_sha256,
            "assignments_file": assignment.path.name,
            "assignments_sha256": assignment.layout_sha256,
            "total_point_copies": assignment.physical_point_count,
        },
    )
    owner_copy_path = output_dir / f"l1-owner-{binding.arm.arm}.i32le"
    shutil.copyfile(binding.owner.path, owner_copy_path)
    if sha256_path(owner_copy_path) != binding.owner.sha256:
        raise RuntimeError("copied L1 owner differs from the frozen Phase-A owner")

    diagnostics: dict[str, Any] = {
        "arm": binding.arm.arm,
        "input_scope": (
            "production_upper_graph_upper_vectors_and_upper_self_navigation_only"
        ),
        "balance_contract": {"variant": binding.arm.balance_variant},
        "phase_a_role": binding.arm.phase_a_role,
        "phase_b_role": binding.arm.phase_b_role,
        "phase_a_manifest_sha256": binding.frozen.manifest_sha256,
        "phase_a_owner_sha256": binding.owner.sha256,
        "phase_a_owner_record_sha256": binding.owner.owner_record_sha256,
        "owner_sha256": binding.owner.sha256,
        "phase_b_manifest_sha256": binding.screen_sha256,
        "phase_b_evaluator_source_code_record_sha256": (
            binding.evaluator_source_code_record_sha256
        ),
        "attachments_sha256": binding.inputs.attachments_sha256,
        "golden_assignment_bytes_sha256": binding.record[
            "materialization_parity"
        ]["assignment_bytes_sha256"],
        "upper_replay_manifest_sha256": replay_gate["manifest_sha256"],
        "selected_adoption_candidate": binding.arm.selected_adoption_candidate,
        "identity_all_pass": binding.record["identity_all_pass"],
        "topology_all_pass": binding.record["topology_all_pass"],
        "topology_metrics": binding.record["topology_metrics"],
        "graph_topology_gates": binding.record["graph_topology_gates"],
        "query_topology_gates": binding.record["query_topology_gates"],
        "physical_copy_load_improves_over_reference": binding.record[
            "physical_copy_load_improves_over_reference"
        ],
        "l1_sizes": binding.owner.record["partition_sizes"],
        "multi_assignment_after_owner_freeze": True,
        "forbidden_stage_invocations": dict(
            binding.forbidden_stage_invocations
        ),
        "invocation_counts": dict(MATERIALIZER_INVOCATION_COUNTS),
    }
    if extra_diagnostics:
        diagnostics.update(dict(extra_diagnostics))
    if binding.arm.arm == "C_CNBR":
        if (
            binding.selection_path is None
            or binding.selection_sha256 is None
            or binding.selection_source_code_record_sha256 is None
            or binding.construction_cost_path is None
            or binding.construction_cost_sha256 is None
            or binding.construction_cost_gate is None
        ):
            raise RuntimeError(
                "C_CNBR materialization lost its selection/cost binding"
            )
        diagnostics.update(
            {
                "parent_n_owner_sha256": binding.frozen.owners[
                    "N_native"
                ].sha256,
                "parent_n_manifest_sha256": binding.frozen.manifest_sha256,
                "selection_manifest_sha256": binding.selection_sha256,
                "selection_source_code_record_sha256": (
                    binding.selection_source_code_record_sha256
                ),
                "construction_cost_v4_gate": "PASS",
                "construction_cost_v4_audit_sha256": (
                    binding.construction_cost_sha256
                ),
            }
        )

    upper_build_provenance = _upper_build_provenance(binding)
    parameters = _build_parameters(binding, generation)
    if parameter_overrides:
        parameters.update(dict(parameter_overrides))
    load_mean = assignment.physical_point_count / binding.frozen.num_partitions
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    build_manifest_path = output_dir / "build-manifest.json"
    dataset_manifest = load_json_object(
        binding.dataset_manifest_path, "dataset manifest"
    )
    build_manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "tool": tool_path,
        "mode": "production_bundle",
        "created_at": created_at,
        "dataset": {
            "path": str(binding.dataset_path),
            "sha256": binding.dataset_sha256,
            "size_bytes": binding.dataset_path.stat().st_size,
            "name": dataset_manifest["dataset"].get("name"),
            "dimension": dimension,
            "train_rows_total": int(source_artifact["logical_point_count"]),
            "train_rows_used": int(source_artifact["logical_point_count"]),
        },
        "parameters": parameters,
        "artifact_binding": {
            "generation": generation,
            "layout_sha256": assignment.layout_sha256,
            "logical_point_count": int(source_artifact["logical_point_count"]),
            "physical_point_count": assignment.physical_point_count,
            "shard_count": int(binding.frozen.num_partitions),
        },
        "routing": {
            "initial_num_shards": int(binding.frozen.num_partitions),
            "effective_num_shards": int(binding.frozen.num_partitions),
            "logical_point_count": int(source_artifact["logical_point_count"]),
            "physical_point_count": assignment.physical_point_count,
            "expansion_ratio": assignment.physical_point_count
            / int(source_artifact["logical_point_count"]),
            "shard_counts": [int(value) for value in assignment.shard_counts],
            "upper_point_count": len(binding.frozen.labels),
            "fission_events": [],
            "l1_partition_diagnostics": diagnostics,
            "load_summary": {
                "min": int(assignment.shard_counts.min()),
                "max": int(assignment.shard_counts.max()),
                "mean": load_mean,
                "cv": float(np.std(assignment.shard_counts) / load_mean),
                "max_over_mean": float(
                    assignment.shard_counts.max() / load_mean
                ),
                "min_over_mean": float(
                    assignment.shard_counts.min() / load_mean
                ),
            },
            "copy_count_histogram": {
                str(key): int(value)
                for key, value in sorted(assignment.copy_histogram.items())
            },
        },
        "provenance": {
            "phase_a_manifest": str(binding.frozen.manifest_path),
            "phase_a_manifest_sha256": binding.frozen.manifest_sha256,
            "phase_a_owner": str(binding.owner.path),
            "phase_a_owner_sha256": binding.owner.sha256,
            "phase_a_owner_record": str(binding.owner.owner_record_path),
            "phase_a_owner_record_sha256": binding.owner.owner_record_sha256,
            "phase_b_screen": str(binding.screen_path),
            "phase_b_screen_sha256": binding.screen_sha256,
            "phase_b_evaluator_source_code_record_sha256": (
                binding.evaluator_source_code_record_sha256
            ),
            "source_artifact": str(binding.source_artifact_path),
            "source_artifact_sha256": binding.source_artifact_sha256,
            "attachments": str(binding.inputs.attachments_path),
            "attachments_manifest": str(
                binding.inputs.attachments_manifest_path
            ),
            "vectors_source": str(binding.vectors_path),
            "vectors_source_sha256": binding.vectors_sha256,
            "vectors_source_build_manifest": str(
                binding.source_build_manifest_path
            ),
            "vectors_source_build_manifest_sha256": (
                binding.source_build_manifest_sha256
            ),
            "upper_build_parameter_provenance": upper_build_provenance,
            "dataset_manifest": str(binding.dataset_manifest_path),
            "dataset_manifest_sha256": binding.dataset_manifest_sha256,
            "point_id_order_contract": (
                "attachment row i = hardlinked vector row i = logical point id i"
            ),
            "vectors_materialization": "hardlink",
            "vectors_same_file": True,
            "source_vector_mode_preserved": f"{source_vector_mode:o}",
            "rebind_binary": str(rebind_binary),
            "rebind_binary_sha256": sha256_path(rebind_binary),
            "upper_replay_verifier": str(replay_verifier),
            "upper_replay_verifier_sha256": sha256_path(replay_verifier),
            "upper_replay_queries": str(binding.replay_queries_path),
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
    if extra_provenance:
        build_manifest["provenance"].update(dict(extra_provenance))
    if binding.arm.arm == "C_CNBR":
        assert binding.selection_path is not None
        assert binding.selection_sha256 is not None
        assert binding.selection_source_code_record_sha256 is not None
        assert binding.construction_cost_path is not None
        assert binding.construction_cost_sha256 is not None
        assert binding.construction_cost_gate is not None
        build_manifest["provenance"].update(
            {
                "selection_manifest": str(binding.selection_path),
                "selection_manifest_sha256": binding.selection_sha256,
                "selection_source_code_record_sha256": (
                    binding.selection_source_code_record_sha256
                ),
                "construction_cost_v4_audit": str(
                    binding.construction_cost_path
                ),
                "construction_cost_v4_audit_sha256": (
                    binding.construction_cost_sha256
                ),
                "construction_cost_v4_gate": binding.construction_cost_gate,
            }
        )

    payload_hashes: dict[Path, str] = {
        rebound_path: rebound_sha256,
        rebound_sidecar_path: sha256_path(rebound_sidecar_path),
        assignment.path: assignment.layout_sha256,
        vector_link_path: binding.vectors_sha256,
        import_manifest_path: sha256_path(import_manifest_path),
        owner_copy_path: binding.owner.sha256,
        membership_sidecar_path: sha256_path(membership_sidecar_path),
        replay_path: replay_gate["replay_sha256"],
        replay_manifest_path: replay_gate["manifest_sha256"],
        rebind_stdout_path: sha256_path(rebind_stdout_path),
        output_dir / "upper-replay.stdout.log": sha256_path(
            output_dir / "upper-replay.stdout.log"
        ),
    }
    for optional in (
        rebind_stderr_path,
        output_dir / "upper-replay.stderr.log",
    ):
        if optional.exists():
            payload_hashes[optional] = sha256_path(optional)
    build_manifest["outputs"]["files"] = {
        path.name: file_record(path, digest)
        for path, digest in sorted(
            payload_hashes.items(), key=lambda item: item[0].name
        )
    }
    write_json_new(build_manifest_path, build_manifest)
    build_manifest_sha256 = sha256_path(build_manifest_path)
    checksums_path = output_dir / "checksums.sha256"
    with checksums_path.open("x", encoding="ascii", newline="\n") as handle:
        for path, digest in sorted(
            {**payload_hashes, build_manifest_path: build_manifest_sha256}.items(),
            key=lambda item: item[0].name,
        ):
            handle.write(f"{digest}  {path.name}\n")

    # A hardlink shares the source inode.  Never chmod it: doing so would mutate
    # the canonical source vector file.  Every newly created metadata/payload
    # file is frozen read-only after all checksums are final.
    for path in output_dir.iterdir():
        if path.is_file() and path != vector_link_path:
            os.chmod(path, 0o444)
    if stat.S_IMODE(binding.vectors_path.stat().st_mode) != source_vector_mode:
        raise RuntimeError("bundle freeze changed source vector permissions")
    return {
        "bundle": str(output_dir),
        "artifact": str(rebound_path),
        "artifact_sha256": rebound_sha256,
        "build_manifest": str(build_manifest_path),
        "build_manifest_sha256": build_manifest_sha256,
        "layout_sha256": assignment.layout_sha256,
        "physical_point_count": assignment.physical_point_count,
        "balance_variant": binding.arm.balance_variant,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm == "C_CNBR" and args.selection_manifest in (None, ""):
        raise ValueError("C_CNBR requires --selection-manifest")
    if args.arm == "C_CNBR" and args.construction_cost_audit in (None, ""):
        raise ValueError("C_CNBR requires --construction-cost-audit")
    binding = validate_phase_b_screen(
        args.phase_b_screen,
        args.arm,
        args.selection_manifest,
        args.construction_cost_audit,
    )
    validation = {
        "status": "PASS",
        "arm": binding.arm.arm,
        "balance_variant": binding.arm.balance_variant,
        "phase_a_manifest_sha256": binding.frozen.manifest_sha256,
        "phase_a_owner_sha256": binding.owner.sha256,
        "phase_a_owner_record_sha256": binding.owner.owner_record_sha256,
        "phase_b_manifest_sha256": binding.screen_sha256,
        "attachments_sha256": binding.inputs.attachments_sha256,
        "logical_point_count": int(binding.frozen.artifact["logical_point_count"]),
        "upper_point_count": len(binding.frozen.labels),
        "shard_count": int(binding.frozen.num_partitions),
    }
    if binding.selection_sha256 is not None:
        validation["selection_manifest_sha256"] = binding.selection_sha256
    if binding.construction_cost_sha256 is not None:
        validation["construction_cost_v4_audit_sha256"] = (
            binding.construction_cost_sha256
        )
        validation["construction_cost_v4_gate"] = "PASS"
    if args.validate_only:
        return validation
    missing = [
        name
        for name in ("generation", "rebind_binary", "replay_verifier", "output_dir")
        if getattr(args, name, None) in (None, "")
    ]
    if missing:
        raise ValueError(
            "materialization requires arguments: " + ", ".join(missing)
        )
    rebind_binary = require_file(args.rebind_binary, "rebind binary")
    replay_verifier = require_file(args.replay_verifier, "upper replay verifier")
    output_dir = Path(args.output_dir).expanduser().resolve()
    return materialize_bundle(
        binding,
        generation=int(args.generation),
        rebind_binary=rebind_binary,
        replay_verifier=replay_verifier,
        output_dir=output_dir,
        chunk_size=int(args.chunk_size),
    )


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
