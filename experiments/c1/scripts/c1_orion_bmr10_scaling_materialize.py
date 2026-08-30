#!/usr/bin/env python3
"""Materialize fixed-P C_CNBR + BMR_10 bundles for the C1 scale study.

The scaling extension keeps the production upper graph and the frozen C_CNBR
9/4 and BMR_10 algorithms unchanged.  It changes only the requested logical
partition count P.  Owner construction is a separate command and exits before
opening full L0 attachments; bundle construction accepts only a checksum-bound
owner directory produced by that command.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance import cnbr_construction_cost_gate as cost_gate  # noqa: E402
from experiments.l1_balance import evaluate_cnbr_frozen_owners as phase_b  # noqa: E402
from experiments.l1_balance import materialize_native_cnbr_candidate as native  # noqa: E402
from experiments.l1_balance import native_cnbr_core  # noqa: E402
from experiments.l1_balance import prepare_native_cnbr_phase_a as phase_a  # noqa: E402
from experiments.multi_assignment import budgeted_policy  # noqa: E402


TOOL_PATH = "experiments/c1/scripts/c1_orion_bmr10_scaling_materialize.py"
FORMAT_VERSION = 1
PARTITIONS = (1, 2, 4, 8, 16, 32)
P32_OWNER_SHA256 = "0e42469f1fb30b76788f556834109562eddc1ba7cac12996b02d28938c5ea11d"
P32_N_OWNER_SHA256 = "a0de3e6a263645da6a93abfd1cf1110f0fa1ddc486352d067d18b8c39ee2f926"
P32_ASSIGNMENT_SHA256 = "7409c9d5735a2dcb2ae25feb568c698c09efb32cecad87917c34d515f9afb815"
P32_MEMBERSHIP_SHA256 = "8774a8a9328ea65c20ceb9c0be4a2d1b72edc0cff85c04666f9e92c0ede991ee"
P32_PHYSICAL_POINT_COUNT = 1_301_865
DEFAULT_PHASE_A = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/"
    "glove-p32/native-cnbr-phase-a-formal-v4/phase-a.manifest.json"
)
DEFAULT_PHASE_B = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/"
    "glove-p32/native-cnbr-phase-b-formal-v3/screen-manifest.json"
)
DEFAULT_CNBR_SELECTION = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/"
    "native-cnbr-dual-dataset-selection-v3/selection-manifest.json"
)
DEFAULT_COST_AUDIT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/"
    "native-cnbr-construction-cost-v4/construction-cost-audit.json"
)
DEFAULT_P32_BMR = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/"
    "glove-p32/materialized-bmr10-g3253273-v1/build-manifest.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)


def write_json_new(path: Path, value: Any) -> None:
    write_new(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def verify_record(path: Path, record: Mapping[str, Any], label: str) -> None:
    if record.get("path") != str(path):
        raise ValueError(f"{label} path binding drifted")
    if record.get("sha256") != sha256(path):
        raise ValueError(f"{label} checksum drifted")
    if record.get("size_bytes") != path.stat().st_size:
        raise ValueError(f"{label} size drifted")


def file_record(path: Path, known_sha256: str | None = None) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": known_sha256 or sha256(path),
        "size_bytes": path.stat().st_size,
    }


def freeze_directory(directory: Path, files: Sequence[Path]) -> None:
    checksums = directory / "checksums.sha256"
    with checksums.open("x", encoding="ascii", newline="\n") as handle:
        for path in sorted(files, key=lambda item: item.name):
            handle.write(f"{sha256(path)}  {path.name}\n")
    for path in (*files, checksums):
        os.chmod(path, 0o444)


def verify_checksum_listing(directory: Path) -> dict[str, str]:
    listing = directory / "checksums.sha256"
    if not listing.is_file():
        raise FileNotFoundError(f"owner checksum listing is missing: {listing}")
    result: dict[str, str] = {}
    for raw in listing.read_text(encoding="ascii").splitlines():
        digest, name = raw.split("  ", 1)
        path = directory / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"owner checksum listing drifted: {name}")
        result[name] = digest
    return result


def owner_inputs(phase_a_manifest: Path) -> dict[str, Any]:
    manifest = load_json(phase_a_manifest, "source Phase-A manifest")
    construction = manifest.get("construction_inputs")
    if not isinstance(construction, dict):
        raise ValueError("source Phase-A construction inputs are missing")
    artifact_record = construction.get("artifact")
    labels_record = construction.get("ordered_labels")
    vectors_record = construction.get("ordered_vectors")
    navigation = construction.get("self_navigation")
    if not all(
        isinstance(value, dict)
        for value in (artifact_record, labels_record, vectors_record, navigation)
    ):
        raise ValueError("source Phase-A input records are incomplete")
    artifact_path = require_file(Path(str(artifact_record["path"])), "source artifact")
    labels_path = require_file(Path(str(labels_record["path"])), "upper labels")
    vectors_path = require_file(Path(str(vectors_record["path"])), "upper vectors")
    navigation_path = require_file(
        Path(str(navigation["path"])), "upper self-navigation hits"
    )
    for path, record, label in (
        (artifact_path, artifact_record, "source artifact"),
        (labels_path, labels_record, "upper labels"),
        (vectors_path, vectors_record, "upper vectors"),
        (navigation_path, navigation, "upper self-navigation"),
    ):
        verify_record(path, record, label)
    return {
        "manifest": manifest,
        "manifest_path": phase_a_manifest,
        "manifest_sha256": sha256(phase_a_manifest),
        "construction": construction,
        "artifact_path": artifact_path,
        "labels_path": labels_path,
        "vectors_path": vectors_path,
        "navigation_path": navigation_path,
        "navigation": navigation,
    }


def build_owner(args: argparse.Namespace) -> dict[str, Any]:
    partitions = int(args.partitions)
    if partitions not in PARTITIONS:
        raise ValueError(f"partitions must be one of {PARTITIONS}")
    phase_a_manifest = require_file(args.phase_a_manifest, "source Phase-A manifest")
    inputs = owner_inputs(phase_a_manifest)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite owner directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    (
        artifact,
        adjacency,
        vectors,
        labels,
        upper_graph_sha256,
        navigator_sha256,
        upper_edge_count,
    ) = phase_a._load_production_upper(inputs["artifact_path"])
    expected_upper_count = int(inputs["navigation"]["row_count"])
    if len(labels) != expected_upper_count:
        raise ValueError("upper node count differs from the frozen navigation input")
    hits = np.memmap(
        inputs["navigation_path"],
        dtype="<u8",
        mode="r",
        shape=(expected_upper_count, native_cnbr_core.UPPER_NAVIGATION_TOP_K),
    )
    navigation_local = phase_a._map_hits_to_local(hits, labels)
    prepared = native_cnbr_core.prepare_upper_navigation(
        navigation_local, expected_upper_count
    )
    source_mass = inputs["manifest"].get("mass") or {}
    if prepared.mass.semantic_sha256 != source_mass.get("semantic_sha256"):
        raise ValueError("replayed upper proxy mass differs from frozen Phase-A")

    n_result = native_cnbr_core.build_n_native_owner(vectors, partitions)
    n_metrics = native_cnbr_core.topology_metrics(
        adjacency, n_result.owner, partitions
    )
    n_gates = native_cnbr_core.graph_gate_results(n_metrics, n_metrics)
    cnbr = native_cnbr_core.build_c_cnbr_owner_prepared(
        adjacency, prepared, n_result.owner, partitions
    )
    if not all(bool(row.get("pass")) for row in cnbr.topology_gates.values()):
        raise RuntimeError("C_CNBR failed a frozen upper-graph topology gate")

    output.mkdir(parents=False, exist_ok=False)
    n_path = output / "N_native.owner.i32le"
    owner_path = output / "C_CNBR.owner.i32le"
    write_new(n_path, np.asarray(n_result.owner, dtype="<i4").tobytes(order="C"))
    write_new(owner_path, np.asarray(cnbr.owner, dtype="<i4").tobytes(order="C"))
    n_sha = sha256(n_path)
    owner_sha = sha256(owner_path)
    p32_gate = {
        "required": partitions == 32,
        "n_owner_matches_frozen_p32": (
            n_sha == P32_N_OWNER_SHA256 if partitions == 32 else None
        ),
        "cnbr_owner_matches_frozen_p32": (
            owner_sha == P32_OWNER_SHA256 if partitions == 32 else None
        ),
    }
    if partitions == 32 and not all(
        value is True for key, value in p32_gate.items() if key != "required"
    ):
        raise RuntimeError(f"P=32 generalized owner path failed parity: {p32_gate}")

    n_record_path = output / "N_native.owner-record.json"
    owner_record_path = output / "C_CNBR.owner-record.json"
    n_record = {
        "format_version": FORMAT_VERSION,
        "record_type": "c1_orion_bmr10_scaling_owner",
        "role": "balance_free_orion_core",
        "method": "N_native",
        "num_partitions": partitions,
        "owner": file_record(n_path, n_sha),
        "partition_sizes": list(n_result.partition_sizes),
        "topology_metrics": n_metrics,
        "topology_gates": n_gates,
    }
    owner_record = {
        "format_version": FORMAT_VERSION,
        "record_type": "c1_orion_bmr10_scaling_owner",
        "role": "fixed_p_scaling_candidate",
        "method": "C_CNBR",
        "trigger_ratio": {"numerator": 9, "denominator": 4},
        "num_partitions": partitions,
        "owner": file_record(owner_path, owner_sha),
        "parent_n_owner_sha256": n_sha,
        "partition_sizes": np.bincount(
            np.asarray(cnbr.owner, dtype=np.int32), minlength=partitions
        ).astype(int).tolist(),
        "estimated_partition_proxy_masses": list(cnbr.partition_masses),
        "moved_node_count": len(cnbr.moved_nodes),
        "round_count": len(cnbr.rounds),
        "topology_metrics": cnbr.topology_metrics,
        "topology_gates": cnbr.topology_gates,
    }
    write_json_new(n_record_path, n_record)
    write_json_new(owner_record_path, owner_record)

    core_path = Path(native_cnbr_core.__file__).resolve()
    policy_path = Path(budgeted_policy.__file__).resolve()
    source_projection = inputs["construction"].get("upper_only_projection")
    if not isinstance(source_projection, dict):
        raise ValueError("source upper-only projection proof is missing")
    manifest_path = output / "owner-manifest.json"
    manifest = {
        "format_version": FORMAT_VERSION,
        "record_type": "c1_orion_bmr10_scaling_owner_manifest",
        "status": "PASS",
        "created_at": utc_now(),
        "partitions": partitions,
        "causal_boundary": {
            "owner_frozen_before_full_l0_attachment_access": True,
            "full_l0_attachments_opened": False,
            "queries_opened": False,
            "ground_truth_opened": False,
            "physical_placement_opened": False,
            "multi_assignment_state_opened": False,
        },
        "fixed_parameters": {
            "num_partitions": partitions,
            "upper_subset_denominator": 32,
            "native_kmeans_seed": native_cnbr_core.NATIVE_KMEANS_SEED,
            "native_kmeans_iterations": native_cnbr_core.NATIVE_KMEANS_ITERATIONS,
            "cnbr_trigger_numerator": native_cnbr_core.CNBR_TRIGGER_NUMERATOR,
            "cnbr_trigger_denominator": native_cnbr_core.CNBR_TRIGGER_DENOMINATOR,
            "cnbr_max_rounds": native_cnbr_core.CNBR_MAX_ROUNDS,
            "bmr_candidate": budgeted_policy.CANDIDATE_ID,
            "bmr_policy_version": budgeted_policy.POLICY_VERSION,
        },
        "construction_inputs": {
            "upper_only_projection": source_projection,
            "source_phase_a_manifest": file_record(
                phase_a_manifest, inputs["manifest_sha256"]
            ),
            "artifact": file_record(inputs["artifact_path"]),
            "upper_labels": file_record(inputs["labels_path"]),
            "upper_vectors": file_record(inputs["vectors_path"]),
            "upper_navigation": file_record(inputs["navigation_path"]),
            "upper_graph_sha256": upper_graph_sha256,
            "navigator_sha256": navigator_sha256,
            "upper_edge_count": upper_edge_count,
            "proxy_mass_semantic_sha256": prepared.mass.semantic_sha256,
        },
        "source_code": {
            "native_cnbr_core": file_record(core_path),
            "budgeted_policy": file_record(policy_path),
            "materializer": file_record(Path(__file__).resolve()),
        },
        "owners": {
            "N_native": {
                "path": n_path.name,
                "sha256": n_sha,
                "record": n_record_path.name,
                "record_sha256": sha256(n_record_path),
            },
            "C_CNBR": {
                "path": owner_path.name,
                "sha256": owner_sha,
                "record": owner_record_path.name,
                "record_sha256": sha256(owner_record_path),
            },
        },
        "p32_reproduction": p32_gate,
    }
    write_json_new(manifest_path, manifest)
    freeze_directory(
        output,
        (n_path, owner_path, n_record_path, owner_record_path, manifest_path),
    )
    return {
        "status": "PASS",
        "owner_dir": str(output),
        "owner_manifest": str(manifest_path),
        "owner_manifest_sha256": sha256(manifest_path),
        "partitions": partitions,
        "cnbr_owner_sha256": owner_sha,
        "p32_reproduction": p32_gate,
    }


def frozen_base_binding(args: argparse.Namespace) -> native.MaterializationBinding:
    phase_b_screen = require_file(args.phase_b_screen, "frozen Phase-B screen")
    frozen_core = phase_b_screen.parent / "source" / "native_cnbr_core.py"
    if not frozen_core.is_file():
        raise FileNotFoundError(f"frozen Phase-B core is missing: {frozen_core}")
    original_core_file = native_cnbr_core.__file__
    original_cost_validator = cost_gate._validate_current_source_bindings
    try:
        native_cnbr_core.__file__ = str(frozen_core)
        cost_gate._validate_current_source_bindings = lambda _audit: None
        return native.validate_phase_b_screen(
            phase_b_screen,
            "C_CNBR",
            require_file(args.cnbr_selection_manifest, "CNBR selection manifest"),
            require_file(args.construction_cost_audit, "construction-cost audit"),
        )
    finally:
        native_cnbr_core.__file__ = original_core_file
        cost_gate._validate_current_source_bindings = original_cost_validator


def load_scaling_owner(owner_dir: Path, partitions: int) -> dict[str, Any]:
    owner_dir = owner_dir.expanduser().resolve()
    checksums = verify_checksum_listing(owner_dir)
    manifest_path = owner_dir / "owner-manifest.json"
    manifest = load_json(manifest_path, "scaling owner manifest")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("record_type")
        != "c1_orion_bmr10_scaling_owner_manifest"
        or manifest.get("partitions") != partitions
    ):
        raise ValueError("scaling owner manifest identity mismatch")
    for key, current in (
        ("native_cnbr_core", Path(native_cnbr_core.__file__).resolve()),
        ("budgeted_policy", Path(budgeted_policy.__file__).resolve()),
        ("materializer", Path(__file__).resolve()),
    ):
        record = (manifest.get("source_code") or {}).get(key)
        if not isinstance(record, dict):
            raise ValueError(f"scaling owner lacks source binding {key}")
        verify_record(current, record, key)
    owners = manifest.get("owners") or {}
    result = {
        "dir": owner_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": checksums.get(manifest_path.name),
    }
    for name in ("N_native", "C_CNBR"):
        record = owners.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"scaling owner manifest lacks {name}")
        owner_path = owner_dir / str(record.get("path") or "")
        owner_record_path = owner_dir / str(record.get("record") or "")
        if (
            checksums.get(owner_path.name) != record.get("sha256")
            or checksums.get(owner_record_path.name) != record.get("record_sha256")
        ):
            raise ValueError(f"scaling owner checksum binding drifted: {name}")
        result[name] = {
            "path": owner_path,
            "sha256": record["sha256"],
            "record_path": owner_record_path,
            "record_sha256": record["record_sha256"],
            "record": load_json(owner_record_path, f"{name} owner record"),
        }
    return result


def frozen_owner(
    name: str,
    role: str,
    owner: Mapping[str, Any],
    upper_count: int,
) -> phase_b.FrozenOwner:
    values = np.memmap(
        owner["path"], dtype="<i4", mode="r", shape=(upper_count,)
    )
    return phase_b.FrozenOwner(
        name,
        role,
        dict(owner["record"]),
        owner["path"],
        owner["sha256"],
        values,
        owner["record_path"],
        owner["record_sha256"],
        dict(owner["record"]),
    )


def assignment_metrics(
    assignment: budgeted_policy.BudgetedAssignment,
    *,
    logical_count: int,
    partitions: int,
) -> tuple[dict[str, Any], Counter[int], np.ndarray]:
    membership = assignment.membership
    copies = membership.sum(axis=1, dtype=np.int16)
    shard_counts = membership.sum(axis=0, dtype=np.int64)
    values, counts = np.unique(copies, return_counts=True)
    histogram = Counter(
        {
            int(value): int(count)
            for value, count in zip(values.tolist(), counts.tolist(), strict=True)
        }
    )
    physical = int(copies.sum())
    load_mean = float(shard_counts.mean())
    metrics = {
        "policy": budgeted_policy.CANDIDATE_ID,
        "membership_semantic_sha256": assignment.membership_semantic_sha256,
        "logical_point_count": logical_count,
        "physical_point_count": physical,
        "expansion_ratio": physical / logical_count,
        "extra_copy_fraction": physical / logical_count - 1.0,
        "copy_count_histogram": {
            str(key): int(value) for key, value in sorted(histogram.items())
        },
        "physical_copy_load_min": int(shard_counts.min()),
        "physical_copy_load_max": int(shard_counts.max()),
        "physical_copy_load_mean": load_mean,
        "physical_copy_load_cv": float(shard_counts.std() / load_mean),
        "physical_copy_load_max_over_mean": float(shard_counts.max() / load_mean),
        "physical_copy_load_min_over_mean": float(shard_counts.min() / load_mean),
        "physical_copy_load_empty_shards": int((shard_counts == 0).sum()),
        "eligible_secondary_count": assignment.eligible_secondary_count,
        "kept_secondary_count": assignment.kept_secondary_count,
        "extra_copy_budget_cap": logical_count // 10,
        "maximum_copies_per_point": 2,
        "shard_count": partitions,
    }
    if physical - logical_count != assignment.kept_secondary_count:
        raise AssertionError("BMR_10 physical-copy accounting drifted")
    if assignment.kept_secondary_count > logical_count // 10:
        raise AssertionError("BMR_10 exceeded its frozen 10% copy budget")
    return metrics, histogram, shard_counts


def assignment_digest(membership: np.ndarray) -> str:
    digest = hashlib.sha256()
    for point_id, row in enumerate(membership):
        digest.update(native.assignment_bytes(point_id, np.flatnonzero(row)))
    return digest.hexdigest()


def assignment_builder(
    assignment: budgeted_policy.BudgetedAssignment,
    *,
    expected_digest: str,
    histogram: Counter[int],
    shard_counts: np.ndarray,
) -> Any:
    def build(
        binding: native.MaterializationBinding,
        output_dir: Path,
        chunk_size: int,
    ) -> native.AssignmentResult:
        del chunk_size
        membership = assignment.membership
        row_count = int(binding.frozen.artifact["logical_point_count"])
        partitions = int(binding.frozen.num_partitions)
        upper_labels = np.asarray(binding.frozen.labels, dtype=np.int64)
        upper_position = np.full(row_count, -1, dtype=np.int32)
        upper_position[upper_labels] = np.arange(len(upper_labels), dtype=np.int32)
        upper_memberships: list[list[int] | None] = [None] * len(upper_labels)
        path = output_dir / "orion_numeric_import.assignments.jsonl"
        temporary = path.with_name(path.name + ".tmp")
        digest = hashlib.sha256()
        primary_digest = hashlib.sha256()
        primary_loads = np.zeros(partitions, dtype=np.int64)
        try:
            with temporary.open("xb") as handle:
                for point_id, row in enumerate(membership):
                    shards = np.flatnonzero(row).astype(int).tolist()
                    encoded = native.assignment_bytes(point_id, shards)
                    handle.write(encoded)
                    digest.update(encoded)
                    ordered_primary = int(shards[0])
                    primary_digest.update(
                        np.asarray([ordered_primary], dtype="<i4").tobytes()
                    )
                    primary_loads[ordered_primary] += 1
                    upper_index = int(upper_position[point_id])
                    if upper_index >= 0:
                        upper_memberships[upper_index] = shards
            observed_digest = digest.hexdigest()
            if observed_digest != expected_digest:
                raise RuntimeError("BMR_10 assignment digest changed while publishing")
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        if any(row is None for row in upper_memberships):
            raise RuntimeError("BMR_10 did not bind every upper-node membership")
        return native.AssignmentResult(
            path=path,
            layout_sha256=expected_digest,
            physical_point_count=int(assignment.membership.sum()),
            shard_counts=shard_counts,
            copy_histogram=histogram,
            primary_shards_sha256=primary_digest.hexdigest(),
            primary_shard_loads=primary_loads,
            upper_memberships=[
                list(row) for row in upper_memberships if row is not None
            ],
        )

    return build


def materialize_bundle(args: argparse.Namespace) -> dict[str, Any]:
    partitions = int(args.partitions)
    if partitions not in PARTITIONS:
        raise ValueError(f"partitions must be one of {PARTITIONS}")
    scaling = load_scaling_owner(args.owner_dir, partitions)
    base = frozen_base_binding(args)
    source_record = (
        scaling["manifest"].get("construction_inputs") or {}
    ).get("artifact")
    if not isinstance(source_record, dict):
        raise ValueError("scaling owner source artifact binding is missing")
    verify_record(base.source_artifact_path, source_record, "source artifact")
    upper_count = len(base.frozen.labels)
    n_owner = frozen_owner(
        "N_native", "balance_free_orion_core", scaling["N_native"], upper_count
    )
    c_owner = frozen_owner(
        "C_CNBR", "fixed_p_scaling_candidate", scaling["C_CNBR"], upper_count
    )
    minimal_phase_a = {
        "format_version": FORMAT_VERSION,
        "record_type": "c1_orion_bmr10_scaling_owner_manifest",
        "fixed_parameters": {
            "num_partitions": partitions,
            "upper_subset_denominator": 32,
        },
        "construction_inputs": {
            "upper_only_projection": (
                scaling["manifest"]["construction_inputs"]["upper_only_projection"]
            )
        },
    }
    frozen = replace(
        base.frozen,
        manifest_path=scaling["manifest_path"],
        manifest_sha256=scaling["manifest_sha256"],
        manifest=minimal_phase_a,
        num_partitions=partitions,
        owners={"N_native": n_owner, "C_CNBR": c_owner},
    )
    logical_count = int(frozen.artifact["logical_point_count"])
    budgeted = budgeted_policy.build_budgeted_assignment(
        owner=c_owner.values,
        attachment_local=base.inputs.attachment_local,
        proxy_mass=frozen.mass_values,
        num_partitions=partitions,
    )
    metrics, histogram, shard_counts = assignment_metrics(
        budgeted, logical_count=logical_count, partitions=partitions
    )
    layout_sha = assignment_digest(budgeted.membership)
    metrics["assignment_jsonl_sha256"] = layout_sha
    p32_reproduction = {
        "required": partitions == 32,
        "owner_sha256_matches": (
            c_owner.sha256 == P32_OWNER_SHA256 if partitions == 32 else None
        ),
        "assignment_sha256_matches": (
            layout_sha == P32_ASSIGNMENT_SHA256 if partitions == 32 else None
        ),
        "membership_semantic_sha256_matches": (
            metrics["membership_semantic_sha256"] == P32_MEMBERSHIP_SHA256
            if partitions == 32
            else None
        ),
        "physical_point_count_matches": (
            metrics["physical_point_count"] == P32_PHYSICAL_POINT_COUNT
            if partitions == 32
            else None
        ),
    }
    if partitions == 32 and not all(
        value is True
        for key, value in p32_reproduction.items()
        if key != "required"
    ):
        raise RuntimeError(
            f"P=32 generalized BMR_10 path failed parity: {p32_reproduction}"
        )

    arm = native.ArmContract(
        arm="C_CNBR_BMR_10_scaling",
        balance_variant="cnbr",
        phase_a_role="fixed_p_scaling_candidate",
        phase_b_role="online_recall_validation_required",
        selected_adoption_candidate=False,
    )
    record = {
        "identity_all_pass": True,
        "topology_all_pass": True,
        "topology_metrics": scaling["C_CNBR"]["record"]["topology_metrics"],
        "graph_topology_gates": scaling["C_CNBR"]["record"]["topology_gates"],
        "query_topology_gates": {
            "online_recall_at_10": {
                "status": "REQUIRED_BEFORE_QPS_ACCEPTANCE",
                "threshold": 0.90,
            }
        },
        "physical_copy_load_improves_over_reference": False,
        "materialization_parity": {"assignment_bytes_sha256": layout_sha},
    }
    binding = replace(
        base,
        arm=arm,
        screen_path=scaling["manifest_path"],
        screen_sha256=scaling["manifest_sha256"],
        screen=scaling["manifest"],
        record=record,
        frozen=frozen,
        owner=c_owner,
        selection_path=None,
        selection_sha256=None,
        selection_source_code_record_sha256=None,
        construction_cost_path=None,
        construction_cost_sha256=None,
        construction_cost_gate=None,
    )
    output = args.output_dir.expanduser().resolve()
    result = native.materialize_bundle(
        binding,
        generation=int(args.generation),
        rebind_binary=require_file(args.rebind_binary, "rebind binary"),
        replay_verifier=require_file(args.replay_verifier, "upper replay verifier"),
        output_dir=output,
        chunk_size=100_000,
        tool_path=TOOL_PATH,
        parameter_overrides={
            "l1_partitioner": "C_CNBR",
            "balance_mode": "cnbr",
            "initial_num_shards": partitions,
            "multi_assign_max_shards": 2,
            "multi_assign_extra_copy_budget_numerator": 1,
            "multi_assign_extra_copy_budget_denominator": 10,
            "multi_assign_score": budgeted_policy.SCORE,
            "multi_assignment_policy": budgeted_policy.CANDIDATE_ID,
            "multi_assignment_policy_version": budgeted_policy.POLICY_VERSION,
        },
        extra_diagnostics={
            "scaling_extension": "fixed_p_c_cnbr_bmr10_v1",
            "scaling_owner_manifest_sha256": scaling["manifest_sha256"],
            "owner_sha256": c_owner.sha256,
            "l1_sizes": scaling["C_CNBR"]["record"]["partition_sizes"],
            "multi_assignment_contract": {
                "candidate_id": budgeted_policy.CANDIDATE_ID,
                "version": budgeted_policy.POLICY_VERSION,
                "extra_copy_budget_numerator": 1,
                "extra_copy_budget_denominator": 10,
                "maximum_copies_per_point": 2,
                "score": budgeted_policy.SCORE,
                "load_balance_owner_unchanged": True,
                "canonical_default_eligible": False,
            },
            "assignment_metrics": metrics,
            "p32_reproduction": p32_reproduction,
        },
        extra_provenance={
            "scaling_owner_manifest": str(scaling["manifest_path"]),
            "scaling_owner_manifest_sha256": scaling["manifest_sha256"],
            "bmr10_policy_source": str(Path(budgeted_policy.__file__).resolve()),
            "bmr10_policy_source_sha256": sha256(
                Path(budgeted_policy.__file__).resolve()
            ),
            "p32_reference_build_manifest": str(args.p32_bmr_manifest.resolve()),
            "p32_reference_build_manifest_sha256": sha256(
                require_file(args.p32_bmr_manifest, "P=32 BMR reference manifest")
            ),
        },
        assignment_builder=assignment_builder(
            budgeted,
            expected_digest=layout_sha,
            histogram=histogram,
            shard_counts=shard_counts,
        ),
    )
    return {
        **result,
        "status": "PASS",
        "partitions": partitions,
        "assignment_metrics": metrics,
        "p32_reproduction": p32_reproduction,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    owner = subparsers.add_parser("owner")
    owner.add_argument("--partitions", type=int, required=True)
    owner.add_argument("--phase-a-manifest", type=Path, default=DEFAULT_PHASE_A)
    owner.add_argument("--output-dir", type=Path, required=True)

    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--partitions", type=int, required=True)
    bundle.add_argument("--owner-dir", type=Path, required=True)
    bundle.add_argument("--phase-b-screen", type=Path, default=DEFAULT_PHASE_B)
    bundle.add_argument(
        "--cnbr-selection-manifest", type=Path, default=DEFAULT_CNBR_SELECTION
    )
    bundle.add_argument(
        "--construction-cost-audit", type=Path, default=DEFAULT_COST_AUDIT
    )
    bundle.add_argument("--p32-bmr-manifest", type=Path, default=DEFAULT_P32_BMR)
    bundle.add_argument("--generation", type=int, required=True)
    bundle.add_argument("--rebind-binary", type=Path, required=True)
    bundle.add_argument("--replay-verifier", type=Path, required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_owner(args) if args.command == "owner" else materialize_bundle(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
