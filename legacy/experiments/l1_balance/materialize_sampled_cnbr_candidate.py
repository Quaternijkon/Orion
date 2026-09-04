#!/usr/bin/env python3
"""Materialize the fixed sampled-CNBR research candidate for online A/B.

The candidate is deliberately not promoted to the canonical CNBR contract.  It
uses the frozen formal GloVe upper graph and native owner, reads exactly one
deterministic 0.25% sample of the already exported L0 attachment rows, blends
that proxy at 25% with the 75% upper-only prior, and applies the unchanged CNBR
9/4 repair.  Orion's enabled/2/0/0 multi-assignment remains a later, orthogonal
materialization stage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance import evaluate_cnbr_frozen_owners as phase_b  # noqa: E402
from experiments.l1_balance import materialize_native_cnbr_candidate as native  # noqa: E402
from experiments.l1_balance import sampled_fill_screen as sampled  # noqa: E402
from experiments.l1_balance.run_offline_screen import load_upper  # noqa: E402


FORMAT_VERSION = 1
TOOL_PATH = "experiments/l1_balance/materialize_sampled_cnbr_candidate.py"
ARM_NAME = "sampled_CNBR"
BALANCE_VARIANT = "sampled_cnbr_research"
EXPECTED_FRACTION = 0.0025
EXPECTED_SEED = 100
EXPECTED_SAMPLE_WEIGHT = 0.25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-manifest", required=True)
    parser.add_argument("--phase-b-screen", required=True)
    parser.add_argument("--sampled-screen", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--construction-cost-audit", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--rebind-binary", required=True)
    parser.add_argument("--replay-verifier", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--fraction", type=float, default=EXPECTED_FRACTION)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--sample-weight", type=float, default=EXPECTED_SAMPLE_WEIGHT)
    return parser.parse_args(argv)


def sha256_path(path: Path) -> str:
    return native.sha256_path(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json_new(path: Path, value: Any) -> None:
    native.write_json_new(path, jsonable(value))


def require_frozen_file(raw: str, label: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def selected_screen_row(
    screen_path: Path,
    *,
    fraction: float,
    seed: int,
    sample_weight: float,
) -> tuple[dict[str, Any], str]:
    screen = native.load_json_object(screen_path, "sampled-CNBR screen")
    rows = screen.get("rows")
    if not isinstance(rows, list):
        raise ValueError("sampled-CNBR screen lacks rows")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("dataset") == "glove"
        and float(row.get("fraction_requested", -1.0)) == fraction
        and int(row.get("seed", -1)) == seed
        and float(row.get("sample_weight", -1.0)) == sample_weight
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one selected GloVe sampled row, found {len(matches)}")
    row = dict(matches[0])
    required_true = (
        "offline_eligible",
        "graph_topology_all_pass",
        "query_topology_all_pass",
        "incremental_cost_gate_pass",
        "physical_max_over_mean_better_than_current_cnbr",
        "physical_cv_no_worse_than_current_cnbr",
    )
    if any(row.get(key) is not True for key in required_true):
        raise ValueError("selected sampled-CNBR row did not pass every offline gate")
    return row, sha256_path(screen_path)


def compute_assignment_parity(
    owner: np.ndarray,
    attachment_local: np.ndarray,
    partitions: int,
    chunk_size: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    primary_digest = hashlib.sha256()
    physical_loads = np.zeros(partitions, dtype=np.int64)
    primary_loads = np.zeros(partitions, dtype=np.int64)
    copy_histogram: Counter[int] = Counter()
    physical_count = 0
    rows = len(attachment_local)
    for start in range(0, rows, chunk_size):
        stop = min(rows, start + chunk_size)
        membership, copy_count, primary = native.compact_membership(
            owner,
            np.asarray(attachment_local[start:stop]),
            partitions,
        )
        physical_loads += membership.sum(axis=0, dtype=np.int64)
        primary_loads += np.bincount(primary, minlength=partitions).astype(
            np.int64, copy=False
        )
        primary_digest.update(
            np.ascontiguousarray(primary, dtype="<i4").tobytes(order="C")
        )
        values, counts = np.unique(copy_count, return_counts=True)
        copy_histogram.update(
            {
                int(value): int(count)
                for value, count in zip(values.tolist(), counts.tolist(), strict=True)
            }
        )
        physical_count += int(copy_count.sum())
        for offset, row in enumerate(membership):
            shards = np.flatnonzero(row).astype(int).tolist()
            digest.update(native.assignment_bytes(start + offset, shards))
    return {
        "canonical_format": native.ASSIGNMENT_FORMAT,
        "assignment_bytes_sha256": digest.hexdigest(),
        "logical_point_count": rows,
        "physical_point_count": physical_count,
        "primary_shards_sha256": primary_digest.hexdigest(),
        "primary_shard_loads": [int(value) for value in primary_loads],
        "physical_copy_shard_loads": [int(value) for value in physical_loads],
        "copy_count_histogram": {
            str(key): int(value) for key, value in sorted(copy_histogram.items())
        },
    }


def build_source_context(
    phase_a_path: Path,
    phase_b_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    phase_a_manifest = native.load_json_object(phase_a_path, "formal Phase-A")
    phase_b_screen = native.load_json_object(phase_b_path, "formal Phase-B")
    dataset = sampled._load_dataset("glove", phase_a_path, phase_b_path)
    return phase_a_manifest, phase_b_screen, dataset


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.fraction != EXPECTED_FRACTION
        or args.seed != EXPECTED_SEED
        or args.sample_weight != EXPECTED_SAMPLE_WEIGHT
    ):
        raise ValueError(
            "this research materializer is frozen to fraction=0.0025, seed=100, "
            "sample_weight=0.25"
        )
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")

    phase_a_path = require_frozen_file(args.phase_a_manifest, "formal Phase-A")
    phase_b_path = require_frozen_file(args.phase_b_screen, "formal Phase-B")
    sampled_screen_path = require_frozen_file(args.sampled_screen, "sampled screen")
    selection_path = require_frozen_file(args.selection_manifest, "CNBR selection")
    cost_path = require_frozen_file(
        args.construction_cost_audit, "CNBR construction-cost audit"
    )
    rebind_binary = require_frozen_file(args.rebind_binary, "rebind binary")
    replay_verifier = require_frozen_file(args.replay_verifier, "replay verifier")
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if evidence_dir.exists() or output_dir.exists():
        raise FileExistsError("evidence-dir and output-dir must both be fresh")
    if not evidence_dir.parent.is_dir() or not output_dir.parent.is_dir():
        raise FileNotFoundError("evidence/output parent directory is missing")

    selected_row, sampled_screen_sha256 = selected_screen_row(
        sampled_screen_path,
        fraction=args.fraction,
        seed=args.seed,
        sample_weight=args.sample_weight,
    )
    phase_a_manifest, phase_b_screen, dataset = build_source_context(
        phase_a_path, phase_b_path
    )
    source_artifact_record = phase_a_manifest["construction_inputs"]["artifact"]
    source_artifact_path = Path(source_artifact_record["path"]).resolve()
    if sha256_path(source_artifact_path) != source_artifact_record["sha256"]:
        raise ValueError("source upper-only artifact checksum drifted")
    source_artifact, _adjacency, source_vectors, labels, _entry, edges_l, edges_r, _nav = (
        load_upper(source_artifact_path)
    )
    if not np.array_equal(labels, dataset["labels"]):
        raise ValueError("sampled dataset and source artifact upper labels differ")

    indices = sampled.deterministic_sample_indices(
        dataset["row_count"], args.fraction, args.seed
    )
    sample_local = np.asarray(dataset["attachment_local"][indices]).copy()
    raw_sample_mass = sampled.occurrence_mass(sample_local, dataset["upper_count"])
    blended_mass = sampled.blended_proxy_mass(
        raw_sample_mass, dataset["upper_mass"], args.sample_weight
    )
    result = sampled.core.build_cnbr_owner_with_external_mass_for_research(
        dataset["adjacency"],
        dataset["navigation_local"],
        blended_mass,
        dataset["owners"]["N_native"],
        sampled.NUM_PARTITIONS,
    )
    owner = np.asarray(result.owner, dtype="<i4")
    owner_sha256 = sampled.owner_sha256(owner)
    if owner_sha256 != selected_row.get("owner_sha256"):
        raise ValueError(
            "regenerated sampled owner differs from the selected offline screen"
        )
    full_metrics, _full_loads = sampled._evaluate_owner(
        owner,
        dataset["attachment_local"],
        dataset["query_local"],
        dataset["ground_truth"],
        dataset["labels"],
        dynamic_ef_base=int(source_artifact["dynamic_ef_base"]),
        dynamic_ef_factor=int(source_artifact["dynamic_ef_factor"]),
    )
    n_metrics, _n_loads = sampled._evaluate_owner(
        dataset["owners"]["N_native"],
        dataset["attachment_local"],
        dataset["query_local"],
        dataset["ground_truth"],
        dataset["labels"],
        dynamic_ef_base=int(source_artifact["dynamic_ef_base"]),
        dynamic_ef_factor=int(source_artifact["dynamic_ef_factor"]),
    )
    query_gates = phase_b.query_gate_results(
        full_metrics, n_metrics, dataset["query_count"]
    )
    graph_gates = jsonable(result.topology_gates)
    if not all(bool(record["pass"]) for record in graph_gates.values()):
        raise ValueError("regenerated sampled owner failed a graph topology gate")
    if not all(bool(record["pass"]) for record in query_gates.values()):
        raise ValueError("regenerated sampled owner failed a query topology gate")
    for key in (
        "physical_point_count",
        "physical_copy_load_max",
        "routed_shards_mean",
        "gt_routing_coverage_mean",
    ):
        if not np.isclose(float(full_metrics[key]), float(selected_row[key])):
            raise ValueError(f"regenerated sampled metric drifted: {key}")

    parity = compute_assignment_parity(
        owner,
        dataset["attachment_local"],
        sampled.NUM_PARTITIONS,
        args.chunk_size,
    )
    if parity["physical_point_count"] != int(full_metrics["physical_point_count"]):
        raise ValueError("assignment replay physical count differs from offline screen")

    evidence_dir.mkdir(parents=False, exist_ok=False)
    owner_path = evidence_dir / "sampled-cnbr.owner.i32le"
    owner.tofile(owner_path)
    sample_indices_path = evidence_dir / "sample-indices.i64le"
    np.asarray(indices, dtype="<i8").tofile(sample_indices_path)
    sample_mass_path = evidence_dir / "sample-occurrence-mass.u64le"
    np.asarray(raw_sample_mass, dtype="<u8").tofile(sample_mass_path)
    blended_mass_path = evidence_dir / "blended-proxy-mass.u64le"
    np.asarray(blended_mass, dtype="<u8").tofile(blended_mass_path)

    owner_sizes = [
        int(value)
        for value in np.bincount(owner, minlength=sampled.NUM_PARTITIONS).tolist()
    ]
    evidence_path = evidence_dir / "sampled-owner-evidence.json"
    evidence = {
        "format_version": FORMAT_VERSION,
        "record_type": "sampled_cnbr_research_owner",
        "status": "PASS",
        "canonical_default_eligible": False,
        "dataset": "glove-200-angular",
        "method": ARM_NAME,
        "balance_variant": BALANCE_VARIANT,
        "parameters": {
            "fraction": args.fraction,
            "seed": args.seed,
            "sample_weight": args.sample_weight,
            "upper_prior_weight": 1.0 - args.sample_weight,
            "cnbr_ratio": [9, 4],
            "num_partitions": sampled.NUM_PARTITIONS,
            "multi_assignment": sampled.MULTI_ASSIGNMENT_CONTRACT,
        },
        "partition_sizes": owner_sizes,
        "owner": native.file_record(owner_path, owner_sha256),
        "owner_path": str(owner_path),
        "owner_sha256": owner_sha256,
        "sample_indices": native.file_record(sample_indices_path),
        "raw_sample_mass": native.file_record(sample_mass_path),
        "blended_proxy_mass": native.file_record(blended_mass_path),
        "source": {
            "phase_a_manifest": str(phase_a_path),
            "phase_a_manifest_sha256": sha256_path(phase_a_path),
            "phase_b_screen": str(phase_b_path),
            "phase_b_screen_sha256": sha256_path(phase_b_path),
            "sampled_screen": str(sampled_screen_path),
            "sampled_screen_sha256": sampled_screen_sha256,
            "source_artifact": str(source_artifact_path),
            "source_artifact_sha256": sha256_path(source_artifact_path),
            "native_owner_sha256": sampled.owner_sha256(
                dataset["owners"]["N_native"]
            ),
            "current_cnbr_owner_sha256": sampled.owner_sha256(
                dataset["owners"]["C_CNBR"]
            ),
            "attachments_sha256": phase_b_screen[
                "post_freeze_evaluation_inputs"
            ]["attachments_sha256"],
        },
        "source_code": {
            path.relative_to(REPO_ROOT).as_posix(): native.file_record(path)
            for path in (
                Path(sampled.__file__).resolve(),
                Path(sampled.core.__file__).resolve(),
                Path(__file__).resolve(),
            )
        },
        "sample_index_sha256": selected_row["sample_index_sha256"],
        "topology_metrics": jsonable(result.topology_metrics),
        "graph_topology_gates": graph_gates,
        "query_topology_gates": jsonable(query_gates),
        "full_metrics": jsonable(full_metrics),
        "selected_offline_row": selected_row,
        "materialization_parity": parity,
    }
    write_json_new(evidence_path, evidence)
    evidence_sha256 = sha256_path(evidence_path)
    evidence_sidecar = evidence_path.with_name(evidence_path.name + ".sha256")
    evidence_sidecar.write_text(evidence_sha256 + "\n", encoding="ascii")
    for path in evidence_dir.iterdir():
        os.chmod(path, 0o444)

    phase_a_performance_path = phase_a_path.with_name("phase-a.performance.json")
    fixed_parameters = phase_a_manifest.get("fixed_parameters") or {}
    frozen = phase_b.FrozenPhaseA(
        manifest_path=phase_a_path,
        manifest_sha256=sha256_path(phase_a_path),
        manifest=phase_a_manifest,
        artifact_path=source_artifact_path,
        artifact=source_artifact,
        labels=np.asarray(labels),
        vectors=np.asarray(source_vectors),
        edge_left=np.asarray(edges_l),
        edge_right=np.asarray(edges_r),
        num_partitions=sampled.NUM_PARTITIONS,
        candidate_set="sampled-research",
        candidate_ratios={ARM_NAME: (9, 4)},
        selected_candidate_name=ARM_NAME,
        mass_values=np.asarray(blended_mass),
        owners={},
        performance_path=phase_a_performance_path,
        performance_sha256=sha256_path(phase_a_performance_path),
        performance=native.load_json_object(
            phase_a_performance_path, "Phase-A performance"
        ),
        frozen_files=(),
    )

    post = phase_b_screen["post_freeze_evaluation_inputs"]
    dataset_manifest_path = Path(post["dataset_manifest"]).resolve()
    dataset_manifest = native.load_json_object(dataset_manifest_path, "dataset manifest")
    source_build_manifest_path = Path(post["source_build_manifest"]).resolve()
    attachments_path = Path(post["attachments"]).resolve()
    attachments_manifest_path = Path(post["attachments_manifest"]).resolve()
    query_hits_path = Path(post["query_hits"]).resolve()
    query_manifest_path = Path(post["query_hits_manifest"]).resolve()
    ground_truth_path = Path(post["ground_truth"]).resolve()
    ground_truth_manifest_path = Path(post["ground_truth_manifest"]).resolve()
    inputs = phase_b.PhaseBInputs(
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=sha256_path(dataset_manifest_path),
        dataset_sha256=str(dataset_manifest["dataset"]["sha256"]),
        source_build_manifest_path=source_build_manifest_path,
        source_build_manifest_sha256=sha256_path(source_build_manifest_path),
        attachments_path=attachments_path,
        attachments_manifest_path=attachments_manifest_path,
        attachments_sha256=sha256_path(attachments_path),
        attachment_local=np.asarray(dataset["attachment_local"]),
        query_hits_path=query_hits_path,
        query_manifest_path=query_manifest_path,
        query_hits_sha256=sha256_path(query_hits_path),
        query_manifest=native.load_json_object(query_manifest_path, "query manifest"),
        query_local=np.asarray(dataset["query_local"]),
        ground_truth_path=ground_truth_path,
        ground_truth_manifest_path=ground_truth_manifest_path,
        ground_truth_manifest_sha256=sha256_path(ground_truth_manifest_path),
        ground_truth_sha256=sha256_path(ground_truth_path),
        ground_truth=np.asarray(dataset["ground_truth"]),
    )
    source_dataset_path = Path(dataset_manifest["source_dataset"]["path"]).resolve()
    attachment_manifest = native.load_json_object(
        attachments_manifest_path, "attachment manifest"
    )
    vectors_path = Path(attachment_manifest["vectors_path"]).resolve()
    replay_queries_path = Path(dataset_manifest["files"]["queries"]["path"]).resolve()

    owner_record = native.load_json_object(evidence_path, "sampled owner evidence")
    frozen_owner = phase_b.FrozenOwner(
        name=ARM_NAME,
        role="research_candidate",
        record=owner_record,
        path=owner_path,
        sha256=owner_sha256,
        values=owner,
        owner_record_path=evidence_path,
        owner_record_sha256=evidence_sha256,
        owner_record=owner_record,
    )
    arm = native.ArmContract(
        arm=ARM_NAME,
        balance_variant=BALANCE_VARIANT,
        phase_a_role="sampled_research_derivation",
        phase_b_role="research_candidate",
        selected_adoption_candidate=False,
    )
    record = {
        "materialization_parity": parity,
        "identity_all_pass": True,
        "topology_all_pass": True,
        "topology_metrics": jsonable(result.topology_metrics),
        "graph_topology_gates": graph_gates,
        "query_topology_gates": jsonable(query_gates),
        "physical_copy_load_improves_over_reference": True,
    }
    binding = native.MaterializationBinding(
        arm=arm,
        screen_path=sampled_screen_path,
        screen_sha256=sampled_screen_sha256,
        screen=native.load_json_object(sampled_screen_path, "sampled screen"),
        record=record,
        frozen=frozen,
        inputs=inputs,
        owner=frozen_owner,
        source_artifact_path=source_artifact_path,
        source_artifact_sha256=sha256_path(source_artifact_path),
        vectors_path=vectors_path,
        vectors_sha256=sha256_path(vectors_path),
        source_build_manifest_path=source_build_manifest_path,
        source_build_manifest_sha256=sha256_path(source_build_manifest_path),
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=sha256_path(dataset_manifest_path),
        dataset_path=source_dataset_path,
        dataset_sha256=sha256_path(source_dataset_path),
        replay_queries_path=replay_queries_path,
        replay_query_rows=int(dataset_manifest["dataset"]["expected_query_rows"]),
        forbidden_stage_invocations=dict(
            phase_a_manifest["forbidden_stage_invocations"]
        ),
        evaluator_source_code_record_sha256=evidence_sha256,
        selection_path=None,
        selection_sha256=None,
        selection_source_code_record_sha256=None,
        construction_cost_path=None,
        construction_cost_sha256=None,
        construction_cost_gate=None,
    )

    source_mode = stat.S_IMODE(vectors_path.stat().st_mode)
    result_bundle = native.materialize_bundle(
        binding,
        generation=args.generation,
        rebind_binary=rebind_binary,
        replay_verifier=replay_verifier,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        tool_path=TOOL_PATH,
        parameter_overrides={
            "l1_partitioner": ARM_NAME,
            "balance_mode": BALANCE_VARIANT,
            "sample_fill_fraction": args.fraction,
            "sample_fill_seed": args.seed,
            "sample_fill_weight": args.sample_weight,
            "upper_prior_weight": 1.0 - args.sample_weight,
        },
        extra_diagnostics={
            "input_scope": (
                "production_upper_graph_upper_self_navigation_and_sampled_l0_attachments"
            ),
            "canonical_default_eligible": False,
            "offline_research_only": True,
            "sampled_owner_evidence_sha256": evidence_sha256,
            "sampled_attachment_rows_read": int(len(indices)),
            "sampled_attachment_fraction": args.fraction,
            "materialization_parity": parity,
        },
        extra_provenance={
            "sampled_owner_evidence": str(evidence_path),
            "sampled_owner_evidence_sha256": evidence_sha256,
            "sampled_screen": str(sampled_screen_path),
            "sampled_screen_sha256": sampled_screen_sha256,
            "source_formal_phase_a": str(phase_a_path),
            "source_formal_phase_a_sha256": sha256_path(phase_a_path),
            "source_formal_phase_b": str(phase_b_path),
            "source_formal_phase_b_sha256": sha256_path(phase_b_path),
            "source_cnbr_selection": str(selection_path),
            "source_cnbr_selection_sha256": sha256_path(selection_path),
            "source_cnbr_construction_cost_audit": str(cost_path),
            "source_cnbr_construction_cost_audit_sha256": sha256_path(cost_path),
        },
    )
    if stat.S_IMODE(vectors_path.stat().st_mode) != source_mode:
        raise RuntimeError("sampled materialization changed canonical vector mode")
    return {
        **result_bundle,
        "evidence": str(evidence_path),
        "evidence_sha256": evidence_sha256,
        "owner_sha256": owner_sha256,
        "offline_physical_max_over_mean": full_metrics[
            "physical_copy_load_max_over_mean"
        ],
        "offline_routed_shards_mean": full_metrics["routed_shards_mean"],
    }


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
