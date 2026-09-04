#!/usr/bin/env python3
"""Materialize C_CNBR with the frozen BMR_10 multi-assignment policy."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance import materialize_native_cnbr_candidate as native  # noqa: E402
from experiments.l1_balance import native_cnbr_core  # noqa: E402
from experiments.l1_balance import cnbr_construction_cost_gate as cost_gate  # noqa: E402
from experiments.multi_assignment import budgeted_policy  # noqa: E402


TOOL_PATH = "experiments/multi_assignment/materialize_budgeted_candidate.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-b-screen", required=True)
    parser.add_argument("--cnbr-selection-manifest", required=True)
    parser.add_argument("--construction-cost-audit", required=True)
    parser.add_argument("--bmr-selection-manifest", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--generation", type=int)
    parser.add_argument("--rebind-binary")
    parser.add_argument("--replay-verifier")
    parser.add_argument("--output-dir")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def load_bmr_selection(
    path: Path, binding: native.MaterializationBinding
) -> tuple[dict[str, Any], str]:
    digest = native.sha256_path(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != digest:
        raise ValueError("BMR_10 selection checksum sidecar mismatch")
    native.require_read_only(path, "BMR_10 selection manifest")
    native.require_read_only(sidecar, "BMR_10 selection sidecar")
    selection = native.load_json_object(path, "BMR_10 selection manifest")
    expected = {
        "format_version": 1,
        "record_type": "fixed_cnbr_bmr10_selection",
        "status": "PASS",
        "selection_status": "post_exploratory_sift_and_glove_tuning_selection",
        "strict_glove_holdout_claim": False,
        "glove_online_measurement_queries_seen": False,
        "selected_candidate": budgeted_policy.CANDIDATE_ID,
        "fallback": "C_CNBR_current_all_max",
        "no_fallback_retuning": True,
    }
    drift = {
        key: {"expected": value, "actual": selection.get(key)}
        for key, value in expected.items()
        if selection.get(key) != value
    }
    if drift:
        raise ValueError(f"BMR_10 selection contract drifted: {drift}")
    rows = selection.get("rows")
    if not isinstance(rows, dict) or not isinstance(
        rows.get("glove-200-angular"), dict
    ):
        raise ValueError("BMR_10 selection lacks GloVe evidence")
    glove = rows["glove-200-angular"]
    if glove.get("owner_sha256") != binding.owner.sha256:
        raise ValueError("BMR_10 selection binds a different C_CNBR owner")
    candidate = glove.get("candidate")
    if not isinstance(candidate, dict) or float(candidate["expansion_ratio"]) > 1.10:
        raise ValueError("BMR_10 selection lost its low-expansion gate")
    formal_record = selection["inputs"]["glove-200-angular"]
    formal_path = Path(formal_record["formal_evaluation"]).expanduser().resolve()
    if native.sha256_path(formal_path) != formal_record["formal_evaluation_sha256"]:
        raise ValueError("BMR_10 formal GloVe evidence checksum drifted")
    formal = native.load_json_object(formal_path, "BMR_10 formal GloVe evidence")
    if (
        formal["inputs"]["phase_b_screen_sha256"] != binding.screen_sha256
        or formal["causal_boundary"]["owner_sha256"] != binding.owner.sha256
        or formal["candidate"] != candidate
        or formal.get("all_gates_pass") is not True
    ):
        raise ValueError("BMR_10 formal evidence differs from materialization inputs")
    source_code = selection.get("source_code")
    current_sources = {
        "synthesizer": REPO_ROOT
        / "experiments/multi_assignment/synthesize_budgeted_selection.py",
        "policy": Path(budgeted_policy.__file__).resolve(),
        "addendum": REPO_ROOT
        / "experiments/multi_assignment/BUDGETED_REPLICATION_ADDENDUM.md",
    }
    if not isinstance(source_code, dict) or set(source_code) != set(current_sources):
        raise ValueError("BMR_10 selection source-code set drifted")
    for name, current in current_sources.items():
        record = source_code[name]
        if (
            record.get("sha256") != native.sha256_path(current)
            or record.get("size_bytes") != current.stat().st_size
        ):
            raise ValueError(f"BMR_10 selection source drifted: {name}")
    return selection, digest


def assignment_builder(
    selection: dict[str, Any],
) -> Callable[[native.MaterializationBinding, Path, int], native.AssignmentResult]:
    expected_metrics = selection["rows"]["glove-200-angular"]["candidate"]

    def build(
        binding: native.MaterializationBinding,
        output_dir: Path,
        chunk_size: int,
    ) -> native.AssignmentResult:
        del chunk_size
        mass_record = binding.frozen.manifest["mass"]["values"]
        mass_path = (
            binding.frozen.manifest_path.parent / mass_record["path"]
        ).resolve()
        if native.sha256_path(mass_path) != mass_record["sha256"]:
            raise ValueError("BMR_10 proxy-mass checksum drifted")
        proxy_mass = np.memmap(
            mass_path,
            dtype="<u8",
            mode="r",
            shape=(int(mass_record["row_count"]),),
        )
        assignment = budgeted_policy.build_budgeted_assignment(
            owner=binding.owner.values,
            attachment_local=binding.inputs.attachment_local,
            proxy_mass=proxy_mass,
            num_partitions=int(binding.frozen.num_partitions),
        )
        membership = assignment.membership
        row_count = int(binding.frozen.artifact["logical_point_count"])
        partitions = int(binding.frozen.num_partitions)
        if membership.shape != (row_count, partitions):
            raise AssertionError("BMR_10 membership shape drifted")
        copy_count = membership.sum(axis=1, dtype=np.int16)
        shard_counts = membership.sum(axis=0, dtype=np.int64)
        physical_point_count = int(copy_count.sum())
        values, counts = np.unique(copy_count, return_counts=True)
        copy_histogram = Counter(
            {
                int(value): int(count)
                for value, count in zip(
                    values.tolist(), counts.tolist(), strict=True
                )
            }
        )
        observed = {
            "physical_point_count": physical_point_count,
            "expansion_ratio": physical_point_count / row_count,
            "copy_count_histogram": {
                str(key): int(value)
                for key, value in sorted(copy_histogram.items())
            },
            "physical_copy_load_min": int(shard_counts.min()),
            "physical_copy_load_max": int(shard_counts.max()),
            "physical_copy_load_mean": float(shard_counts.mean()),
            "physical_copy_load_cv": float(
                shard_counts.std() / shard_counts.mean()
            ),
            "physical_copy_load_max_over_mean": float(
                shard_counts.max() / shard_counts.mean()
            ),
            "membership_semantic_sha256": assignment.membership_semantic_sha256,
        }
        drift = {
            key: {"expected": expected_metrics.get(key), "observed": value}
            for key, value in observed.items()
            if expected_metrics.get(key) != value
        }
        if drift:
            raise ValueError(f"BMR_10 materialization metrics drifted: {drift}")

        upper_labels = np.asarray(binding.frozen.labels, dtype=np.int64)
        upper_position = np.full(row_count, -1, dtype=np.int32)
        upper_position[upper_labels] = np.arange(len(upper_labels), dtype=np.int32)
        upper_memberships: list[list[int] | None] = [None] * len(upper_labels)
        assignment_path = output_dir / "orion_numeric_import.assignments.jsonl"
        temporary_path = assignment_path.with_name(assignment_path.name + ".tmp")
        assignment_digest = hashlib.sha256()
        primary_digest = hashlib.sha256()
        primary_shard_loads = np.zeros(partitions, dtype=np.int64)
        try:
            with temporary_path.open("xb") as handle:
                for point_id, row in enumerate(membership):
                    shards = np.flatnonzero(row).astype(int).tolist()
                    encoded = native.assignment_bytes(point_id, shards)
                    handle.write(encoded)
                    assignment_digest.update(encoded)
                    ordered_primary = int(shards[0])
                    primary_digest.update(
                        np.asarray([ordered_primary], dtype="<i4").tobytes()
                    )
                    primary_shard_loads[ordered_primary] += 1
                    upper_index = int(upper_position[point_id])
                    if upper_index >= 0:
                        upper_memberships[upper_index] = shards
            layout_sha256 = assignment_digest.hexdigest()
            if layout_sha256 != expected_metrics["assignment_jsonl_sha256"]:
                raise ValueError("BMR_10 assignment JSONL checksum drifted")
            os.replace(temporary_path, assignment_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        if any(row is None for row in upper_memberships):
            raise RuntimeError("BMR_10 did not assign every upper point")
        if native.sha256_path(assignment_path) != layout_sha256:
            raise RuntimeError("BMR_10 published assignment checksum drifted")
        return native.AssignmentResult(
            path=assignment_path,
            layout_sha256=layout_sha256,
            physical_point_count=physical_point_count,
            shard_counts=shard_counts,
            copy_histogram=copy_histogram,
            primary_shards_sha256=primary_digest.hexdigest(),
            primary_shard_loads=primary_shard_loads,
            upper_memberships=[
                list(row) for row in upper_memberships if row is not None
            ],
        )

    return build


def run(args: argparse.Namespace) -> dict[str, Any]:
    formal_screen = Path(args.phase_b_screen).expanduser().resolve()
    frozen_core = formal_screen.parent / "source" / "native_cnbr_core.py"
    if not frozen_core.is_file():
        raise FileNotFoundError(f"frozen Phase-B core is missing: {frozen_core}")
    original_core_file = native_cnbr_core.__file__
    original_cost_source_validator = cost_gate._validate_current_source_bindings
    try:
        # The formal evidence must be replayed against its immutable source
        # snapshot.  Today's working tree contains later additive research and
        # is not the execution source for the already-frozen CNBR owner/cost
        # decision.
        native_cnbr_core.__file__ = str(frozen_core)
        cost_gate._validate_current_source_bindings = lambda _audit: None
        binding = native.validate_phase_b_screen(
            formal_screen,
            "C_CNBR",
            args.cnbr_selection_manifest,
            args.construction_cost_audit,
        )
    finally:
        native_cnbr_core.__file__ = original_core_file
        cost_gate._validate_current_source_bindings = original_cost_source_validator
    selection_path = native.require_file(
        args.bmr_selection_manifest, "BMR_10 selection manifest"
    )
    selection, selection_sha256 = load_bmr_selection(selection_path, binding)
    validation = {
        "status": "PASS",
        "candidate": budgeted_policy.CANDIDATE_ID,
        "phase_a_owner_sha256": binding.owner.sha256,
        "phase_b_screen_sha256": binding.screen_sha256,
        "bmr_selection_sha256": selection_sha256,
        "logical_point_count": int(binding.frozen.artifact["logical_point_count"]),
        "shard_count": int(binding.frozen.num_partitions),
    }
    if args.validate_only:
        return validation
    missing = [
        name
        for name in ("generation", "rebind_binary", "replay_verifier", "output_dir")
        if getattr(args, name, None) in (None, "")
    ]
    if missing:
        raise ValueError("materialization requires arguments: " + ", ".join(missing))
    return native.materialize_bundle(
        binding,
        generation=int(args.generation),
        rebind_binary=native.require_file(args.rebind_binary, "rebind binary"),
        replay_verifier=native.require_file(
            args.replay_verifier, "upper replay verifier"
        ),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        chunk_size=int(args.chunk_size),
        tool_path=TOOL_PATH,
        parameter_overrides={
            "multi_assignment_policy": budgeted_policy.CANDIDATE_ID,
            "multi_assignment_policy_version": budgeted_policy.POLICY_VERSION,
            "multi_assign_extra_copy_budget_numerator": (
                budgeted_policy.EXTRA_COPY_BUDGET_NUMERATOR
            ),
            "multi_assign_extra_copy_budget_denominator": (
                budgeted_policy.EXTRA_COPY_BUDGET_DENOMINATOR
            ),
            "multi_assign_score": budgeted_policy.SCORE,
            "multi_assign_max_shards": 2,
        },
        extra_diagnostics={
            "multi_assignment_contract": {
                "candidate_id": budgeted_policy.CANDIDATE_ID,
                "version": budgeted_policy.POLICY_VERSION,
                "extra_copy_budget_numerator": (
                    budgeted_policy.EXTRA_COPY_BUDGET_NUMERATOR
                ),
                "extra_copy_budget_denominator": (
                    budgeted_policy.EXTRA_COPY_BUDGET_DENOMINATOR
                ),
                "score": budgeted_policy.SCORE,
                "maximum_copies_per_point": 2,
                "load_balance_owner_unchanged": True,
                "canonical_default_eligible": False,
            },
            "golden_assignment_bytes_sha256": selection["rows"][
                "glove-200-angular"
            ]["candidate"]["assignment_jsonl_sha256"],
            "input_scope": (
                "frozen_cnbr_owner_upper_proxy_mass_and_normal_l0_attachments"
            ),
        },
        extra_provenance={
            "bmr10_selection_manifest": str(selection_path),
            "bmr10_selection_manifest_sha256": selection_sha256,
            "bmr10_policy_source": str(Path(budgeted_policy.__file__).resolve()),
            "bmr10_policy_source_sha256": native.sha256_path(
                Path(budgeted_policy.__file__).resolve()
            ),
        },
        assignment_builder=assignment_builder(selection),
    )


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
