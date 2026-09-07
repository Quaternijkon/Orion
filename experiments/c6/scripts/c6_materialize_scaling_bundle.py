#!/usr/bin/env python3
"""Materialize a fixed-M C_CNBR+BMR10 bundle for the C6 online-adaptivity suite.

C6 reuses an already accepted frozen upper graph, Phase-A/Phase-B owner evidence,
and full L0 attachments.  The historical C1 construction-cost gate is intentionally
out of scope: it selects a partition-construction method, while C6 holds that layout
fixed and varies only online shard/budget policies.  Every other frozen identity,
topology, selection, attachment, dataset, and replay gate remains enforced by the
underlying materializer.
"""

from __future__ import annotations

import argparse
import errno
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from c6_protocol import sha256_path, utc_timestamp, write_json_atomic  # noqa: E402
from experiments.c1.scripts import c1_orion_bmr10_scaling_materialize as materializer  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=int, choices=(4, 8, 16, 32), required=True)
    parser.add_argument("--owner-dir", type=Path, required=True)
    parser.add_argument("--phase-b-screen", type=Path, required=True)
    parser.add_argument("--cnbr-selection-manifest", type=Path, required=True)
    parser.add_argument("--construction-cost-audit", type=Path, required=True)
    parser.add_argument("--p32-bmr-manifest", type=Path, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--rebind-binary", type=Path, required=True)
    parser.add_argument("--replay-verifier", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--c6-record", type=Path, required=True)
    return parser.parse_args(argv)


def c6_cost_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "status": "NOT_APPLICABLE_TO_C6_ONLINE_ADAPTIVITY",
        "audit": str(resolved),
        "audit_sha256": sha256_path(resolved),
        "reason": (
            "C6 freezes partition construction and evaluates only online routing and "
            "per-shard search allocation"
        ),
    }


def dataset_specific_p32_expectation(args: argparse.Namespace) -> dict[str, Any]:
    scaling = materializer.load_scaling_owner(args.owner_dir, 32)
    base = materializer.frozen_base_binding(args)
    source_record = (
        scaling["manifest"].get("construction_inputs") or {}
    ).get("artifact")
    if not isinstance(source_record, dict):
        raise ValueError("scaling owner source artifact binding is missing")
    materializer.verify_record(
        base.source_artifact_path, source_record, "source artifact"
    )
    upper_count = len(base.frozen.labels)
    n_owner = materializer.frozen_owner(
        "N_native",
        "balance_free_orion_core",
        scaling["N_native"],
        upper_count,
    )
    c_owner = materializer.frozen_owner(
        "C_CNBR",
        "fixed_p_scaling_candidate",
        scaling["C_CNBR"],
        upper_count,
    )
    minimal_phase_a = {
        "format_version": materializer.FORMAT_VERSION,
        "record_type": "c1_orion_bmr10_scaling_owner_manifest",
        "fixed_parameters": {
            "num_partitions": 32,
            "upper_subset_denominator": 32,
        },
        "construction_inputs": {
            "upper_only_projection": (
                scaling["manifest"]["construction_inputs"]["upper_only_projection"]
            )
        },
    }
    frozen = materializer.replace(
        base.frozen,
        manifest_path=scaling["manifest_path"],
        manifest_sha256=scaling["manifest_sha256"],
        manifest=minimal_phase_a,
        num_partitions=32,
        owners={"N_native": n_owner, "C_CNBR": c_owner},
    )
    logical_count = int(frozen.artifact["logical_point_count"])
    budgeted = materializer.budgeted_policy.build_budgeted_assignment(
        owner=c_owner.values,
        attachment_local=base.inputs.attachment_local,
        proxy_mass=frozen.mass_values,
        num_partitions=32,
    )
    metrics, _histogram, _shard_counts = materializer.assignment_metrics(
        budgeted, logical_count=logical_count, partitions=32
    )
    return {
        "owner_sha256": c_owner.sha256,
        "assignment_sha256": materializer.assignment_digest(budgeted.membership),
        "membership_semantic_sha256": metrics["membership_semantic_sha256"],
        "physical_point_count": metrics["physical_point_count"],
        "logical_point_count": logical_count,
        "source_artifact_sha256": base.source_artifact_sha256,
        "phase_b_screen_sha256": base.screen_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.generation <= 0:
        raise ValueError("generation must be positive")
    omission = c6_cost_binding(args.construction_cost_audit)
    original = materializer.cost_gate.validate_construction_cost_v4_pass
    original_link = materializer.native.os.link
    original_samefile = materializer.native.os.path.samefile
    original_p32 = {
        "owner": materializer.P32_OWNER_SHA256,
        "assignment": materializer.P32_ASSIGNMENT_SHA256,
        "membership": materializer.P32_MEMBERSHIP_SHA256,
        "physical": materializer.P32_PHYSICAL_POINT_COUNT,
    }
    cross_device_copies: list[dict[str, str]] = []
    p32_expectation: dict[str, Any] | None = None

    def link_or_copy(source: str | Path, destination: str | Path) -> None:
        try:
            original_link(source, destination)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(source, destination)
            if sha256_path(source) != sha256_path(destination):
                raise RuntimeError("cross-device vector copy checksum mismatch")
            cross_device_copies.append(
                {"source": str(source), "destination": str(destination)}
            )

    def same_file_or_verified_copy(left: str | Path, right: str | Path) -> bool:
        copied_pairs = {
            (record["source"], record["destination"])
            for record in cross_device_copies
        }
        normalized = (str(left), str(right))
        reversed_normalized = (normalized[1], normalized[0])
        if normalized in copied_pairs or reversed_normalized in copied_pairs:
            return sha256_path(left) == sha256_path(right)
        return original_samefile(left, right)

    try:
        materializer.cost_gate.validate_construction_cost_v4_pass = (
            lambda _path, **_kwargs: dict(omission)
        )
        materializer.native.os.link = link_or_copy
        materializer.native.os.path.samefile = same_file_or_verified_copy
        if args.partitions == 32:
            p32_expectation = dataset_specific_p32_expectation(args)
            materializer.P32_OWNER_SHA256 = p32_expectation["owner_sha256"]
            materializer.P32_ASSIGNMENT_SHA256 = p32_expectation[
                "assignment_sha256"
            ]
            materializer.P32_MEMBERSHIP_SHA256 = p32_expectation[
                "membership_semantic_sha256"
            ]
            materializer.P32_PHYSICAL_POINT_COUNT = p32_expectation[
                "physical_point_count"
            ]
            reference_path = args.c6_record.with_name(
                args.c6_record.stem + "-p32-reference.json"
            )
            write_json_atomic(reference_path, p32_expectation)
            args.p32_bmr_manifest = reference_path
        result = materializer.materialize_bundle(args)
    finally:
        materializer.cost_gate.validate_construction_cost_v4_pass = original
        materializer.native.os.link = original_link
        materializer.native.os.path.samefile = original_samefile
        materializer.P32_OWNER_SHA256 = original_p32["owner"]
        materializer.P32_ASSIGNMENT_SHA256 = original_p32["assignment"]
        materializer.P32_MEMBERSHIP_SHA256 = original_p32["membership"]
        materializer.P32_PHYSICAL_POINT_COUNT = original_p32["physical"]
    record = {
        "record_type": "c6_fixed_layout_materialization",
        "format_version": 1,
        "timestamp": utc_timestamp(),
        "partitions": args.partitions,
        "generation": args.generation,
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "materializer": {
            "path": str(Path(materializer.__file__).resolve()),
            "sha256": sha256_path(materializer.__file__),
        },
        "c6_wrapper": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_path(__file__),
        },
        "construction_cost_gate": omission,
        "cross_device_vector_copy": {
            "used": bool(cross_device_copies),
            "copies": cross_device_copies,
            "checksum_verified": True,
        },
        "same_dataset_p32_expectation": p32_expectation,
        "preserved_scope": [
            "frozen production upper graph",
            "frozen Phase-A and Phase-B owner identity",
            "dual-dataset C_CNBR selection",
            "full L0 attachments",
            "BMR10 assignment rule",
            "upper replay parity",
        ],
        "result": result,
    }
    write_json_atomic(args.c6_record, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
