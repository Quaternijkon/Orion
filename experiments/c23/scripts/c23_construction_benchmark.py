#!/usr/bin/env python3
"""Collect protocol-required C23 partition-construction overheads.

Random and K-Means primary artifacts predate construction timing metadata, so
this script replays those deterministic builds with the accepted parameters in
fresh child processes.  Full Orion and Orion-NoRefinement already carry
instrumented construction timings in their accepted partition artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c1.scripts import c1_protocol
from experiments.c23.scripts import c23_e1, c23_protocol


LOGICAL_SHARDS = (2, 4, 8, 16, 32)
BASELINE_METHODS = ("random", "kmeans")
ORION_METHODS = ("orion", "orion_no_refinement")


def write_json_new(path: str | Path, value: Any) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv_new(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_artifact(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as archive:
        assignments = np.asarray(archive["assignments"], dtype=np.int32)
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    return assignments, metadata


def peak_process_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def run_single(args: argparse.Namespace) -> int:
    destination = Path(args.single_output).expanduser().resolve()
    report_path = Path(args.single_report).expanduser().resolve()
    accepted = Path(args.accepted_artifact).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or report_path.exists():
        raise FileExistsError("single-run output already exists")

    spec = c1_protocol.DATASETS[args.dataset]
    started = time.perf_counter()
    record = c1_protocol.partition_artifact(
        args.hdf5_path,
        spec,
        args.single_method,
        args.single_shards,
        destination,
        seed=20260820,
        sample_size=100_000,
        iterations=12,
        tolerance=1e-4,
        batch_size=16_384,
    )
    elapsed = time.perf_counter() - started
    peak_rss = peak_process_rss_bytes()

    replay_assignments, replay_metadata = load_artifact(destination)
    accepted_assignments, accepted_metadata = load_artifact(accepted)
    assignments_equal = bool(np.array_equal(replay_assignments, accepted_assignments))
    accepted_sha256 = c23_e1.sha256_path(accepted)
    replay_sha256 = c23_e1.sha256_path(destination)
    report = {
        "dataset": args.dataset,
        "partition_method": args.single_method,
        "partition_seed": 20260820,
        "logical_shards": args.single_shards,
        "point_count": len(replay_assignments),
        "partition_construction_seconds": elapsed,
        "peak_construction_memory_bytes": peak_rss,
        "accepted_artifact_path": str(accepted),
        "accepted_artifact_sha256": accepted_sha256,
        "replay_artifact_path": str(destination),
        "replay_artifact_sha256": replay_sha256,
        "record_artifact_sha256": str(record["artifact_sha256"]),
        "assignments_equal_to_accepted": assignments_equal,
        "metadata_equal_to_accepted_ignoring_timestamp": {
            key: value
            for key, value in replay_metadata.items()
            if key != "timestamp"
        }
        == {
            key: value
            for key, value in accepted_metadata.items()
            if key != "timestamp"
        },
        "shard_counts": np.bincount(
            replay_assignments.astype(np.int64), minlength=args.single_shards
        ).tolist(),
    }
    write_json_new(report_path, report)
    return 0


def orion_rows(dataset: str, manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest = json.loads(
        Path(manifest_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    for record in manifest["artifacts"]:
        method = str(record["method"])
        logical_shards = int(record["logical_shards"])
        if method not in ORION_METHODS or logical_shards not in LOGICAL_SHARDS:
            continue
        artifact_path = Path(record["artifact_path"]).expanduser().resolve()
        assignments, metadata = load_artifact(artifact_path)
        timings = dict(metadata["construction_timings"])
        navigation_seconds = float(metadata["navigation_graph_construction_seconds"])
        attachment_seconds = float(metadata["attachment_search_seconds"])
        initial_label_seconds = float(timings["initial_label_seconds"])
        refinement_seconds = float(timings["topology_refinement_seconds"])
        assignment_seconds = float(timings["unsampled_assignment_seconds"])
        total_seconds = (
            navigation_seconds
            + attachment_seconds
            + initial_label_seconds
            + refinement_seconds
            + assignment_seconds
        )
        counts = np.bincount(assignments.astype(np.int64), minlength=logical_shards)
        rows.append(
            {
                "dataset": dataset,
                "partition_method": method,
                "partition_seed": int(metadata["partition_seed"]),
                "logical_shards": logical_shards,
                "point_count": len(assignments),
                "measurement_scope": "accepted_artifact_instrumentation",
                "partition_construction_seconds": total_seconds,
                "peak_construction_memory_bytes": int(
                    metadata["peak_construction_memory_bytes"]
                ),
                "navigation_graph_construction_seconds": navigation_seconds,
                "navigation_graph_memory_bytes": int(
                    metadata["navigation_graph_memory_bytes"]
                ),
                "attachment_search_seconds": attachment_seconds,
                "initial_label_seconds": initial_label_seconds,
                "topology_refinement_seconds": refinement_seconds,
                "topology_refinement_iterations": int(
                    timings["topology_refinement_iterations"]
                ),
                "unsampled_assignment_seconds": assignment_seconds,
                "final_shard_vector_counts_json": json.dumps(
                    counts.tolist(), separators=(",", ":")
                ),
                "artifact_path": str(artifact_path),
                "artifact_sha256": c23_e1.sha256_path(artifact_path),
                "replay_artifact_path": "",
                "replay_artifact_sha256": "",
                "assignments_equal_to_accepted": True,
                "notes": (
                    "Standalone total includes the shared navigation-graph and "
                    "attachment-search stages."
                ),
            }
        )
    return rows


def baseline_row(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": report["dataset"],
        "partition_method": report["partition_method"],
        "partition_seed": report["partition_seed"],
        "logical_shards": report["logical_shards"],
        "point_count": report["point_count"],
        "measurement_scope": "deterministic_same_parameter_replay",
        "partition_construction_seconds": report["partition_construction_seconds"],
        "peak_construction_memory_bytes": report["peak_construction_memory_bytes"],
        "navigation_graph_construction_seconds": 0.0,
        "navigation_graph_memory_bytes": 0,
        "attachment_search_seconds": 0.0,
        "initial_label_seconds": 0.0,
        "topology_refinement_seconds": 0.0,
        "topology_refinement_iterations": 0,
        "unsampled_assignment_seconds": 0.0,
        "final_shard_vector_counts_json": json.dumps(
            report["shard_counts"], separators=(",", ":")
        ),
        "artifact_path": report["accepted_artifact_path"],
        "artifact_sha256": report["accepted_artifact_sha256"],
        "replay_artifact_path": report["replay_artifact_path"],
        "replay_artifact_sha256": report["replay_artifact_sha256"],
        "assignments_equal_to_accepted": bool(
            report["assignments_equal_to_accepted"]
        ),
        "assignment_disagreement_count": int(
            report["assignment_disagreement_count"]
        ),
        "assignment_disagreement_fraction": float(
            report["assignment_disagreement_fraction"]
        ),
        "parameter_contract_equal_to_accepted": bool(
            report["parameter_contract_equal_to_accepted"]
        ),
        "notes": (
            "Replay uses the accepted seed, 100k K-Means sample, 12 iterations, "
            "1e-4 tolerance, and batch size 16384. Small high-M assignment "
            "differences are recorded as floating-point reduction drift; accepted "
            "E1-E4 evidence continues to use the primary artifact."
        ),
    }


def enrich_replay_report(report: dict[str, Any]) -> dict[str, Any]:
    accepted_assignments, accepted_metadata = load_artifact(
        report["accepted_artifact_path"]
    )
    replay_assignments, replay_metadata = load_artifact(report["replay_artifact_path"])
    disagreement_count = int(
        np.count_nonzero(accepted_assignments != replay_assignments)
    )
    disagreement_fraction = disagreement_count / len(accepted_assignments)

    def parameter_contract(metadata: dict[str, Any]) -> dict[str, Any]:
        kmeans = metadata.get("kmeans")
        return {
            "dataset_sha256": metadata["dataset_sha256"],
            "dimension": int(metadata["dimension"]),
            "logical_shards": int(metadata["logical_shards"]),
            "method": metadata["method"],
            "partition_seed": int(metadata["partition_seed"]),
            "point_count": int(metadata["point_count"]),
            "kmeans_parameters": None
            if kmeans is None
            else {
                "batch_size": int(kmeans["batch_size"]),
                "maximum_iterations": int(kmeans["maximum_iterations"]),
                "sample_size": int(kmeans["sample_size"]),
                "tolerance": float(kmeans["tolerance"]),
            },
        }

    return {
        **report,
        "assignment_disagreement_count": disagreement_count,
        "assignment_disagreement_fraction": disagreement_fraction,
        "parameter_contract_equal_to_accepted": (
            parameter_contract(accepted_metadata) == parameter_contract(replay_metadata)
        ),
    }


def run_parent(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and not args.reuse_completed_replays:
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=args.reuse_completed_replays)

    dataset_inputs = {
        "sift1m": {
            "hdf5": args.sift_hdf5,
            "orion_manifest": args.sift_orion_manifest,
        },
        "glove-200-angular": {
            "hdf5": args.glove_hdf5,
            "orion_manifest": args.glove_orion_manifest,
        },
    }
    reports: list[dict[str, Any]] = []
    step = 0
    total = len(dataset_inputs) * len(BASELINE_METHODS) * len(LOGICAL_SHARDS)
    for dataset, inputs in dataset_inputs.items():
        for method in BASELINE_METHODS:
            for logical_shards in LOGICAL_SHARDS:
                step += 1
                dataset_dir = output / dataset
                artifact = dataset_dir / f"{method}-m{logical_shards}-seed20260820.npz"
                report = dataset_dir / f"{method}-m{logical_shards}-seed20260820.json"
                accepted = (
                    Path(args.primary_baseline_root).expanduser().resolve()
                    / dataset
                    / f"{method}-m{logical_shards}.npz"
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--single",
                    "--dataset",
                    dataset,
                    "--hdf5-path",
                    str(inputs["hdf5"]),
                    "--single-method",
                    method,
                    "--single-shards",
                    str(logical_shards),
                    "--single-output",
                    str(artifact),
                    "--single-report",
                    str(report),
                    "--accepted-artifact",
                    str(accepted),
                ]
                if args.reuse_completed_replays and artifact.is_file() and report.is_file():
                    replay_report = json.loads(report.read_text(encoding="utf-8"))
                else:
                    subprocess.run(command, cwd=REPO_ROOT, check=True)
                    replay_report = json.loads(report.read_text(encoding="utf-8"))
                reports.append(enrich_replay_report(replay_report))
                print(
                    f"[{step}/{total}] replayed {dataset} {method} M={logical_shards}",
                    flush=True,
                )

    rows = [baseline_row(report) for report in reports]
    for dataset, inputs in dataset_inputs.items():
        rows.extend(orion_rows(dataset, inputs["orion_manifest"]))
    rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            c23_e1.METHOD_ORDER.index(str(row["partition_method"])),
            int(row["logical_shards"]),
        )
    )

    expected_keys = {
        (dataset, method, logical_shards)
        for dataset in dataset_inputs
        for method in c23_e1.METHOD_ORDER
        for logical_shards in LOGICAL_SHARDS
    }
    actual_keys = {
        (str(row["dataset"]), str(row["partition_method"]), int(row["logical_shards"]))
        for row in rows
    }
    checks = {
        "complete_40_configuration_matrix": "PASS"
        if actual_keys == expected_keys and len(rows) == 40
        else "FAIL",
        "baseline_replay_parameter_contract_matches_accepted": "PASS"
        if all(
            bool(report["parameter_contract_equal_to_accepted"])
            for report in reports
        )
        else "FAIL",
        "baseline_replay_assignment_agreement_at_least_99_98pct": "PASS"
        if all(
            float(report["assignment_disagreement_fraction"]) <= 0.0002
            for report in reports
        )
        else "FAIL",
        "positive_construction_time_and_peak_memory": "PASS"
        if all(
            float(row["partition_construction_seconds"]) > 0
            and int(row["peak_construction_memory_bytes"]) > 0
            for row in rows
        )
        else "FAIL",
        "orion_stage_timings_present": "PASS"
        if all(
            float(row["attachment_search_seconds"]) > 0
            and float(row["unsampled_assignment_seconds"]) > 0
            for row in rows
            if row["partition_method"] in ORION_METHODS
        )
        else "FAIL",
    }
    failed = [name for name, status in checks.items() if status != "PASS"]
    if failed:
        raise ValueError(f"construction benchmark audit failed: {failed}")

    csv_path = output / "construction_overhead.csv"
    write_csv_new(csv_path, rows)
    manifest = {
        "protocol_version": c23_protocol.PROTOCOL_VERSION,
        "timestamp": c23_e1.utc_timestamp(),
        "git_commit": c23_protocol.git_commit(REPO_ROOT),
        "experiment": "C23 partition construction overheads",
        "configuration_count": len(rows),
        "baseline_replay_configuration_count": len(reports),
        "checks": checks,
        "measurement_note": (
            "Random and K-Means times are deterministic same-parameter replays "
            "because the accepted primary artifacts did not persist timings. "
            "High-M K-Means replays may differ at a tiny fraction of assignments "
            "because of floating-point reduction order; exact disagreement rates "
            "are retained and replay artifacts are not used for E1-E4 results."
        ),
        "files": {
            "construction_overhead": str(csv_path),
            "construction_overhead_sha256": c23_e1.sha256_path(csv_path),
        },
    }
    manifest_path = output / "manifest.json"
    write_json_new(manifest_path, manifest)
    audit = {
        **manifest,
        "manifest": str(manifest_path),
        "manifest_sha256": c23_e1.sha256_path(manifest_path),
        "rows": rows,
    }
    write_json_new(args.audit_output, audit)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--dataset", choices=("sift1m", "glove-200-angular"))
    parser.add_argument("--hdf5-path")
    parser.add_argument("--single-method", choices=BASELINE_METHODS)
    parser.add_argument("--single-shards", type=int, choices=LOGICAL_SHARDS)
    parser.add_argument("--single-output")
    parser.add_argument("--single-report")
    parser.add_argument("--accepted-artifact")
    parser.add_argument("--sift-hdf5")
    parser.add_argument("--glove-hdf5")
    parser.add_argument("--primary-baseline-root")
    parser.add_argument("--sift-orion-manifest")
    parser.add_argument("--glove-orion-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--audit-output")
    parser.add_argument("--reuse-completed-replays", action="store_true")
    args = parser.parse_args()

    if args.single:
        required = (
            "dataset",
            "hdf5_path",
            "single_method",
            "single_shards",
            "single_output",
            "single_report",
            "accepted_artifact",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"single mode missing arguments: {missing}")
        return run_single(args)

    required = (
        "sift_hdf5",
        "glove_hdf5",
        "primary_baseline_root",
        "sift_orion_manifest",
        "glove_orion_manifest",
        "output_dir",
        "audit_output",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"parent mode missing arguments: {missing}")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
