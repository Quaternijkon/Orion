#!/usr/bin/env python3
"""Run the resumable sequential C23 local HNSW matrix with one shard active at a time."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c23.scripts import c23_e1, c23_protocol


def validate_completed_run(path: Path, configuration: dict, expected: dict) -> dict:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"incomplete output directory blocks resume: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "dataset": expected["dataset"],
        "dataset_sha256": expected["dataset_sha256"],
        "partition_method": configuration["partition_method"],
        "partition_seed": int(configuration["partition_seed"]),
        "partition_sha256": configuration["raw_assignment_sha256"],
        "logical_shards": int(configuration["logical_shards"]),
        "point_count": int(configuration["point_count"]),
        "dimension": int(expected["dimension"]),
        "hnsw_m": int(expected["hnsw_m"]),
        "hnsw_ef_construction": int(expected["hnsw_ef_construction"]),
        "hnsw_full_scan_threshold_kb": int(expected["full_scan_threshold_kb"]),
        "hnsw_graph_seed": int(expected["graph_seed"]),
        "measurement_start": int(expected["measurement_start"]),
        "measurement_query_count": int(expected["measurement_count"]),
        "e3_ef_values": list(expected["e3_ef_values"]),
        "e4_ef_search": int(expected["e4_ef"]),
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in checks.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"completed-run manifest mismatch for {path}: {mismatches}")
    if len(manifest.get("shards", [])) != int(configuration["logical_shards"]):
        raise ValueError(f"completed-run shard coverage mismatch for {path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-suffix", default="v2")
    parser.add_argument("--measurement-start", type=int, default=1000)
    parser.add_argument("--measurement-count", type=int, default=9000)
    parser.add_argument("--e3-ef-values", default="10,20,40,80,160,320")
    parser.add_argument("--e4-ef", type=int, required=True)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--full-scan-threshold-kb", type=int, default=10)
    parser.add_argument("--graph-seed", type=int, default=20260821)
    parser.add_argument("--registry-output", required=True)
    args = parser.parse_args()

    binary = Path(args.binary).expanduser().resolve()
    input_manifest_path = Path(args.input_manifest).expanduser().resolve()
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    e3_ef_values = [int(value) for value in args.e3_ef_values.split(",")]
    expected = {
        "dataset": input_manifest["dataset"],
        "dataset_sha256": input_manifest["dataset_sha256"],
        "dimension": int(input_manifest["dimension"]),
        "hnsw_m": args.hnsw_m,
        "hnsw_ef_construction": args.ef_construction,
        "full_scan_threshold_kb": args.full_scan_threshold_kb,
        "graph_seed": args.graph_seed,
        "measurement_start": args.measurement_start,
        "measurement_count": args.measurement_count,
        "e3_ef_values": e3_ef_values,
        "e4_ef": args.e4_ef,
    }
    completed = []
    for index, configuration in enumerate(input_manifest["configurations"], 1):
        configuration_id = str(configuration["configuration_id"])
        output = output_root / f"{configuration_id}-{args.run_suffix}"
        if output.exists():
            manifest = validate_completed_run(output, configuration, expected)
            print(f"[{index}/20] accepted existing {configuration_id}", flush=True)
        else:
            command = [
                str(binary),
                "--vectors",
                str(Path(args.vectors).expanduser().resolve()),
                "--queries",
                str(Path(args.queries).expanduser().resolve()),
                "--ground-truth",
                str(Path(args.ground_truth).expanduser().resolve()),
                "--assignments",
                str(Path(configuration["raw_assignment_path"])),
                "--output-dir",
                str(output),
                "--dataset",
                str(input_manifest["dataset"]),
                "--dataset-sha256",
                str(input_manifest["dataset_sha256"]),
                "--partition-method",
                str(configuration["partition_method"]),
                "--partition-seed",
                str(configuration["partition_seed"]),
                "--partition-sha256",
                str(configuration["raw_assignment_sha256"]),
                "--dimension",
                str(input_manifest["dimension"]),
                "--distance",
                str(input_manifest["distance"]),
                "--logical-shards",
                str(configuration["logical_shards"]),
                "--top",
                "10",
                "--ground-truth-width",
                "10",
                "--measurement-start",
                str(args.measurement_start),
                "--measurement-count",
                str(args.measurement_count),
                "--e3-ef-values",
                args.e3_ef_values,
                "--e4-ef",
                str(args.e4_ef),
                "--m",
                str(args.hnsw_m),
                "--ef-construction",
                str(args.ef_construction),
                "--full-scan-threshold-kb",
                str(args.full_scan_threshold_kb),
                "--graph-seed",
                str(args.graph_seed),
            ]
            print(f"[{index}/20] starting {configuration_id}", flush=True)
            subprocess.run(command, check=True)
            manifest = validate_completed_run(output, configuration, expected)
            print(f"[{index}/20] completed {configuration_id}", flush=True)
        completed.append(
            {
                "configuration_id": configuration_id,
                "run_dir": str(output),
                "manifest": str(output / "manifest.json"),
                "manifest_sha256": c23_e1.sha256_path(output / "manifest.json"),
                "total_seconds": float(manifest["total_seconds"]),
            }
        )
    registry = {
        "protocol_version": c23_protocol.PROTOCOL_VERSION,
        "timestamp": c23_e1.utc_timestamp(),
        "git_commit": c23_protocol.git_commit(REPO_ROOT),
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": c23_e1.sha256_path(input_manifest_path),
        "configuration_count": len(completed),
        "settings": expected,
        "runs": completed,
        "checks": {"complete_sequential_local_matrix": "PASS"},
    }
    c23_e1.write_json_new(args.registry_output, registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
