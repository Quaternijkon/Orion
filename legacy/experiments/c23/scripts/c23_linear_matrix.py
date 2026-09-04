#!/usr/bin/env python3
"""Run the resumable two-dataset C23 linear-resource E3/E4 matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import c23_linear_e3e4 as experiment
import c23_linear_resources as resources


DEFAULT_RAW_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/c23-linear-20260824-v1"
)
DEFAULT_CALIBRATIONS = {
    "sift1m": DEFAULT_RAW_ROOT / "calibration" / "sift1m-v2" / "manifest.json",
    "glove-200-angular": (
        DEFAULT_RAW_ROOT / "calibration" / "glove-200-angular-v1" / "manifest.json"
    ),
}


def load_calibration(path: Path, dataset: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload.get("checks") or {}
    if (
        payload.get("status") != "PASS"
        or payload.get("dataset") != dataset
        or int(payload.get("logical_shards") or 0) != 1
        or not isinstance(payload.get("selected_ef_search"), int)
        or any(value != "PASS" for value in checks.values())
        or checks.get("benchmark_client_affinity") != "PASS"
    ):
        raise ValueError(f"invalid calibration manifest: {path}")
    return payload


def parse_values(raw: str, *, allowed: tuple[Any, ...], conversion):
    values = tuple(conversion(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value not in allowed for value in values):
        raise ValueError(f"values must be a non-empty subset of {allowed}: {values}")
    return values


def configuration_id(dataset: str, method: str, logical_shards: int) -> str:
    dataset_token = "glove" if dataset == "glove-200-angular" else dataset
    method_token = method.replace("_", "-")
    return f"s13-{dataset_token}-{method_token}-m{logical_shards}"


def valid_completed(
    manifest_path: Path,
    *,
    dataset: str,
    method: str,
    logical_shards: int,
    ef_search: int,
    query_start: int,
    query_count: int,
) -> bool:
    if not manifest_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_method = "common_m1" if logical_shards == 1 else method
    required_phase_checks = {
        "resource_after_upload",
        "resource_after_index",
        "resource_after_e4",
        "resource_after_e3",
    }
    snapshots = payload.get("resource_phase_snapshots") or {}
    return (
        payload.get("status") == "PASS"
        and payload.get("dataset") == dataset
        and payload.get("partition_method") == expected_method
        and int(payload.get("logical_shards") or 0) == logical_shards
        and int(payload.get("ef_search") or 0) == ef_search
        and payload.get("query_range") == [query_start, query_start + query_count]
        and (payload.get("checks") or {}).get("benchmark_client_affinity") == "PASS"
        and (payload.get("checks") or {}).get("complete_e3_grid") == "PASS"
        and required_phase_checks.issubset(payload.get("checks") or {})
        and set(snapshots)
        == {"before_upload", "after_upload", "after_index", "after_e4", "after_e3"}
        and all(value == "PASS" for value in (payload.get("checks") or {}).values())
        and (payload.get("cleanup") or {}).get("restored", {}).get("status") == "PASS"
    )


def write_registry(path: Path, rows: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    resources.write_json_atomic(
        path,
        {
            "protocol_version": resources.PROTOCOL_VERSION,
            "timestamp": resources.utc_timestamp(),
            "record_type": "c23_linear_resource_matrix_registry",
            "settings": settings,
            "completed_count": sum(row["status"] == "PASS" for row in rows),
            "configuration_count": len(rows),
            "configurations": rows,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default=str(resources.DEFAULT_TOPOLOGY))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--datasets", default=",".join(experiment.DATASETS))
    parser.add_argument("--methods", default=",".join(experiment.METHODS))
    parser.add_argument("--shards", default="1,2,4,8,16,32")
    parser.add_argument("--sift-calibration", default=str(DEFAULT_CALIBRATIONS["sift1m"]))
    parser.add_argument(
        "--glove-calibration", default=str(DEFAULT_CALIBRATIONS["glove-200-angular"])
    )
    parser.add_argument("--query-start", type=int, default=1000)
    parser.add_argument("--query-count", type=int, default=9000)
    parser.add_argument("--query-concurrency", type=int, default=16)
    parser.add_argument("--request-workers", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--upload-chunk-size", type=int, default=8192)
    parser.add_argument("--upload-batch-size", type=int, default=256)
    parser.add_argument("--index-timeout", type=float, default=10_800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = parse_values(
        args.datasets,
        allowed=tuple(experiment.DATASETS),
        conversion=str,
    )
    methods = parse_values(args.methods, allowed=experiment.METHODS, conversion=str)
    shards = parse_values(
        args.shards,
        allowed=resources.ALLOWED_LOGICAL_SHARDS,
        conversion=int,
    )
    calibration_paths = {
        "sift1m": Path(args.sift_calibration).expanduser().resolve(),
        "glove-200-angular": Path(args.glove_calibration).expanduser().resolve(),
    }
    calibrations = {
        dataset: load_calibration(calibration_paths[dataset], dataset) for dataset in datasets
    }
    raw_root = Path(args.raw_root).expanduser().resolve()
    matrix_root = raw_root / "matrix"
    registry_path = matrix_root / "registry.json"
    settings = {
        "topology": str(Path(args.topology).expanduser().resolve()),
        "datasets": list(datasets),
        "methods": list(methods),
        "logical_shards": list(shards),
        "query_range": [args.query_start, args.query_start + args.query_count],
        "query_concurrency": args.query_concurrency,
        "request_workers": args.request_workers,
        "calibrations": {dataset: str(calibration_paths[dataset]) for dataset in datasets},
        "selected_ef": {
            dataset: calibrations[dataset]["selected_ef_search"] for dataset in datasets
        },
        "e3_ef_grid": list(experiment.EF_GRID),
    }
    planned: list[tuple[str, str, int]] = []
    for dataset in datasets:
        for logical_shards in shards:
            if logical_shards == 1:
                planned.append((dataset, "kmeans", 1))
            else:
                planned.extend((dataset, method, logical_shards) for method in methods)
    rows: list[dict[str, Any]] = []
    total = len(planned)
    for index, (dataset, method, logical_shards) in enumerate(planned, 1):
        config_id = configuration_id(dataset, "common" if logical_shards == 1 else method, logical_shards)
        output = matrix_root / dataset / config_id
        manifest_path = output / "manifest.json"
        ef_search = int(calibrations[dataset]["selected_ef_search"])
        if valid_completed(
            manifest_path,
            dataset=dataset,
            method=method,
            logical_shards=logical_shards,
            ef_search=ef_search,
            query_start=args.query_start,
            query_count=args.query_count,
        ):
            print(f"[{index}/{total}] resume PASS {config_id}", flush=True)
            rows.append(
                {
                    "configuration_id": config_id,
                    "dataset": dataset,
                    "partition_method": "common_m1" if logical_shards == 1 else method,
                    "logical_shards": logical_shards,
                    "ef_search": ef_search,
                    "manifest": str(manifest_path),
                    "status": "PASS",
                    "resumed": True,
                }
            )
            write_registry(registry_path, rows, settings)
            continue
        if output.exists():
            raise FileExistsError(f"incomplete output blocks resume: {output}")
        command = [
            "taskset",
            "-c",
            "8-15,24-31",
            sys.executable,
            str(SCRIPT_DIR / "c23_linear_e3e4.py"),
            "run-config",
            "--topology",
            str(Path(args.topology).expanduser().resolve()),
            "--run-id",
            config_id,
            "--dataset",
            dataset,
            "--method",
            method,
            "--logical-shards",
            str(logical_shards),
            "--ef-search",
            str(ef_search),
            "--e3-ef-grid",
            ",".join(map(str, experiment.EF_GRID)),
            "--query-start",
            str(args.query_start),
            "--query-count",
            str(args.query_count),
            "--query-concurrency",
            str(args.query_concurrency),
            "--request-workers",
            str(args.request_workers),
            "--progress-every",
            str(args.progress_every),
            "--upload-chunk-size",
            str(args.upload_chunk_size),
            "--upload-batch-size",
            str(args.upload_batch_size),
            "--index-timeout",
            str(args.index_timeout),
            "--output",
            str(output),
        ]
        print(f"[{index}/{total}] start {config_id}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            rows.append(
                {
                    "configuration_id": config_id,
                    "dataset": dataset,
                    "partition_method": "common_m1" if logical_shards == 1 else method,
                    "logical_shards": logical_shards,
                    "ef_search": ef_search,
                    "manifest": str(manifest_path),
                    "status": "PLANNED",
                    "resumed": False,
                }
            )
            continue
        subprocess.run(command, check=True)
        if not valid_completed(
            manifest_path,
            dataset=dataset,
            method=method,
            logical_shards=logical_shards,
            ef_search=ef_search,
            query_start=args.query_start,
            query_count=args.query_count,
        ):
            raise RuntimeError(f"configuration failed final validation: {manifest_path}")
        rows.append(
            {
                "configuration_id": config_id,
                "dataset": dataset,
                "partition_method": "common_m1" if logical_shards == 1 else method,
                "logical_shards": logical_shards,
                "ef_search": ef_search,
                "manifest": str(manifest_path),
                "status": "PASS",
                "resumed": False,
            }
        )
        write_registry(registry_path, rows, settings)
    write_registry(registry_path, rows, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
