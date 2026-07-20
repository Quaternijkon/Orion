#!/usr/bin/env python3
"""Build a static native Simple KMeans layout using the existing baseline code."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import orion_native_layout as common  # noqa: E402
from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402


GRAPHLESS_NAME = "simple-kmeans-graphless.json"
BUILD_MANIFEST_NAME = common.BUILD_MANIFEST_NAME
CHECKSUMS_NAME = common.CHECKSUMS_NAME


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a native Simple KMeans numeric-shard layout by reusing "
            "build_cpp_kmeans_baseline_assignments."
        )
    )
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory. Existing paths are never overwritten.",
    )
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--p", "--num-shards", dest="num_shards", type=int, default=46)
    parser.add_argument(
        "--vector-distance",
        choices=("cosine", "euclid", "l2"),
        default="cosine",
    )
    parser.add_argument("--vector-name", default="")
    parser.add_argument("--nprobe", type=int, default=1)
    parser.add_argument("--lower-hnsw-ef", type=int, default=48)
    parser.add_argument("--kmeans-train-size", type=int, default=10_000)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--kmeans-seed", type=int, default=1)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument(
        "--cargo-target-dir",
        default=None,
        help="Optional external target directory passed as CARGO_TARGET_DIR.",
    )
    parser.add_argument("--bundle-prefix", default="simple_kmeans_numeric_import")
    parser.add_argument("--bundle-row-chunk-size", type=int, default=16_384)
    parser.add_argument(
        "--graphless-only",
        action="store_true",
        help="Stop before invoking the Rust artifact canonicalizer.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "generation": args.generation,
        "num_shards": args.num_shards,
        "nprobe": args.nprobe,
        "lower_hnsw_ef": args.lower_hnsw_ef,
        "kmeans_train_size": args.kmeans_train_size,
        "kmeans_iters": args.kmeans_iters,
        "bundle_row_chunk_size": args.bundle_row_chunk_size,
    }
    for name, value in positive.items():
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.train_limit is not None and int(args.train_limit) <= 0:
        raise ValueError("--train-limit must be positive")
    if int(args.nprobe) > int(args.num_shards):
        raise ValueError("--nprobe must not exceed --p")
    if not args.bundle_prefix or Path(args.bundle_prefix).name != args.bundle_prefix:
        raise ValueError("--bundle-prefix must be a non-empty file-name component")


def rust_builder_command(
    args: argparse.Namespace,
    graphless_path: Path,
    production_path: Path,
) -> list[str]:
    return [
        str(args.cargo),
        "run",
        "--release",
        "-p",
        "collection",
        "--example",
        "simple_kmeans_build_artifact",
        "--",
        str(graphless_path),
        str(production_path),
    ]


def run_rust_builder(
    args: argparse.Namespace,
    graphless_path: Path,
    production_path: Path,
) -> list[str]:
    command = rust_builder_command(args, graphless_path, production_path)
    environment = os.environ.copy()
    cargo_target_dir = common.effective_cargo_target_dir(args)
    if cargo_target_dir:
        environment["CARGO_TARGET_DIR"] = cargo_target_dir
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=environment)
    return command


def read_artifact_binding(
    artifact_path: Path,
    args: argparse.Namespace,
    logical_point_count: int,
) -> dict[str, Any]:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Simple KMeans artifact root must be a JSON object")
    expected = {
        "format_version": 1,
        "generation": int(args.generation),
        "shard_count": int(args.num_shards),
        "logical_point_count": int(logical_point_count),
        "physical_point_count": int(logical_point_count),
        "routing_distance": "squared_l2",
        "nprobe": int(args.nprobe),
        "lower_hnsw_ef": int(args.lower_hnsw_ef),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(
                f"Simple KMeans artifact {field} mismatch: "
                f"expected={value!r}, actual={payload.get(field)!r}"
            )
    layout_sha256 = str(payload.get("layout_sha256") or "").lower()
    if len(layout_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in layout_sha256
    ):
        raise RuntimeError("Simple KMeans artifact layout_sha256 is invalid")
    centroids = payload.get("centroids")
    if not isinstance(centroids, list) or len(centroids) != int(args.num_shards):
        raise RuntimeError("Simple KMeans artifact must contain one centroid per shard")
    if "upper_graph" in payload:
        raise RuntimeError("Simple KMeans artifact must not contain upper_graph")
    return {
        **expected,
        "layout_sha256": layout_sha256,
    }


def verify_production_artifact(
    production_path: Path,
    args: argparse.Namespace,
    logical_point_count: int,
) -> tuple[str, dict[str, Any]]:
    checksum_path = Path(f"{production_path}.sha256")
    if not production_path.is_file() or not checksum_path.is_file():
        raise RuntimeError("Rust canonicalizer did not create artifact and checksum files")
    expected_sha256 = checksum_path.read_text(encoding="utf-8").strip().lower()
    actual_sha256 = common.sha256_path(production_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Simple KMeans production artifact checksum mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    return actual_sha256, read_artifact_binding(
        production_path,
        args,
        logical_point_count,
    )


def parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generation": int(args.generation),
        "num_shards": int(args.num_shards),
        "vector_distance": str(args.vector_distance),
        "vector_name": str(args.vector_name),
        "routing_distance": "squared_l2",
        "nprobe": int(args.nprobe),
        "lower_hnsw_ef": int(args.lower_hnsw_ef),
        "kmeans_train_size": int(args.kmeans_train_size),
        "kmeans_iters": int(args.kmeans_iters),
        "kmeans_seed": int(args.kmeans_seed),
        "cargo_target_dir": common.effective_cargo_target_dir(args),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    hdf5_path = Path(args.hdf5_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite existing output path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    train, dataset_record = common.load_train_vectors(
        hdf5_path,
        args.train_limit,
        args.vector_distance,
    )
    if len(train) < int(args.num_shards):
        raise ValueError(
            f"selected train rows {len(train)} are fewer than P={args.num_shards}"
        )
    assignments, centroids = experiment.build_cpp_kmeans_baseline_assignments(
        train,
        int(args.num_shards),
        int(args.kmeans_train_size),
        int(args.kmeans_iters),
        int(args.kmeans_seed),
    )
    point_to_shards = [[int(shard_id)] for shard_id in assignments.tolist()]
    shard_counts = experiment.np.bincount(
        assignments.astype(experiment.np.int64, copy=False),
        minlength=int(args.num_shards),
    )
    distance_config = experiment.vector_distance_config(args.vector_distance)
    graphless_path = output_dir / GRAPHLESS_NAME
    experiment.write_simple_kmeans_graphless_artifact(
        train,
        centroids,
        point_to_shards,
        int(args.num_shards),
        graphless_path,
        generation=int(args.generation),
        vector_distance=distance_config["name"],
        nprobe=int(args.nprobe),
        lower_hnsw_ef=int(args.lower_hnsw_ef),
        vector_name=str(args.vector_name),
    )
    artifact_binding = read_artifact_binding(
        graphless_path,
        args,
        logical_point_count=len(train),
    )

    rust_command: list[str] | None = None
    production_path: Path | None = None
    production_sha256: str | None = None
    import_manifest_path: Path | None = None
    if not args.graphless_only:
        production_path = output_dir / f"generation-{int(args.generation)}.json"
        rust_command = run_rust_builder(args, graphless_path, production_path)
        production_sha256, production_binding = verify_production_artifact(
            production_path,
            args,
            logical_point_count=len(train),
        )
        if production_binding != artifact_binding:
            raise RuntimeError("Rust canonicalizer changed the Simple KMeans artifact binding")
        import_manifest_path = experiment.write_numeric_shard_import_bundle_v2(
            train,
            point_to_shards,
            int(args.num_shards),
            output_dir,
            routing_policy="simple_kmeans",
            routing_generation=int(args.generation),
            routing_artifact_path=production_path,
            vector_name=str(args.vector_name),
            prefix=str(args.bundle_prefix),
            row_chunk_size=int(args.bundle_row_chunk_size),
        )

    payload_files = common.relative_file_records(
        output_dir,
        {BUILD_MANIFEST_NAME, CHECKSUMS_NAME},
    )
    manifest = {
        "format_version": 1,
        "tool": "tools/simple_kmeans_native_layout.py",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "graphless_only" if args.graphless_only else "production_bundle",
        "dataset": dataset_record,
        "parameters": parameters(args),
        "artifact_binding": artifact_binding,
        "routing": {
            "policy": "simple_kmeans",
            "logical_point_count": int(len(train)),
            "physical_point_count": int(len(train)),
            "expansion_ratio": 1.0,
            "shard_counts": [int(value) for value in shard_counts.tolist()],
        },
        "outputs": {
            "graphless_artifact": graphless_path.name,
            "production_artifact": production_path.name if production_path else None,
            "production_artifact_sha256": production_sha256,
            "import_manifest": import_manifest_path.name if import_manifest_path else None,
            "rust_builder_command": rust_command,
            "files": payload_files,
        },
    }
    build_manifest_path = output_dir / BUILD_MANIFEST_NAME
    common.write_json_new(build_manifest_path, manifest)
    checksums_path = common.write_checksums(output_dir)
    summary = {
        "output_dir": str(output_dir),
        "mode": manifest["mode"],
        "graphless_artifact": str(graphless_path),
        "production_artifact": str(production_path) if production_path else None,
        "production_artifact_sha256": production_sha256,
        "import_manifest": str(import_manifest_path) if import_manifest_path else None,
        "build_manifest": str(build_manifest_path),
        "checksums": str(checksums_path),
        "layout_sha256": artifact_binding["layout_sha256"],
        "logical_point_count": int(len(train)),
        "physical_point_count": int(len(train)),
        "shard_count": int(args.num_shards),
        "nprobe": int(args.nprobe),
        "lower_hnsw_ef": int(args.lower_hnsw_ef),
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        build(parse_args(argv))
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
