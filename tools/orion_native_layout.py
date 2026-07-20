#!/usr/bin/env python3
"""Build a native Orion routing layout without duplicating the Orion algorithm.

This entry point is deliberately a thin orchestration layer. Dataset preparation,
upper sampling/indexing, L0-to-L1 attachment, topology convergence, fission,
multi-assignment, artifact serialization, and import-bundle serialization are all
delegated to ``qdrant_two_level_routing_experiment``.
"""

from __future__ import annotations

import argparse
import hashlib
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

from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402


GRAPHLESS_NAME = "graphless-orion.json"
BUILD_MANIFEST_NAME = "build-manifest.json"
CHECKSUMS_NAME = "checksums.sha256"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a static native Orion numeric-shard layout by reusing the existing "
            "Method4/Orion harness implementation."
        )
    )
    parser.add_argument("--hdf5-path", required=True, help="ANN-benchmark HDF5 dataset.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory. Existing paths are never overwritten.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Use the first N train rows for a smoke build; omit for the full dataset.",
    )
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--p", "--num-shards", dest="num_shards", type=int, default=31)
    parser.add_argument(
        "--vector-distance",
        choices=("cosine", "euclid", "l2"),
        default="cosine",
    )
    parser.add_argument("--vector-name", default="")

    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=100)
    parser.add_argument("--upper-k", type=int, default=100)
    parser.add_argument("--k-overlap", type=int, default=10)
    parser.add_argument("--upper-build-batch-size", type=int, default=10_000)

    parser.add_argument("--dynamic-ef-base", type=int, default=20)
    parser.add_argument("--dynamic-ef-factor", type=int, default=4)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--kmeans-seed", type=int, default=1)
    parser.add_argument("--topology-iters", type=int, default=50)
    parser.add_argument("--disable-multi-assign", action="store_true")
    parser.add_argument("--multi-assign-min-max-vote", type=int, default=2)
    parser.add_argument("--multi-assign-vote-delta", type=int, default=0)
    parser.add_argument("--multi-assign-max-shards", type=int, default=0)
    parser.add_argument("--disable-fission", action="store_true")

    parser.add_argument(
        "--upper-graph-seed",
        type=int,
        default=100,
        help="Deterministic seed passed to the Rust production upper-HNSW builder.",
    )
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument(
        "--cargo-target-dir",
        default=None,
        help=(
            "Optional external Cargo target directory, passed as CARGO_TARGET_DIR. "
            "Use this to avoid filling the repository filesystem."
        ),
    )
    parser.add_argument("--bundle-prefix", default="orion_numeric_import")
    parser.add_argument("--bundle-row-chunk-size", type=int, default=16_384)
    parser.add_argument(
        "--graphless-only",
        action="store_true",
        help="Stop after the graphless artifact and build evidence; do not invoke Rust.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = {
        "generation": args.generation,
        "num_shards": args.num_shards,
        "sample_denominator": args.sample_denominator,
        "upper_m": args.upper_m,
        "upper_ef_construction": args.upper_ef_construction,
        "upper_search_ef": args.upper_search_ef,
        "upper_k": args.upper_k,
        "k_overlap": args.k_overlap,
        "upper_build_batch_size": args.upper_build_batch_size,
        "dynamic_ef_base": args.dynamic_ef_base,
        "kmeans_iters": args.kmeans_iters,
        "topology_iters": args.topology_iters,
        "multi_assign_min_max_vote": args.multi_assign_min_max_vote,
        "bundle_row_chunk_size": args.bundle_row_chunk_size,
    }
    for name, value in positive_fields.items():
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.train_limit is not None and int(args.train_limit) <= 0:
        raise ValueError("--train-limit must be positive")
    if int(args.dynamic_ef_factor) < 0:
        raise ValueError("--dynamic-ef-factor must be non-negative")
    if int(args.multi_assign_vote_delta) < 0:
        raise ValueError("--multi-assign-vote-delta must be non-negative")
    if int(args.multi_assign_max_shards) < 0:
        raise ValueError("--multi-assign-max-shards must be non-negative")
    if int(args.upper_search_ef) < max(int(args.upper_k), int(args.k_overlap)):
        raise ValueError("--upper-search-ef must be at least max(--upper-k, --k-overlap)")
    if not args.bundle_prefix or Path(args.bundle_prefix).name != args.bundle_prefix:
        raise ValueError("--bundle-prefix must be a non-empty file-name component")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_vectors(
    hdf5_path: Path,
    train_limit: int | None,
    vector_distance: str,
) -> tuple[Any, dict[str, Any]]:
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 dataset not found: {hdf5_path}")
    with experiment.h5py.File(hdf5_path, "r") as handle:
        if "train" not in handle:
            raise ValueError(f"HDF5 dataset {hdf5_path} has no 'train' dataset")
        train_dataset = handle["train"]
        if len(train_dataset.shape) != 2:
            raise ValueError("HDF5 train dataset must be two-dimensional")
        total_rows = int(train_dataset.shape[0])
        dimension = int(train_dataset.shape[1])
        selected = experiment.slice_train_rows(train_dataset, train_limit)
        train = selected[:].astype(experiment.np.float32, copy=True)

    distance_config = experiment.vector_distance_config(vector_distance)
    train = experiment.prepare_vectors_for_distance(train, distance_config["name"])
    train = experiment.np.ascontiguousarray(train, dtype=experiment.np.float32)
    if train.ndim != 2 or len(train) == 0 or train.shape[1] == 0:
        raise ValueError("selected train dataset must be a non-empty two-dimensional array")
    if not experiment.np.isfinite(train).all():
        raise ValueError("selected train dataset contains a non-finite value")
    return train, {
        "path": str(hdf5_path),
        "size_bytes": hdf5_path.stat().st_size,
        "sha256": sha256_path(hdf5_path),
        "train_rows_total": total_rows,
        "train_rows_used": int(len(train)),
        "dimension": dimension,
    }


def validate_dataset_dependent_args(args: argparse.Namespace, upper_count: int) -> None:
    if upper_count < int(args.num_shards):
        raise ValueError(
            "upper sample is smaller than P: "
            f"upper_count={upper_count}, P={args.num_shards}; increase --train-limit, "
            "decrease --sample-denominator, or decrease --p"
        )
    if int(args.upper_k) > upper_count:
        raise ValueError(
            f"--upper-k {args.upper_k} exceeds upper sample size {upper_count}"
        )


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
        "orion_build_artifact",
        "--",
        str(graphless_path),
        str(production_path),
        "--seed",
        str(args.upper_graph_seed),
        "--m",
        str(args.upper_m),
        "--ef",
        str(args.upper_ef_construction),
    ]


def effective_cargo_target_dir(args: argparse.Namespace) -> str | None:
    configured = args.cargo_target_dir or os.environ.get("CARGO_TARGET_DIR")
    if not configured:
        return None
    return str(Path(configured).expanduser().resolve())


def run_rust_builder(
    args: argparse.Namespace,
    graphless_path: Path,
    production_path: Path,
) -> list[str]:
    command = rust_builder_command(args, graphless_path, production_path)
    environment = os.environ.copy()
    cargo_target_dir = effective_cargo_target_dir(args)
    if cargo_target_dir:
        environment["CARGO_TARGET_DIR"] = cargo_target_dir
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=environment)
    return command


def verify_production_artifact(production_path: Path) -> str:
    checksum_path = Path(f"{production_path}.sha256")
    if not production_path.is_file():
        raise RuntimeError(f"Rust builder did not create {production_path}")
    if not checksum_path.is_file():
        raise RuntimeError(f"Rust builder did not create {checksum_path}")
    expected = checksum_path.read_text(encoding="utf-8").strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise RuntimeError(f"invalid Rust builder checksum in {checksum_path}")
    actual = sha256_path(production_path)
    if actual != expected:
        raise RuntimeError(
            "Rust production artifact file checksum mismatch: "
            f"expected={expected}, actual={actual}"
        )
    payload = json.loads(production_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("upper_graph"), dict):
        raise RuntimeError("Rust builder output is not a production artifact with upper_graph")
    return actual


def read_graphless_binding(
    graphless_path: Path,
    args: argparse.Namespace,
    logical_point_count: int,
    physical_point_count: int,
    effective_num_shards: int,
) -> dict[str, Any]:
    payload = json.loads(graphless_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("graphless artifact root must be a JSON object")
    expected_fields = {
        "generation": int(args.generation),
        "shard_count": int(effective_num_shards),
        "logical_point_count": int(logical_point_count),
        "physical_point_count": int(physical_point_count),
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise RuntimeError(
                f"graphless artifact {field} mismatch: "
                f"expected={expected!r}, actual={payload.get(field)!r}"
            )
    layout_sha256 = str(payload.get("layout_sha256") or "").lower()
    if len(layout_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in layout_sha256
    ):
        raise RuntimeError("graphless artifact layout_sha256 is invalid")
    return {
        "generation": int(payload["generation"]),
        "layout_sha256": layout_sha256,
        "logical_point_count": int(payload["logical_point_count"]),
        "physical_point_count": int(payload["physical_point_count"]),
        "shard_count": int(payload["shard_count"]),
    }


def relative_file_records(output_dir: Path, excluded: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        if relative in excluded:
            continue
        records[relative] = {
            "sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_checksums(output_dir: Path) -> Path:
    checksum_path = output_dir / CHECKSUMS_NAME
    records = relative_file_records(output_dir, {CHECKSUMS_NAME})
    with checksum_path.open("x", encoding="utf-8", newline="\n") as handle:
        for relative, metadata in records.items():
            handle.write(f"{metadata['sha256']}  {relative}\n")
    return checksum_path


def routing_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generation": int(args.generation),
        "initial_num_shards": int(args.num_shards),
        "vector_distance": str(args.vector_distance),
        "vector_name": str(args.vector_name),
        "sample_denominator": int(args.sample_denominator),
        "upper_sample_seed": int(args.upper_sample_seed),
        "upper_m": int(args.upper_m),
        "upper_ef_construction": int(args.upper_ef_construction),
        "upper_search_ef": int(args.upper_search_ef),
        "upper_k": int(args.upper_k),
        "k_overlap": int(args.k_overlap),
        "upper_build_batch_size": int(args.upper_build_batch_size),
        "dynamic_ef_base": int(args.dynamic_ef_base),
        "dynamic_ef_factor": int(args.dynamic_ef_factor),
        "kmeans_iters": int(args.kmeans_iters),
        "kmeans_seed": int(args.kmeans_seed),
        "topology_iters": int(args.topology_iters),
        "use_multi_assign": not bool(args.disable_multi_assign),
        "multi_assign_min_max_vote": int(args.multi_assign_min_max_vote),
        "multi_assign_vote_delta": int(args.multi_assign_vote_delta),
        "multi_assign_max_shards": int(args.multi_assign_max_shards),
        "enable_fission": not bool(args.disable_fission),
        "upper_graph_seed": int(args.upper_graph_seed),
        "cargo_target_dir": effective_cargo_target_dir(args),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    hdf5_path = Path(args.hdf5_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite existing output path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    train, dataset_record = load_train_vectors(
        hdf5_path,
        args.train_limit,
        args.vector_distance,
    )
    distance_config = experiment.vector_distance_config(args.vector_distance)
    upper_indices = experiment.global_upper_indices(
        len(train),
        int(args.sample_denominator),
        int(args.upper_sample_seed),
    )
    validate_dataset_dependent_args(args, len(upper_indices))
    upper_index = experiment.build_upper_index(
        train[upper_indices],
        upper_indices.astype(experiment.np.int64, copy=False),
        int(train.shape[1]),
        int(args.upper_m),
        int(args.upper_ef_construction),
        int(args.upper_search_ef),
        distance_config["hnsw_space"],
    )
    point_to_l1s = experiment.compute_point_to_l1s(
        upper_index,
        train,
        int(args.k_overlap),
        int(args.upper_build_batch_size),
    )
    routing = experiment.build_original_routing_state(
        train,
        upper_indices,
        point_to_l1s,
        int(args.num_shards),
        int(args.kmeans_iters),
        int(args.kmeans_seed),
        int(args.topology_iters),
        use_multi_assign=not bool(args.disable_multi_assign),
        enable_fission=not bool(args.disable_fission),
        multi_assign_min_max_vote=int(args.multi_assign_min_max_vote),
        multi_assign_vote_delta=int(args.multi_assign_vote_delta),
        multi_assign_max_shards=int(args.multi_assign_max_shards),
    )

    graphless_path = output_dir / GRAPHLESS_NAME
    experiment.write_orion_graphless_artifact(
        train,
        upper_indices,
        routing.point_to_shards,
        int(routing.num_shards),
        graphless_path,
        generation=int(args.generation),
        vector_distance=distance_config["name"],
        upper_k=int(args.upper_k),
        upper_ef_search=int(args.upper_search_ef),
        dynamic_ef_base=int(args.dynamic_ef_base),
        dynamic_ef_factor=int(args.dynamic_ef_factor),
        vector_name=str(args.vector_name),
    )
    artifact_binding = read_graphless_binding(
        graphless_path,
        args,
        logical_point_count=len(train),
        physical_point_count=int(routing.total_assigned),
        effective_num_shards=int(routing.num_shards),
    )

    rust_command: list[str] | None = None
    production_path: Path | None = None
    production_sha256: str | None = None
    import_manifest_path: Path | None = None
    if not args.graphless_only:
        production_path = output_dir / f"generation-{int(args.generation)}.json"
        rust_command = run_rust_builder(args, graphless_path, production_path)
        production_sha256 = verify_production_artifact(production_path)
        import_manifest_path = experiment.write_orion_numeric_shard_import_bundle(
            train,
            routing.point_to_shards,
            int(routing.num_shards),
            output_dir,
            orion_artifact_path=production_path,
            vector_name=str(args.vector_name),
            prefix=str(args.bundle_prefix),
            row_chunk_size=int(args.bundle_row_chunk_size),
        )

    payload_files = relative_file_records(
        output_dir,
        {BUILD_MANIFEST_NAME, CHECKSUMS_NAME},
    )
    manifest = {
        "format_version": 1,
        "tool": "tools/orion_native_layout.py",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "graphless_only" if args.graphless_only else "production_bundle",
        "dataset": dataset_record,
        "parameters": routing_parameters(args),
        "artifact_binding": artifact_binding,
        "routing": {
            "initial_num_shards": int(routing.initial_num_shards),
            "effective_num_shards": int(routing.num_shards),
            "upper_point_count": int(len(upper_indices)),
            "logical_point_count": int(len(train)),
            "physical_point_count": int(routing.total_assigned),
            "expansion_ratio": float(routing.expansion_ratio),
            "topology_iterations": int(routing.topology_iterations),
            "shard_counts": [int(value) for value in routing.shard_counts.tolist()],
            "fission_events": routing.fission_events,
        },
        "outputs": {
            "graphless_artifact": graphless_path.name,
            "production_artifact": production_path.name if production_path else None,
            "import_manifest": import_manifest_path.name if import_manifest_path else None,
            "rust_builder_command": rust_command,
            "files": payload_files,
        },
    }
    build_manifest_path = output_dir / BUILD_MANIFEST_NAME
    write_json_new(build_manifest_path, manifest)
    checksum_path = write_checksums(output_dir)
    summary = {
        "output_dir": str(output_dir),
        "mode": manifest["mode"],
        "graphless_artifact": str(graphless_path),
        "production_artifact": str(production_path) if production_path else None,
        "production_artifact_sha256": production_sha256,
        "import_manifest": str(import_manifest_path) if import_manifest_path else None,
        "build_manifest": str(build_manifest_path),
        "checksums": str(checksum_path),
        "logical_point_count": int(len(train)),
        "physical_point_count": int(routing.total_assigned),
        "effective_num_shards": int(routing.num_shards),
        "expansion_ratio": float(routing.expansion_ratio),
        "layout_sha256": artifact_binding["layout_sha256"],
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
