#!/usr/bin/env python3
"""Build a native Orion routing layout without duplicating the Orion algorithm.

This entry point is deliberately an orchestration layer around one immutable
Qdrant/Rust upper HNSW.  It builds that graph once, exports every offline
L0-to-L1 attachment through the production ``OrionRouter``, derives topology and
memberships, then finalizes those memberships onto the unchanged graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
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
UPPER_SEED_GRAPHLESS_NAME = "upper-seed-graphless-orion.json"
ATTACHMENT_HITS_NAME = "upper-attachments.counted.bin"
ATTACHMENT_MANIFEST_NAME = "upper-attachments.manifest.json"
FINALIZATION_SIDECAR_NAME = "finalize-memberships.json"
BUILD_MANIFEST_NAME = "build-manifest.json"
CHECKSUMS_NAME = "checksums.sha256"
DETERMINISTIC_ATTACHMENT_BUILD_THREADS = 1


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
    parser.add_argument(
        "--attachment-search-ef",
        type=int,
        default=10,
        help=(
            "Canonical build-time efs. It must equal --k-overlap, so the search "
            "budget and the number of voting results are the same."
        ),
    )
    parser.add_argument("--upper-search-ef", type=int, default=100)
    parser.add_argument("--upper-k", type=int, default=100)
    parser.add_argument(
        "--allow-decoupled-runtime-upper-search",
        action="store_true",
        help=(
            "Diagnostic-only: allow runtime upper_search_ef != upper_k. "
            "Faithful main-idea builds keep both equal to EF_SEARCH_UP."
        ),
    )
    parser.add_argument("--k-overlap", type=int, default=10)
    parser.add_argument("--upper-build-batch-size", type=int, default=10_000)

    parser.add_argument("--dynamic-ef-base", type=int, default=20)
    parser.add_argument("--dynamic-ef-factor", type=int, default=4)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--kmeans-seed", type=int, default=1)
    parser.add_argument("--topology-iters", type=int, default=50)
    parser.add_argument(
        "--disable-topology-refinement",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-multi-assign", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--multi-assign-min-max-vote", type=int, default=2, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--multi-assign-vote-delta", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--multi-assign-max-shards", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument("--disable-fission", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--balance-mode",
        choices=(
            "none",
            "capacity_constrained",
            "post_layout_capacity_constrained",
        ),
        default="none",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--balance-min-load-ratio", type=float, default=0.99, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--balance-max-load-ratio", type=float, default=1.01, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--balance-max-passes", type=int, default=8, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--balance-max-vote-loss",
        type=int,
        default=3,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--balance-l1-max-vote-loss",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--balance-l0-max-vote-loss",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--upper-graph-seed",
        type=int,
        default=100,
        help="Deterministic seed passed to the Rust production upper-HNSW builder.",
    )
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument(
        "--rust-builder-binary",
        default=None,
        help=(
            "Optional prebuilt orion_build_artifact executable. Its resolved path and "
            "SHA-256 are recorded in the build manifest."
        ),
    )
    parser.add_argument(
        "--rust-upper-hits-binary",
        default=None,
        help=(
            "Optional prebuilt orion_export_upper_hits executable. The canonical "
            "build uses it to generate every offline attachment from the frozen "
            "Qdrant upper graph."
        ),
    )
    parser.add_argument(
        "--rust-rebind-binary",
        default=None,
        help=(
            "Optional prebuilt orion_rebind_memberships executable used only to "
            "finalize memberships onto the already-built upper graph."
        ),
    )
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
        help=(
            "Stop after the final graphless layout and attachment evidence. The "
            "Qdrant upper graph is still built exactly once because canonical "
            "attachments must come from it; membership finalization and the import "
            "manifest are skipped."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = {
        "generation": args.generation,
        "num_shards": args.num_shards,
        "sample_denominator": args.sample_denominator,
        "upper_m": args.upper_m,
        "upper_ef_construction": args.upper_ef_construction,
        "attachment_search_ef": args.attachment_search_ef,
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
    if int(args.attachment_search_ef) != int(args.k_overlap):
        raise ValueError("canonical build requires --attachment-search-ef == --k-overlap")
    if bool(args.disable_topology_refinement):
        raise ValueError("canonical build requires upper self-vote refinement")
    if bool(args.disable_multi_assign):
        raise ValueError("canonical multi-assignment cannot be disabled")
    if int(args.multi_assign_min_max_vote) != 2:
        raise ValueError("canonical multi-assignment uses max_vote > 1")
    if int(args.multi_assign_vote_delta) != 0:
        raise ValueError("canonical multi-assignment includes only maximum-vote ties")
    if int(args.multi_assign_max_shards) != 0:
        raise ValueError("canonical multi-assignment cannot cap maximum-vote ties")
    if str(args.balance_mode) != "none":
        raise ValueError("canonical build currently excludes load-balancing intervention")
    if int(args.upper_search_ef) < int(args.upper_k):
        raise ValueError("--upper-search-ef must be at least --upper-k")
    if (
        int(args.upper_search_ef) != int(args.upper_k)
        and not args.allow_decoupled_runtime_upper_search
    ):
        raise ValueError(
            "faithful Orion requires --upper-search-ef == --upper-k; "
            "use --allow-decoupled-runtime-upper-search only for diagnostics"
        )
    if not args.bundle_prefix or Path(args.bundle_prefix).name != args.bundle_prefix:
        raise ValueError("--bundle-prefix must be a non-empty file-name component")
    for field, label in (
        ("rust_builder_binary", "Rust builder"),
        ("rust_upper_hits_binary", "Rust upper-hits exporter"),
        ("rust_rebind_binary", "Rust membership finalizer"),
    ):
        raw = getattr(args, field)
        if raw is None:
            continue
        binary = Path(raw).expanduser().resolve()
        if not binary.is_file():
            raise FileNotFoundError(f"{label} binary not found: {binary}")
        if not os.access(binary, os.X_OK):
            raise PermissionError(f"{label} binary is not executable: {binary}")


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
    builder_args = [
        str(graphless_path),
        str(production_path),
        "--seed",
        str(args.upper_graph_seed),
        "--m",
        str(args.upper_m),
        "--ef",
        str(args.upper_ef_construction),
    ]
    if args.rust_builder_binary is not None:
        return [
            str(Path(args.rust_builder_binary).expanduser().resolve()),
            *builder_args,
        ]
    return [
        str(args.cargo),
        "run",
        "--release",
        "-p",
        "collection",
        "--example",
        "orion_build_artifact",
        "--",
        *builder_args,
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


def rust_upper_hits_command(
    args: argparse.Namespace,
    production_path: Path,
    vectors_path: Path,
    row_count: int,
    dimension: int,
    hits_path: Path,
    manifest_path: Path,
) -> list[str]:
    exporter_args = [
        str(production_path),
        str(vectors_path),
        str(int(row_count)),
        str(int(dimension)),
        str(int(args.k_overlap)),
        str(int(args.attachment_search_ef)),
        str(hits_path),
        str(manifest_path),
    ]
    if args.rust_upper_hits_binary is not None:
        return [
            str(Path(args.rust_upper_hits_binary).expanduser().resolve()),
            *exporter_args,
        ]
    return [
        str(args.cargo),
        "run",
        "--release",
        "-p",
        "collection",
        "--example",
        "orion_export_upper_hits",
        "--",
        *exporter_args,
    ]


def run_rust_upper_hits_export(
    args: argparse.Namespace,
    production_path: Path,
    vectors_path: Path,
    row_count: int,
    dimension: int,
    hits_path: Path,
    manifest_path: Path,
) -> list[str]:
    command = rust_upper_hits_command(
        args,
        production_path,
        vectors_path,
        row_count,
        dimension,
        hits_path,
        manifest_path,
    )
    environment = os.environ.copy()
    cargo_target_dir = effective_cargo_target_dir(args)
    if cargo_target_dir:
        environment["CARGO_TARGET_DIR"] = cargo_target_dir
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=environment)
    return command


def rust_rebind_command(
    args: argparse.Namespace,
    source_path: Path,
    sidecar_path: Path,
    production_path: Path,
    *,
    mode: str,
) -> list[str]:
    if mode not in {"finalize-build", "runtime-profile"}:
        raise ValueError(f"unsupported Orion rebind mode: {mode}")
    rebind_args = [
        str(source_path),
        str(sidecar_path),
        str(production_path),
        f"--{mode}",
    ]
    if args.rust_rebind_binary is not None:
        return [
            str(Path(args.rust_rebind_binary).expanduser().resolve()),
            *rebind_args,
        ]
    return [
        str(args.cargo),
        "run",
        "--release",
        "-p",
        "collection",
        "--example",
        "orion_rebind_memberships",
        "--",
        *rebind_args,
    ]


def run_rust_rebind(
    args: argparse.Namespace,
    source_path: Path,
    sidecar_path: Path,
    production_path: Path,
    *,
    mode: str,
) -> list[str]:
    command = rust_rebind_command(
        args,
        source_path,
        sidecar_path,
        production_path,
        mode=mode,
    )
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


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def production_upper_graph_sha256(production_path: Path) -> str:
    payload = json.loads(production_path.read_text(encoding="utf-8"))
    graph = payload.get("upper_graph") if isinstance(payload, dict) else None
    if not isinstance(graph, dict):
        raise RuntimeError(f"production artifact {production_path} has no upper_graph")
    return canonical_json_sha256(graph)


def verify_attachment_export(
    args: argparse.Namespace,
    production_path: Path,
    vectors_path: Path,
    hits_path: Path,
    manifest_path: Path,
    *,
    row_count: int,
    dimension: int,
) -> dict[str, Any]:
    if not hits_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Rust upper-hits exporter did not publish both outputs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("upper attachment manifest root must be an object")
    expected = {
        "format_version": 2,
        "artifact_sha256": sha256_path(production_path),
        "upper_graph_present": True,
        "vectors_sha256": sha256_path(vectors_path),
        "row_count": int(row_count),
        "dimension": int(dimension),
        "top_k": int(args.k_overlap),
        "search_ef": int(args.attachment_search_ef),
        "hits_format": "counted_rows_u32le_then_u64le_v1",
        "hits_sha256": sha256_path(hits_path),
    }
    mismatches = {
        field: {"expected": value, "actual": manifest.get(field)}
        for field, value in expected.items()
        if manifest.get(field) != value
    }
    if mismatches:
        raise RuntimeError(f"upper attachment manifest mismatch: {mismatches}")
    total_hits = manifest.get("total_hits")
    min_hits = manifest.get("min_hits_per_row")
    max_hits = manifest.get("max_hits_per_row")
    hits_size_bytes = manifest.get("hits_size_bytes")
    for name, value in (
        ("total_hits", total_hits),
        ("min_hits_per_row", min_hits),
        ("max_hits_per_row", max_hits),
        ("hits_size_bytes", hits_size_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"upper attachment manifest {name} must be non-negative")
    if min_hits > max_hits or max_hits > int(args.k_overlap):
        raise RuntimeError("upper attachment per-row hit counts are invalid")
    if not min_hits * int(row_count) <= total_hits <= max_hits * int(row_count):
        raise RuntimeError("upper attachment total hit count is inconsistent")
    expected_size = int(row_count) * 4 + int(total_hits) * 8
    if hits_size_bytes != expected_size or hits_path.stat().st_size != expected_size:
        raise RuntimeError("upper attachment counted-row file has the wrong size")
    return manifest


def load_point_to_l1s_from_upper_hits(
    hits_path: Path,
    *,
    row_count: int,
    top_k: int,
    upper_labels: set[int],
) -> list[list[int]]:
    point_to_l1s: list[list[int]] = []
    with hits_path.open("rb") as handle:
        for row_index in range(int(row_count)):
            count_bytes = handle.read(4)
            if len(count_bytes) != 4:
                raise RuntimeError(
                    f"upper attachment counted-row file ends before row {row_index}"
                )
            hit_count = struct.unpack("<I", count_bytes)[0]
            if hit_count > int(top_k):
                raise RuntimeError(
                    f"upper attachment row {row_index} has {hit_count} hits, exceeding {top_k}"
                )
            row_bytes = handle.read(int(hit_count) * 8)
            if len(row_bytes) != int(hit_count) * 8:
                raise RuntimeError(
                    f"upper attachment row {row_index} is truncated"
                )
            labels = [
                int(value[0])
                for value in struct.iter_unpack("<Q", row_bytes)
            ]
            unknown = next((label for label in labels if label not in upper_labels), None)
            if unknown is not None:
                raise RuntimeError(
                    f"upper attachment row {row_index} references unknown upper label {unknown}"
                )
            point_to_l1s.append(labels)
        if handle.read(1):
            raise RuntimeError("upper attachment counted-row file has trailing bytes")
    return point_to_l1s


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


def write_finalization_sidecar(
    path: Path,
    *,
    source_artifact_path: Path,
    graphless_path: Path,
    upper_indices: Any,
    upper_owner_by_point: list[int],
) -> Path:
    source = json.loads(source_artifact_path.read_text(encoding="utf-8"))
    graphless = json.loads(graphless_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(graphless, dict):
        raise RuntimeError("Orion finalization inputs must be JSON objects")
    source_generation = int(source.get("generation") or 0)
    final_generation = int(graphless.get("generation") or 0)
    if source_generation <= 0 or source_generation != final_generation:
        raise RuntimeError("build finalization must preserve the source generation")
    upper_owner_shards = [
        int(upper_owner_by_point[int(point_id)]) for point_id in upper_indices.tolist()
    ]
    if len(upper_owner_shards) != len(source.get("upper_nodes") or []):
        raise RuntimeError("finalization owners do not match source upper-node order")
    sidecar = {
        "format_version": 2,
        "source_artifact_sha256": sha256_path(source_artifact_path),
        "source_generation": source_generation,
        "generation": final_generation,
        "layout_sha256": str(graphless.get("layout_sha256") or ""),
        "shard_count": int(graphless.get("shard_count") or 0),
        "physical_point_count": int(graphless.get("physical_point_count") or 0),
        "upper_owner_shards": upper_owner_shards,
    }
    write_json_new(path, sidecar)
    return path


def verify_finalized_artifact(
    source_path: Path,
    graphless_path: Path,
    production_path: Path,
) -> dict[str, str]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    graphless = json.loads(graphless_path.read_text(encoding="utf-8"))
    production = json.loads(production_path.read_text(encoding="utf-8"))
    for name, payload in (
        ("source", source),
        ("graphless", graphless),
        ("production", production),
    ):
        if not isinstance(payload, dict):
            raise RuntimeError(f"{name} artifact root must be an object")
    source_graph = source.get("upper_graph")
    production_graph = production.get("upper_graph")
    if not isinstance(source_graph, dict) or production_graph != source_graph:
        raise RuntimeError("membership finalization changed the frozen upper graph")
    production_without_graph = dict(production)
    production_without_graph.pop("upper_graph", None)
    graphless_without_nodes = dict(graphless)
    production_without_nodes = dict(production_without_graph)
    graphless_nodes = graphless_without_nodes.pop("upper_nodes", None)
    finalized_nodes = production_without_nodes.pop("upper_nodes", None)
    if production_without_nodes != graphless_without_nodes:
        raise RuntimeError(
            "final production artifact differs from graphless layout outside "
            "upper_graph/upper_nodes"
        )
    if not isinstance(graphless_nodes, list) or not isinstance(finalized_nodes, list):
        raise RuntimeError("finalization artifacts must contain upper_nodes arrays")
    if len(graphless_nodes) != len(finalized_nodes):
        raise RuntimeError("finalization changed upper_nodes length")
    for index, (expected_node, actual_node) in enumerate(
        zip(graphless_nodes, finalized_nodes, strict=True)
    ):
        if expected_node.get("label") != actual_node.get("label") or expected_node.get(
            "owner_shard"
        ) != actual_node.get("owner_shard"):
            raise RuntimeError(
                f"final production artifact changed upper node {index} identity or membership"
            )
        expected_vector = experiment.np.asarray(
            expected_node.get("vector"), dtype="<f4"
        )
        actual_vector = experiment.np.asarray(actual_node.get("vector"), dtype="<f4")
        if expected_vector.shape != actual_vector.shape or expected_vector.tobytes() != (
            actual_vector.tobytes()
        ):
            raise RuntimeError(
                f"final production artifact changed upper node {index} vector bits"
            )
    source_nodes = source.get("upper_nodes") or []
    production_nodes = production.get("upper_nodes") or []
    if len(source_nodes) != len(production_nodes):
        raise RuntimeError("membership finalization changed upper-node count")
    for index, (source_node, production_node) in enumerate(
        zip(source_nodes, production_nodes, strict=True)
    ):
        if source_node.get("label") != production_node.get("label"):
            raise RuntimeError(f"membership finalization changed upper label {index}")
        source_vector = experiment.np.asarray(source_node.get("vector"), dtype="<f4")
        production_vector = experiment.np.asarray(
            production_node.get("vector"), dtype="<f4"
        )
        if source_vector.shape != production_vector.shape or source_vector.tobytes() != (
            production_vector.tobytes()
        ):
            raise RuntimeError(
                f"membership finalization changed upper vector bits at node {index}"
            )
    graph_sha256 = canonical_json_sha256(source_graph)
    return {
        "upper_graph_sha256": graph_sha256,
        "source_artifact_sha256": sha256_path(source_path),
        "production_artifact_sha256": sha256_path(production_path),
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
    builder_binary = (
        Path(args.rust_builder_binary).expanduser().resolve()
        if args.rust_builder_binary is not None
        else None
    )
    upper_hits_binary = (
        Path(args.rust_upper_hits_binary).expanduser().resolve()
        if args.rust_upper_hits_binary is not None
        else None
    )
    rebind_binary = (
        Path(args.rust_rebind_binary).expanduser().resolve()
        if args.rust_rebind_binary is not None
        else None
    )
    return {
        "generation": int(args.generation),
        "initial_num_shards": int(args.num_shards),
        "vector_distance": str(args.vector_distance),
        "vector_name": str(args.vector_name),
        "sample_denominator": int(args.sample_denominator),
        "upper_sample_seed": int(args.upper_sample_seed),
        "upper_m": int(args.upper_m),
        "upper_ef_construction": int(args.upper_ef_construction),
        "attachment_search_ef": int(args.attachment_search_ef),
        "efs": int(args.k_overlap),
        "upper_search_ef": int(args.upper_search_ef),
        "upper_k": int(args.upper_k),
        "allow_decoupled_runtime_upper_search": bool(
            args.allow_decoupled_runtime_upper_search
        ),
        "k_overlap": int(args.k_overlap),
        "upper_build_batch_size": int(args.upper_build_batch_size),
        "dynamic_ef_base": int(args.dynamic_ef_base),
        "dynamic_ef_factor": int(args.dynamic_ef_factor),
        "kmeans_iters": int(args.kmeans_iters),
        "kmeans_seed": int(args.kmeans_seed),
        "topology_iters": int(args.topology_iters),
        "enable_topology_refinement": True,
        "initial_partition": "kmeans_without_capacity_correction",
        "use_multi_assign": True,
        "multi_assign_policy": "all_max_vote_ties_else_nearest_when_max_vote_is_one",
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
        "enable_fission": False,
        "balance_mode": "none",
        "lower_hnsw_construction": "independent_full_multilayer_per_shard",
        "lower_hnsw_retains_non_base_layers": True,
        "routed_search_start_level": 0,
        "upper_graph_seed": int(args.upper_graph_seed),
        "attachment_navigator": "qdrant_production_upper_graph",
        "single_upper_graph_build": True,
        "attachment_index_random_seed": int(args.upper_graph_seed),
        "attachment_index_build_threads": DETERMINISTIC_ATTACHMENT_BUILD_THREADS,
        "attachment_query_tie_break": "qdrant_distance_then_node_index",
        "rust_builder_binary": str(builder_binary) if builder_binary else None,
        "rust_builder_binary_sha256": (
            sha256_path(builder_binary) if builder_binary else None
        ),
        "rust_upper_hits_binary": (
            str(upper_hits_binary) if upper_hits_binary else None
        ),
        "rust_upper_hits_binary_sha256": (
            sha256_path(upper_hits_binary) if upper_hits_binary else None
        ),
        "rust_rebind_binary": str(rebind_binary) if rebind_binary else None,
        "rust_rebind_binary_sha256": (
            sha256_path(rebind_binary) if rebind_binary else None
        ),
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

    # Canonical lifecycle: build the Qdrant production upper graph first and
    # exactly once.  The neutral one-shard memberships exist only to satisfy the
    # typed artifact schema before the real layout is known.
    vectors_path = output_dir / f"{str(args.bundle_prefix)}.f32le"
    vectors_sha256 = experiment.write_orion_numeric_vector_file(
        train,
        vectors_path,
        row_chunk_size=int(args.bundle_row_chunk_size),
    )
    upper_seed_graphless_path = output_dir / UPPER_SEED_GRAPHLESS_NAME
    experiment.write_orion_upper_seed_graphless_artifact(
        train,
        upper_indices,
        upper_seed_graphless_path,
        generation=int(args.generation),
        vector_distance=distance_config["name"],
        upper_k=int(args.upper_k),
        upper_ef_search=int(args.upper_search_ef),
        dynamic_ef_base=int(args.dynamic_ef_base),
        dynamic_ef_factor=int(args.dynamic_ef_factor),
        vector_name=str(args.vector_name),
    )
    upper_source_path = output_dir / (
        f"upper-source-generation-{int(args.generation)}.json"
    )
    rust_builder_command_used = run_rust_builder(
        args,
        upper_seed_graphless_path,
        upper_source_path,
    )
    upper_source_artifact_sha256 = verify_production_artifact(upper_source_path)
    upper_graph_sha256 = production_upper_graph_sha256(upper_source_path)

    attachment_hits_path = output_dir / ATTACHMENT_HITS_NAME
    attachment_manifest_path = output_dir / ATTACHMENT_MANIFEST_NAME
    rust_attachment_command_used = run_rust_upper_hits_export(
        args,
        upper_source_path,
        vectors_path,
        len(train),
        int(train.shape[1]),
        attachment_hits_path,
        attachment_manifest_path,
    )
    attachment_manifest = verify_attachment_export(
        args,
        upper_source_path,
        vectors_path,
        attachment_hits_path,
        attachment_manifest_path,
        row_count=len(train),
        dimension=int(train.shape[1]),
    )
    if attachment_manifest["vectors_sha256"] != vectors_sha256:
        raise RuntimeError("upper attachment export used unexpected vector bytes")
    point_to_l1s = load_point_to_l1s_from_upper_hits(
        attachment_hits_path,
        row_count=len(train),
        top_k=int(args.k_overlap),
        upper_labels={int(value) for value in upper_indices.tolist()},
    )
    routing = experiment.build_canonical_routing_state(
        train,
        upper_indices,
        point_to_l1s,
        int(args.num_shards),
        int(args.kmeans_iters),
        int(args.kmeans_seed),
        int(args.topology_iters),
    )

    graphless_path = output_dir / GRAPHLESS_NAME
    experiment.write_orion_graphless_artifact(
        train,
        upper_indices,
        routing.point_to_shards,
        routing.l1_to_shard,
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
    finalization_sidecar_path: Path | None = None
    finalized_binding: dict[str, str] | None = None
    if not args.graphless_only:
        production_path = output_dir / f"generation-{int(args.generation)}.json"
        finalization_sidecar_path = output_dir / FINALIZATION_SIDECAR_NAME
        write_finalization_sidecar(
            finalization_sidecar_path,
            source_artifact_path=upper_source_path,
            graphless_path=graphless_path,
            upper_indices=upper_indices,
            upper_owner_by_point=routing.l1_to_shard,
        )
        rust_command = run_rust_rebind(
            args,
            upper_source_path,
            finalization_sidecar_path,
            production_path,
            mode="finalize-build",
        )
        production_sha256 = verify_production_artifact(production_path)
        finalized_binding = verify_finalized_artifact(
            upper_source_path,
            graphless_path,
            production_path,
        )
        if finalized_binding["upper_graph_sha256"] != upper_graph_sha256:
            raise RuntimeError("finalized upper graph checksum changed")
        import_manifest_path = experiment.write_orion_numeric_shard_import_bundle(
            train,
            routing.point_to_shards,
            int(routing.num_shards),
            output_dir,
            orion_artifact_path=production_path,
            vector_name=str(args.vector_name),
            prefix=str(args.bundle_prefix),
            row_chunk_size=int(args.bundle_row_chunk_size),
            prewritten_vectors_path=vectors_path,
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
        "navigation_binding": {
            "format_version": 1,
            "single_upper_graph_build": True,
            "attachment_navigator": "qdrant_production_upper_graph",
            "upper_source_artifact_sha256": upper_source_artifact_sha256,
            "upper_graph_sha256": upper_graph_sha256,
            "attachments_sha256": str(attachment_manifest["hits_sha256"]),
            "attachments_manifest_sha256": sha256_path(attachment_manifest_path),
            "attachments_source_artifact_sha256": str(
                attachment_manifest["artifact_sha256"]
            ),
            "vectors_sha256": vectors_sha256,
            "layout_sha256": artifact_binding["layout_sha256"],
            "final_upper_graph_sha256": (
                finalized_binding["upper_graph_sha256"]
                if finalized_binding is not None
                else None
            ),
            "graph_identity_verified_after_finalization": (
                finalized_binding is not None
            ),
        },
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
            "balance_diagnostics": getattr(routing, "balance_diagnostics", None),
            "assignment_policy": "all_max_vote_ties_else_nearest_when_max_vote_is_one",
            "initial_partition": "kmeans_without_capacity_correction",
            "self_vote_refinement": True,
            "upper_routing_ownership": "single_frozen_owner",
            "lower_multi_assignment_can_widen_routing": False,
        },
        "outputs": {
            "graphless_artifact": graphless_path.name,
            "upper_seed_graphless_artifact": upper_seed_graphless_path.name,
            "upper_source_artifact": upper_source_path.name,
            "upper_attachments": attachment_hits_path.name,
            "upper_attachments_manifest": attachment_manifest_path.name,
            "finalization_sidecar": (
                finalization_sidecar_path.name if finalization_sidecar_path else None
            ),
            "production_artifact": production_path.name if production_path else None,
            "import_manifest": import_manifest_path.name if import_manifest_path else None,
            "rust_builder_command": rust_builder_command_used,
            "rust_attachment_export_command": rust_attachment_command_used,
            "rust_rebind_command": rust_command,
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
        "upper_source_artifact": str(upper_source_path),
        "upper_source_artifact_sha256": upper_source_artifact_sha256,
        "upper_graph_sha256": upper_graph_sha256,
        "attachments_sha256": str(attachment_manifest["hits_sha256"]),
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
