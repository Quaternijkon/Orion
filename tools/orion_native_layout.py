#!/usr/bin/env python3
"""HDF5 adapter and thin caller of the authoritative Rust Orion layout builder.

Legacy command/audit helpers remain for historical experiment replay only.
The canonical build never calls Python clustering, voting or artifact writers.
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

import h5py
import numpy as np
import tempfile


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
            "Convert HDF5 to raw f32 and invoke the authoritative Rust Orion layout builder."
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
    parser.add_argument("--uri", help="Stream assignments into this Qdrant HTTP endpoint during the build.")
    parser.add_argument("--collection", help="Fresh Orion collection for --uri; created if absent.")
    parser.add_argument("--timeout-secs", type=int, default=7200)

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
    parser.add_argument("--rust-layout-binary", default=None, help="Prebuilt formal orion_build_layout binary.")
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
            "Historical compatibility flag. Native Rust builds reject graphless-only "
            "output and publish independently verified bundles."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if bool(getattr(args, "uri", None)) != bool(getattr(args, "collection", None)):
        raise ValueError("--uri and --collection must be supplied together")
    if getattr(args, "timeout_secs", 7200) <= 0:
        raise ValueError("--timeout-secs must be positive")
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
    with h5py.File(hdf5_path, "r") as handle:
        if "train" not in handle:
            raise ValueError(f"HDF5 dataset {hdf5_path} has no 'train' dataset")
        train_dataset = handle["train"]
        if len(train_dataset.shape) != 2:
            raise ValueError("HDF5 train dataset must be two-dimensional")
        total_rows = int(train_dataset.shape[0])
        dimension = int(train_dataset.shape[1])
        selected = train_dataset if train_limit is None else train_dataset[:train_limit]
        train = selected[:].astype(np.float32, copy=True)

    # Raw dataset adapter only; segment preprocessing is authoritative in Rust.
    train = np.ascontiguousarray(train, dtype=np.float32)
    if train.ndim != 2 or len(train) == 0 or train.shape[1] == 0:
        raise ValueError("selected train dataset must be a non-empty two-dimensional array")
    if not np.isfinite(train).all():
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
        expected_vector = np.asarray(
            expected_node.get("vector"), dtype="<f4"
        )
        actual_vector = np.asarray(actual_node.get("vector"), dtype="<f4")
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
        source_vector = np.asarray(source_node.get("vector"), dtype="<f4")
        production_vector = np.asarray(
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


def rust_layout_command(args: argparse.Namespace, *arguments: str) -> list[str]:
    binary = getattr(args, "rust_layout_binary", None) or os.environ.get("ORION_LAYOUT_BINARY")
    if binary:
        return [str(Path(binary).expanduser().resolve()), *arguments]
    return [str(args.cargo), "run", "--release", "-p", "collection",
            "--bin", "orion_build_layout", "--", *arguments]


def run_layout_cli(args: argparse.Namespace, *arguments: str) -> dict[str, Any]:
    environment = os.environ.copy()
    target = effective_cargo_target_dir(args)
    if target:
        environment["CARGO_TARGET_DIR"] = target
    result = subprocess.run(rust_layout_command(args, *arguments), cwd=REPO_ROOT,
                            env=environment, check=True, text=True, stdout=subprocess.PIPE)
    return json.loads(result.stdout)


def load_verified_rust_bundle(output_dir: Path) -> dict[str, Any]:
    """Read a Rust replay result for experiment clients; never infer collection readiness."""
    args = argparse.Namespace(cargo=os.environ.get("CARGO", "cargo"), cargo_target_dir=None,
                              rust_layout_binary=os.environ.get("ORION_LAYOUT_BINARY"))
    imported = run_layout_cli(args, "verify", "--bundle", str(output_dir))
    build_path = output_dir / BUILD_MANIFEST_NAME
    manifest = json.loads(build_path.read_text(encoding="utf-8"))
    artifact_path = output_dir / imported["orion_artifact_file"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    import_path = output_dir / manifest["import_manifest_file"]
    options = manifest["options"]
    with (output_dir / imported["vectors_file"]).open("rb") as vectors:
        smoke = np.frombuffer(vectors.read(imported["dimension"] * 4), dtype="<f4").tolist()
    parameters = dict(options)
    parameters.update({"upper_search_ef": options["upper_ef_search"],
                       "initial_num_shards": options["shard_count"],
                       "attachment_search_ef": options["efs"], "k_overlap": options["efs"],
                       "algorithm_version": manifest["algorithm_version"]})
    return {
        "layout_dir": str(output_dir), "build_manifest_path": str(build_path),
        "build_manifest_sha256": sha256_path(build_path),
        "artifact_path": str(artifact_path), "artifact_sha256": imported["orion_artifact_sha256"],
        "artifact": artifact, "import_manifest_path": str(import_path),
        "import_manifest_sha256": manifest["import_manifest_sha256"],
        "checksums": {artifact_path.name: imported["orion_artifact_sha256"],
                      import_path.name: manifest["import_manifest_sha256"]},
        "build_parameters": parameters, "attachment_search_ef": options["efs"],
        "balance_layout_proof": None, "l1_partition_layout_proof": None,
        "canonical_navigation_binding": {"single_upper_graph_build": True,
            "upper_graph_sha256": manifest["upper_graph_sha256"],
            "attachments_sha256": manifest["attachments_sha256"],
            "layout_sha256": imported["assignments_sha256"], "rust_replayed": True},
        "generation": imported["orion_generation"], "vector_schema": artifact["vector_schema"],
        "shard_count": imported["shard_count"], "logical_point_count": imported["point_count"],
        "physical_point_count": imported["total_point_copies"], "shard_weights": manifest["shard_counts"],
        "smoke_vector": smoke, "native_bundle_format": 2,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Adapt HDF5 to raw f32; the Rust CLI owns all layout and artifact decisions."""
    validate_args(args)
    if args.graphless_only:
        raise ValueError("--graphless-only is historical; the Rust builder publishes verified bundles")
    for field in ("rust_builder_binary", "rust_upper_hits_binary", "rust_rebind_binary"):
        if getattr(args, field, None):
            raise ValueError(f"--{field.replace('_', '-')} is historical; use --rust-layout-binary")
    output_dir = Path(args.output_dir).expanduser().resolve()
    streaming = bool(getattr(args, "uri", None))
    if os.path.lexists(output_dir) and not streaming:
        raise FileExistsError(f"refusing to overwrite existing output path: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    hdf5_path = Path(args.hdf5_path).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="orion-raw-", dir=output_dir.parent) as staging:
        raw_path = Path(staging) / "input.f32le"
        digest = hashlib.sha256()
        with h5py.File(hdf5_path, "r") as dataset, raw_path.open("xb") as raw:
            train = dataset["train"]
            if len(train.shape) != 2:
                raise ValueError("HDF5 train must be a two-dimensional matrix")
            rows = min(len(train), args.train_limit) if args.train_limit else len(train)
            dimension = int(train.shape[1])
            for start in range(0, rows, args.bundle_row_chunk_size):
                data = np.asarray(train[start:min(start + args.bundle_row_chunk_size, rows)],
                                  dtype="<f4", order="C").tobytes()
                raw.write(data)
                digest.update(data)
            raw.flush()
            os.fsync(raw.fileno())
        request = {
            "input": {"format_version": 1, "path": str(raw_path),
                      "rows": rows, "dimension": dimension, "sha256": digest.hexdigest()},
            "options": {
                "generation": args.generation, "shard_count": args.num_shards,
                "distance": "Cosine" if args.vector_distance == "cosine" else "Euclid",
                "vector_name": args.vector_name, "sample_denominator": args.sample_denominator,
                "sample_seed": args.upper_sample_seed, "upper_m": args.upper_m,
                "upper_ef_construct": args.upper_ef_construction, "upper_seed": args.upper_graph_seed,
                "efs": args.k_overlap, "upper_k": args.upper_k,
                "upper_ef_search": args.upper_search_ef, "dynamic_ef_base": args.dynamic_ef_base,
                "dynamic_ef_factor": args.dynamic_ef_factor, "kmeans_iterations": args.kmeans_iters,
                "kmeans_seed": args.kmeans_seed, "topology_iterations": args.topology_iters,
                "bundle_prefix": args.bundle_prefix,
            },
        }
        request_path = Path(staging) / "request.json"
        write_json_new(request_path, request)
        command = ["build-install" if streaming else "build", "--request", str(request_path),
                   "--output", str(output_dir)]
        if streaming:
            command += ["--uri", args.uri, "--collection", args.collection,
                        "--timeout-secs", str(args.timeout_secs)]
        manifest = run_layout_cli(args, *command)
    import_path = output_dir / manifest["import_manifest_file"]
    imported = json.loads(import_path.read_text(encoding="utf-8"))
    summary = {
        "output_dir": str(output_dir), "mode": "streamed_install" if streaming else "production_bundle",
        "algorithm_version": manifest["algorithm_version"],
        "production_artifact": str(output_dir / imported["orion_artifact_file"]),
        "production_artifact_sha256": imported["orion_artifact_sha256"],
        "import_manifest": str(import_path),
        "build_manifest": str(output_dir / BUILD_MANIFEST_NAME),
        "layout_sha256": imported["assignments_sha256"],
        "logical_point_count": imported["point_count"],
        "physical_point_count": imported["total_point_copies"],
        "effective_num_shards": imported["shard_count"],
        "upper_graph_sha256": manifest["upper_graph_sha256"],
        "attachments_sha256": manifest["attachments_sha256"],
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
