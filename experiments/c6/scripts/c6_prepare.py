#!/usr/bin/env python3
"""Build one frozen custom-shard collection from a production Orion import bundle.

The partition, upper graph, memberships, and vectors come from an existing
checksum-bound layout.  This script only materializes those fixed assignments as
selectable custom shard keys so C6 can switch P0-P6 without rebuilding indexes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from c6_protocol import canonical_json_sha256, sha256_path, utc_timestamp, write_json_atomic  # noqa: E402


@dataclass(frozen=True)
class FrozenBundle:
    layout_dir: Path
    manifest_path: Path
    artifact_path: Path
    assignments_path: Path
    vectors_path: Path
    manifest: dict[str, Any]
    artifact: dict[str, Any]
    point_to_shards: list[list[int]]
    upper_indices: np.ndarray
    vectors: np.memmap


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--layout-dir", required=True)
    parser.add_argument(
        "--vectors-path-override",
        help="Checksum-matching vector file used when a historical bundle hardlink is absent.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=100)
    parser.add_argument("--upload-batch-size", type=int, default=512)
    parser.add_argument("--upload-concurrency", type=int, default=16)
    parser.add_argument("--upload-queue-capacity-batches", type=int, default=4)
    parser.add_argument(
        "--shard-placement",
        choices=("round_robin", "none"),
        default="round_robin",
    )
    parser.add_argument("--peer-ids", type=int, nargs="+", default=[])
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.hnsw_m <= 0
        or args.ef_construct <= 0
        or args.upload_batch_size <= 0
        or getattr(args, "upload_concurrency", 16) <= 0
        or getattr(args, "upload_queue_capacity_batches", 4) <= 0
    ):
        raise ValueError("HNSW and upload parameters must be positive")
    if args.shard_placement == "round_robin" and not args.peer_ids:
        raise ValueError("round_robin placement requires --peer-ids")
    if args.shard_placement == "none" and args.peer_ids:
        raise ValueError("--peer-ids must be omitted with placement=none")
    if any(peer_id < 0 for peer_id in args.peer_ids) or len(set(args.peer_ids)) != len(
        args.peer_ids
    ):
        raise ValueError("peer IDs must be unique non-negative integers")


def safe_bundle_file(layout_dir: Path, raw_name: Any, field: str) -> Path:
    if not isinstance(raw_name, str) or not raw_name or Path(raw_name).name != raw_name:
        raise ValueError(f"{field} must name one file in the layout directory")
    path = (layout_dir / raw_name).resolve()
    if path.parent != layout_dir:
        raise ValueError(f"{field} escapes the layout directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_assignments(
    path: str | Path, *, point_count: int, shard_count: int
) -> list[list[int]]:
    result: list[list[int]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for expected_id, line in enumerate(handle):
            try:
                row = json.loads(line)
                point_id = int(row["id"])
                shards = [int(value) for value in row["shards"]]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid assignment line for point {expected_id}") from exc
            if point_id != expected_id:
                raise ValueError(
                    f"assignment IDs are not contiguous: expected={expected_id}, actual={point_id}"
                )
            if not shards or len(set(shards)) != len(shards):
                raise ValueError(f"point {point_id} has empty or duplicate shard assignments")
            if any(shard_id < 0 or shard_id >= shard_count for shard_id in shards):
                raise ValueError(f"point {point_id} has an out-of-range shard assignment")
            # Canonical strict multi-assignment preserves the first navigation
            # appearance of tied maximum-vote shards so the primary shard is
            # deterministic.  Membership order therefore need not be numeric.
            result.append(shards)
    if len(result) != point_count:
        raise ValueError(
            f"assignment row count differs: actual={len(result)}, expected={point_count}"
        )
    return result


def numeric_upper_indices(artifact: dict[str, Any], point_count: int) -> np.ndarray:
    raw_nodes = artifact.get("upper_nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("production artifact is missing upper_nodes")
    labels = []
    for index, node in enumerate(raw_nodes):
        label = node.get("label") if isinstance(node, dict) else None
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"upper node {index} does not have a numeric point label")
        if label < 0 or label >= point_count:
            raise ValueError(f"upper node label {label} is outside the logical point range")
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise ValueError("production artifact contains duplicate upper labels")
    return np.asarray(labels, dtype=np.int64)


def load_frozen_bundle(
    layout_dir: str | Path, vectors_path_override: str | Path | None = None
) -> FrozenBundle:
    root = Path(layout_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / "orion_numeric_import.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported Orion numeric import manifest version")
    point_count = int(manifest.get("point_count") or 0)
    dimension = int(manifest.get("dimension") or 0)
    shard_count = int(manifest.get("shard_count") or 0)
    total_copies = int(manifest.get("total_point_copies") or 0)
    if min(point_count, dimension, shard_count, total_copies) <= 0:
        raise ValueError("numeric import manifest has invalid dimensions or counts")
    artifact_path = safe_bundle_file(
        root, manifest.get("orion_artifact_file"), "orion_artifact_file"
    )
    assignments_path = safe_bundle_file(
        root, manifest.get("assignments_file"), "assignments_file"
    )
    manifest_vectors_path = (root / str(manifest.get("vectors_file") or "")).resolve()
    if manifest_vectors_path.is_file():
        vectors_path = manifest_vectors_path
    elif vectors_path_override is not None:
        vectors_path = Path(vectors_path_override).expanduser().resolve()
        if not vectors_path.is_file():
            raise FileNotFoundError(vectors_path)
    else:
        raise FileNotFoundError(manifest_vectors_path)
    expected_checksums = {
        artifact_path: str(manifest.get("orion_artifact_sha256") or ""),
        assignments_path: str(manifest.get("assignments_sha256") or ""),
        vectors_path: str(manifest.get("vectors_sha256") or ""),
    }
    for path, expected in expected_checksums.items():
        actual = sha256_path(path)
        if actual != expected:
            raise ValueError(f"bundle checksum mismatch for {path.name}: {actual} != {expected}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("upper_graph") is None:
        raise ValueError("C6 requires a production artifact with a serialized upper graph")
    if int(artifact.get("shard_count") or 0) != shard_count:
        raise ValueError("artifact and import manifest shard counts differ")
    if int(artifact.get("logical_point_count") or 0) != point_count:
        raise ValueError("artifact and import manifest logical point counts differ")
    if int(artifact.get("physical_point_count") or 0) != total_copies:
        raise ValueError("artifact and import manifest physical point counts differ")
    if str(artifact.get("layout_sha256") or "") != expected_checksums[assignments_path]:
        raise ValueError("artifact layout_sha256 differs from the assignment file")
    schema = artifact.get("vector_schema") or {}
    if int(schema.get("dimension") or 0) != dimension:
        raise ValueError("artifact and import manifest dimensions differ")
    point_to_shards = parse_assignments(
        assignments_path, point_count=point_count, shard_count=shard_count
    )
    if sum(map(len, point_to_shards)) != total_copies:
        raise ValueError("assignment copy count differs from the import manifest")
    expected_vector_bytes = point_count * dimension * np.dtype("<f4").itemsize
    if vectors_path.stat().st_size != expected_vector_bytes:
        raise ValueError(
            f"vector file size differs: actual={vectors_path.stat().st_size}, "
            f"expected={expected_vector_bytes}"
        )
    vectors = np.memmap(
        vectors_path,
        dtype="<f4",
        mode="r",
        shape=(point_count, dimension),
    )
    upper_indices = numeric_upper_indices(artifact, point_count)
    return FrozenBundle(
        layout_dir=root,
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        assignments_path=assignments_path,
        vectors_path=vectors_path,
        manifest=manifest,
        artifact=artifact,
        point_to_shards=point_to_shards,
        upper_indices=upper_indices,
        vectors=vectors,
    )


def qdrant_distance(artifact: dict[str, Any]) -> str:
    distance = str((artifact.get("vector_schema") or {}).get("distance") or "")
    mapping = {
        "Cosine": "Cosine",
        "Euclid": "Euclid",
        "Dot": "Dot",
        "Manhattan": "Manhattan",
    }
    if distance not in mapping:
        raise ValueError(f"unsupported Orion vector distance {distance!r}")
    return mapping[distance]


def build_identity(bundle: FrozenBundle, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "routing_mode": "c6_frozen_orion_custom_shards",
        "artifact_sha256": bundle.manifest["orion_artifact_sha256"],
        "layout_sha256": bundle.artifact["layout_sha256"],
        "assignments_sha256": bundle.manifest["assignments_sha256"],
        "vectors_sha256": bundle.manifest["vectors_sha256"],
        "logical_point_count": bundle.manifest["point_count"],
        "physical_point_count": bundle.manifest["total_point_copies"],
        "num_shards": bundle.manifest["shard_count"],
        "distance": qdrant_distance(bundle.artifact),
        "hnsw_m": args.hnsw_m,
        "hnsw_ef_construct": args.ef_construct,
        "entry_point_injection": "production_upper_labels",
        "point_id_encoding": "shard_id_times_N_plus_1_plus_source_id_plus_1",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    bundle = load_frozen_bundle(args.layout_dir, args.vectors_path_override)
    from tools import qdrant_two_level_routing_experiment as experiment

    identity = build_identity(bundle, args)
    result = experiment.ensure_collection_from_point_shards(
        args.base_url,
        args.collection,
        bundle.vectors,
        bundle.point_to_shards,
        bundle.upper_indices,
        int(bundle.manifest["shard_count"]),
        args.hnsw_m,
        args.ef_construct,
        args.upload_batch_size,
        args.reuse_existing,
        args.shard_placement,
        args.peer_ids,
        vector_distance=qdrant_distance(bundle.artifact),
        routing_build_metadata=identity,
        force_hnsw_for_nonempty_shards=True,
        upload_concurrency=args.upload_concurrency,
        upload_queue_capacity_batches=args.upload_queue_capacity_batches,
    )
    shard_key_map = {
        str(shard_id): experiment.shard_key_for_id(shard_id)
        for shard_id in range(int(bundle.manifest["shard_count"]))
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "shard-key-map.json", shard_key_map)
    completion = {
        "record_type": "c6_frozen_collection",
        "format_version": 1,
        "timestamp": utc_timestamp(),
        "collection": args.collection,
        "base_url": args.base_url,
        "layout_dir": str(bundle.layout_dir),
        "layout_manifest_sha256": sha256_path(bundle.manifest_path),
        "artifact_sha256": bundle.manifest["orion_artifact_sha256"],
        "layout_sha256": bundle.artifact["layout_sha256"],
        "index_identity": identity,
        "index_identity_sha256": canonical_json_sha256(identity),
        "hnsw": {"m": args.hnsw_m, "ef_construct": args.ef_construct},
        "shard_key_map": shard_key_map,
        "shard_key_map_sha256": canonical_json_sha256(shard_key_map),
        "collection_result": result,
        "policy_switching_rebuild_required": False,
    }
    write_json_atomic(output / "frozen-collection.json", completion)
    print(json.dumps(completion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
