#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy a Method4 collection with the exact logical point-to-shard "
            "membership recovered from a source collection, but a different "
            "physical placement map."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6833")
    parser.add_argument("--source-collection", default="qdrant_controller_idea_full_20260521")
    parser.add_argument("--target-collection", required=True)
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--num-shards", type=int, default=46)
    parser.add_argument("--placement-map", default="results/method4_claim_g_physical_layout_20260704/analysis_20260704_164039/placement_maps.json")
    parser.add_argument("--worker-count", type=int, default=3)
    parser.add_argument("--placement-name", choices=["round_robin", "size_balanced", "method4_aware"], required=True)
    parser.add_argument("--placement-peer-uri-contains", nargs="+", default=["qdrant_shard_"])
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=100)
    parser.add_argument("--upload-batch-size", type=int, default=1024)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--output-root", default="results/method4_claim_g_matched_layout_deploy_20260704")
    parser.add_argument("--qdrant-tool", default="tools/qdrant_two_level_routing_experiment.py")
    return parser.parse_args()


def load_qdrant_tool(path: str | Path) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location("qdrant_two_level_routing_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_nested_placement_map(args: argparse.Namespace) -> dict[str, int]:
    data = json.loads(Path(args.placement_map).read_text(encoding="utf-8"))
    worker_maps = data.get(str(args.worker_count))
    if not isinstance(worker_maps, dict):
        raise ValueError(f"worker-count map {args.worker_count!r} not found in {args.placement_map}")
    placement = worker_maps.get(args.placement_name)
    if not isinstance(placement, dict):
        raise ValueError(f"placement {args.placement_name!r} not found for worker count {args.worker_count}")
    return {str(shard_key): int(peer_id) for shard_key, peer_id in placement.items()}


def recover_full_point_to_shards(args: argparse.Namespace, q2l: Any, num_points: int) -> tuple[list[list[int]], list[int]]:
    point_to_shards: list[list[int]] = [[] for _ in range(num_points)]
    shard_copy_counts: list[int] = []
    for shard_id in range(args.num_shards):
        shard_key = q2l.shard_key_for_id(shard_id)
        offset: Any = None
        shard_count = 0
        while True:
            body: dict[str, Any] = {
                "limit": args.scroll_page_size,
                "with_payload": ["source_id"],
                "with_vector": False,
                "shard_key": shard_key,
            }
            if offset is not None:
                body["offset"] = offset
            result = q2l.request_json(
                args.base_url,
                "POST",
                f"/collections/{urllib.parse.quote(args.source_collection, safe='')}/points/scroll",
                body=body,
                timeout=300.0,
            )["result"]
            points = result.get("points") or []
            shard_count += len(points)
            for point in points:
                source_id = q2l.source_id_from_scrolled_point(point, num_points)
                if 0 <= source_id < num_points:
                    point_to_shards[int(source_id)].append(int(shard_id))
            offset = result.get("next_page_offset")
            if offset is None:
                break
        shard_copy_counts.append(shard_count)
        print(f"recovered source shard {shard_id}: {shard_count} copies", flush=True)
    missing = sum(1 for shards in point_to_shards if not shards)
    if missing:
        raise RuntimeError(f"source collection has {missing} logical points with no shard copy")
    return point_to_shards, shard_copy_counts


def main() -> int:
    args = parse_args()
    q2l = load_qdrant_tool(args.qdrant_tool)
    output_dir = Path(args.output_root) / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    with h5py.File(args.hdf5_path, "r") as handle:
        train = handle["train"][:].astype(np.float32, copy=True)
    train = q2l.normalize_rows(train)
    num_points = int(train.shape[0])
    upper_indices = q2l.global_upper_indices(num_points, args.sample_denominator, args.upper_sample_seed)
    point_to_shards, shard_copy_counts = recover_full_point_to_shards(args, q2l, num_points)
    placement_map = load_nested_placement_map(args)
    peer_ids = q2l.cluster_peer_ids(args.base_url, args.placement_peer_uri_contains)
    build_row = q2l.ensure_collection_from_point_shards(
        args.base_url,
        args.target_collection,
        train,
        point_to_shards,
        upper_indices,
        args.num_shards,
        args.hnsw_m,
        args.ef_construct,
        args.upload_batch_size,
        args.reuse_existing,
        "map",
        peer_ids,
        placement_map,
    )
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_collection": args.source_collection,
        "target_collection": args.target_collection,
        "hdf5_path": args.hdf5_path,
        "num_points": num_points,
        "num_shards": args.num_shards,
        "worker_count": args.worker_count,
        "placement_name": args.placement_name,
        "placement_map": args.placement_map,
        "placement_peer_uri_contains": args.placement_peer_uri_contains,
        "peer_ids": peer_ids,
        "source_assigned_copies": sum(len(shards) for shards in point_to_shards),
        "source_shard_copy_counts": shard_copy_counts,
        "build": build_row,
        "notes": [
            "Logical point-to-shard membership is recovered from source_collection.",
            "Only physical shard placement is changed for target_collection.",
        ],
    }
    write_csv(output_dir / "builds.csv", [build_row])
    write_csv(
        output_dir / "source_shard_counts.csv",
        [
            {
                "shard_id": shard_id,
                "shard_key": q2l.shard_key_for_id(shard_id),
                "source_scrolled_points": count,
            }
            for shard_id, count in enumerate(shard_copy_counts)
        ],
    )
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(build_row, ensure_ascii=False), flush=True)
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
