#!/usr/bin/env python3
"""Build one deterministic SIFT1M native auto-sharded collection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import h5py
import numpy as np


def load_experiment_module(repo_root: Path):
    source = repo_root / "tools" / "qdrant_two_level_routing_experiment.py"
    spec = importlib.util.spec_from_file_location("qdrant_experiment", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://10.10.1.1:6333")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--machine-count", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--hdf5-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    exp = load_experiment_module(repo_root)
    exp.delete_collection_if_exists(args.base_url, args.collection)
    created = exp.create_numeric_auto_shard_collection(
        args.base_url,
        args.collection,
        dim=128,
        num_shards=args.machine_count,
        m=32,
        ef_construct=200,
        vector_distance="Euclid",
        full_scan_threshold=10,
        indexing_threshold=10,
    )
    _, peers, _ = exp.cluster_peer_map(args.base_url)
    peer_by_ip = {
        uri.split("//", 1)[1].split(":", 1)[0]: peer_id
        for peer_id, uri in peers.items()
    }
    all_peer_ids = [peer_by_ip[f"10.10.1.{index}"] for index in range(1, 5)]
    target = {
        shard_id: all_peer_ids[shard_id]
        for shard_id in range(args.machine_count)
    }
    move = exp.move_numeric_shards_explicit(
        args.base_url,
        args.collection,
        all_peer_ids,
        target,
        expected_shard_count=args.machine_count,
        include_controller=True,
        transfer_method="snapshot",
    )
    with h5py.File(args.hdf5_path, "r") as handle:
        train = np.asarray(handle["train"], dtype=np.float32)
    upload_started = time.time()
    upload = exp.upsert_numeric_auto_points(
        args.base_url,
        args.collection,
        train,
        batch_size=args.batch_size,
        timeout=900.0,
    )
    upload["seconds"] = time.time() - upload_started
    info = exp.wait_collection_indexed(
        args.base_url, args.collection, len(train), timeout_sec=10_800
    )
    cluster = exp.collection_cluster_info(args.base_url, args.collection)
    placement = exp.discover_numeric_shard_placement(
        args.base_url,
        args.collection,
        expected_shard_count=args.machine_count,
    )
    record = {
        "created": created,
        "builder_server": exp.request_json(args.base_url, "GET", "/"),
        "machine_count": args.machine_count,
        "target_placement": target,
        "actual_placement": placement,
        "peer_uris": peers,
        "move": move,
        "upload": upload,
        "collection_info": info,
        "collection_cluster": cluster,
        "build_parameters": {
            "dataset": "SIFT1M",
            "hnsw_m": 32,
            "ef_construct": 200,
            "full_scan_threshold": 10,
            "indexing_threshold": 10,
            "sharding_method": "auto",
            "replication_factor": 1,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "READY",
                "machine_count": args.machine_count,
                "placement": placement,
                "upload_seconds": upload["seconds"],
                "points_count": info.get("points_count"),
                "indexed_vectors_count": info.get("indexed_vectors_count"),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
