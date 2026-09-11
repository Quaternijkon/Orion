"""Upload a point->shard(s) layout as a copy-id-encoded, peer-distributed
collection, using the tool's authoritative multi-assign path.

Unlike upload_custom_shards.py (raw ids, single-assign only), this uses
ensure_collection_from_point_shards, so a *replicated* layout (a point on several
shards) is stored with distinct copy ids + a source_id payload -- exactly the
encoding the navigation router's --source-id-dedup-block-size -1 expects. Shards
are round-robin placed across all discovered peers so single-assign and
replicated collections share the same shard->peer mapping and are comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402
from fanout_headroom import Layout  # noqa: E402
from cluster import discover_peers  # noqa: E402


def discover_peer_ids(base_url: str) -> list[int]:
    info = requests.get(f"{base_url}/cluster", timeout=30).json()["result"]
    ids = {int(info["peer_id"])}
    for pid in (info.get("peers") or {}).keys():
        ids.add(int(pid))
    return sorted(ids)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", required=True)
    p.add_argument("--train-path", required=True)
    p.add_argument("--layout-file", required=True)
    p.add_argument("--distance", choices=("Cosine", "Euclid", "Dot"), default="Cosine")
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--ef-construct", type=int, default=100)
    p.add_argument("--upsert-batch", type=int, default=1000)
    p.add_argument("--normalize", action="store_true")
    args = p.parse_args()

    base_url = discover_peers()[0].base_url
    peer_ids = discover_peer_ids(base_url)
    train = np.load(args.train_path).astype(np.float32)
    if args.normalize:
        train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    layout = Layout.load(args.layout_file)
    if layout.points != train.shape[0]:
        raise SystemExit(f"layout {layout.points} pts != train {train.shape[0]}")
    point_to_shards = [
        layout.indices[layout.indptr[i] : layout.indptr[i + 1]].astype(int).tolist()
        for i in range(layout.points)
    ]
    num_shards = layout.shards
    print(f"{args.collection}: {layout.points} pts, {layout.copies} copies "
          f"(exp={layout.copies/layout.points:.4f}), P={num_shards}, peers={peer_ids}")

    row = experiment.ensure_collection_from_point_shards(
        base_url,
        args.collection,
        train,
        point_to_shards,
        None,  # upper_indices (ordering only)
        num_shards,
        int(args.m),
        int(args.ef_construct),
        int(args.upsert_batch),
        False,  # reuse_existing -> recreate fresh
        "round_robin",
        peer_ids,
        vector_distance=args.distance,
    )
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
