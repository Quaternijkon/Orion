"""Build a custom-sharded Qdrant collection from a point->shard layout.

Used for the baseline arms (hash, k-means): every point is placed on exactly the
shard its layout assigns, under a `centroid_XX` shard key, so the collection has
the same shard structure and HNSW parameters as the Orion collection and differs
only in *where points land*. Single-assignment, so ids are stored raw (no copy-id
encoding, no source_id payload) and recall reads the id directly.

Reuses the REST primitives from tools/qdrant_two_level_routing_experiment.py so
the create/shard-key/upsert calls match exactly what the tool does.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ANALYSIS = REPO_ROOT / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402
from fanout_headroom import Layout  # noqa: E402
from cluster import discover_peers  # noqa: E402


def wait_for_green(base_url: str, collection: str, timeout_s: float = 3600.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = requests.get(f"{base_url}/collections/{collection}", timeout=30).json()
        if info["result"]["status"] == "green":
            return
        time.sleep(5)
    raise TimeoutError(f"{collection} did not reach green within {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--train-path", required=True, help="npy of vectors (row i = point i)")
    parser.add_argument("--layout-file", required=True, help="point->shard .npz")
    parser.add_argument("--distance", choices=("Cosine", "Euclid", "Dot"), required=True)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=100)
    parser.add_argument("--upsert-batch", type=int, default=1000)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    base_url = discover_peers()[0].base_url
    train = np.load(args.train_path, mmap_mode="r")
    layout = Layout.load(args.layout_file)
    if layout.points != train.shape[0]:
        raise SystemExit(
            f"layout has {layout.points} points, train has {train.shape[0]}"
        )
    dim = int(train.shape[1])
    num_shards = layout.shards

    if args.recreate:
        experiment.delete_collection_if_exists(base_url, args.collection)
    experiment.create_collection(
        base_url, args.collection, dim, args.m, args.ef_construct,
        vector_distance=args.distance,
    )
    for shard_id in range(num_shards):
        experiment.create_shard_key(base_url, args.collection, experiment.shard_key_for_id(shard_id))

    # Invert the layout: list of point ids per shard, then upsert per shard in batches.
    points_by_shard: list[list[int]] = [[] for _ in range(num_shards)]
    for point in range(layout.points):
        for shard in layout.shards_of(point):
            points_by_shard[int(shard)].append(point)

    started = time.perf_counter()
    uploaded = 0
    total = layout.copies
    for shard_id, point_ids in enumerate(points_by_shard):
        shard_key = experiment.shard_key_for_id(shard_id)
        for start in range(0, len(point_ids), args.upsert_batch):
            chunk = point_ids[start : start + args.upsert_batch]
            vectors = [np.asarray(train[p], dtype=np.float32).tolist() for p in chunk]
            experiment.upsert_points(base_url, args.collection, shard_key, chunk, vectors)
            uploaded += len(chunk)
            if uploaded % (args.upsert_batch * 40) < args.upsert_batch:
                rate = uploaded / max(time.perf_counter() - started, 1e-9)
                print(f"  upserted {uploaded}/{total} ({rate:.0f} pts/s)", flush=True)

    wait_for_green(base_url, args.collection)
    info = requests.get(f"{base_url}/collections/{args.collection}", timeout=60).json()["result"]
    print(
        f"{args.collection}: status={info['status']} points={info['points_count']} "
        f"shards={num_shards} build_s={time.perf_counter() - started:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
