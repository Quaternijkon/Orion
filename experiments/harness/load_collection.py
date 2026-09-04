"""Create a distributed Qdrant collection and ingest vectors for measurement.

The collection is created with ``shard_number`` shards so that the *server*
performs fan-out and merging. Ingestion runs against the controller peer and is
not part of any measured path.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import requests

from cluster import discover_peers


def wait_for_green(base_url: str, collection: str, timeout_s: float = 1800.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = requests.get(f"{base_url}/collections/{collection}", timeout=30)
        response.raise_for_status()
        status = response.json()["result"]["status"]
        if status == "green":
            return
        time.sleep(5)
    raise TimeoutError(f"{collection} did not reach green within {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--distance", choices=("Euclid", "Cosine", "Dot"), required=True)
    parser.add_argument("--replication", type=int, default=1)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construct", type=int, default=200)
    parser.add_argument("--upsert-batch", type=int, default=1000)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    peers = discover_peers()
    base_url = peers[0].base_url
    train = np.load(args.train_path, mmap_mode="r")

    if args.recreate:
        requests.delete(f"{base_url}/collections/{args.collection}", timeout=120)

    create = requests.put(
        f"{base_url}/collections/{args.collection}",
        json={
            "vectors": {"size": int(train.shape[1]), "distance": args.distance},
            "shard_number": args.shards,
            "replication_factor": args.replication,
            "hnsw_config": {"m": args.m, "ef_construct": args.ef_construct},
        },
        timeout=300,
    )
    if not create.ok:
        raise RuntimeError(f"create failed: {create.status_code} {create.text}")

    started = time.perf_counter()
    total = int(train.shape[0])
    for start in range(0, total, args.upsert_batch):
        window = np.asarray(train[start : start + args.upsert_batch], dtype=np.float32)
        response = requests.put(
            f"{base_url}/collections/{args.collection}/points",
            json={
                "points": [
                    {"id": start + offset, "vector": vector.tolist()}
                    for offset, vector in enumerate(window)
                ]
            },
            params={"wait": "true"},
            timeout=600,
        )
        if not response.ok:
            raise RuntimeError(f"upsert failed at {start}: {response.text}")
        if (start // args.upsert_batch) % 20 == 0:
            done = min(start + args.upsert_batch, total)
            rate = done / max(time.perf_counter() - started, 1e-9)
            print(f"  upserted {done}/{total} ({rate:.0f} pts/s)", flush=True)

    wait_for_green(base_url, args.collection)
    info = requests.get(f"{base_url}/collections/{args.collection}", timeout=60).json()
    result = info["result"]
    print(
        f"{args.collection}: status={result['status']} "
        f"points={result['points_count']} shards={args.shards} "
        f"build_s={time.perf_counter() - started:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
