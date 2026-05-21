#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Qdrant batch search throughput/recall on an existing collection "
            "without modifying or rebuilding Qdrant."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6335")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--query-count", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hnsw-ef", type=int, default=32)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 10, 25, 50, 100, 200, 500],
    )
    parser.add_argument(
        "--output-dir",
        default="results/qdrant_batch_bench",
    )
    return parser.parse_args()


def load_queries_and_gt(path: str, query_count: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        queries = handle["test"][:query_count].astype("float32", copy=True)
        neighbors = handle["neighbors"][:query_count, :top_k].astype("int32", copy=True)

    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    queries = queries / norms
    return queries, neighbors


def batch_search(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    hnsw_ef: int,
    batch_size: int,
) -> dict:
    total_hits = 0
    total_queries = len(queries)
    start = time.perf_counter()

    for start_idx in range(0, total_queries, batch_size):
        chunk = queries[start_idx : start_idx + batch_size]
        body = {
            "searches": [
                {
                    "vector": q.tolist(),
                    "limit": top_k,
                    "params": {"hnsw_ef": hnsw_ef},
                    "with_payload": False,
                    "with_vector": False,
                }
                for q in chunk
            ]
        }
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/collections/{collection}/points/search/batch",
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            payload = json.loads(response.read().decode())

        results = payload["result"]
        for local_idx, result in enumerate(results):
            ids = {int(item["id"]) - 1 for item in result}
            gt = set(map(int, neighbors[start_idx + local_idx]))
            total_hits += len(ids & gt)

    wall = time.perf_counter() - start
    return {
        "batch_size": batch_size,
        "hnsw_ef": hnsw_ef,
        "query_count": total_queries,
        "top_k": top_k,
        "recall_at_k": total_hits / (total_queries * top_k),
        "qps": total_queries / wall,
        "wall_s": wall,
    }


def main() -> int:
    args = parse_args()
    queries, neighbors = load_queries_and_gt(args.hdf5_path, args.query_count, args.top_k)

    output_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for batch_size in args.batch_sizes:
        row = batch_search(
            base_url=args.base_url,
            collection=args.collection,
            queries=queries,
            neighbors=neighbors,
            top_k=args.top_k,
            hnsw_ef=args.hnsw_ef,
            batch_size=batch_size,
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    summary = {
        "base_url": args.base_url,
        "collection": args.collection,
        "hdf5_path": args.hdf5_path,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
