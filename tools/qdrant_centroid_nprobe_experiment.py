#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-centroid Qdrant collections and evaluate centroid-routed nprobe search "
            "without modifying or rebuilding Qdrant."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6335")
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--collection-prefix", default="qdrant_nprobe44")
    parser.add_argument("--num-shards", type=int, default=44)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--assignment-batch-size", type=int, default=50000)
    parser.add_argument("--upload-batch-size", type=int, default=1024)
    parser.add_argument("--query-count", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=100)
    parser.add_argument("--hnsw-ef", type=int, default=32)
    parser.add_argument("--nprobes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 44])
    parser.add_argument("--output-dir", default="results/qdrant_nprobe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def request_json(base_url: str, method: str, path: str, body: dict | None = None, timeout: float = 300.0) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.read().decode()}") from exc
    return payload


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return arr / norms


def kmeans_pp_init(sample: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n, dim = sample.shape
    centroids = np.empty((k, dim), dtype=np.float32)
    first = rng.integers(0, n)
    centroids[0] = sample[first]
    closest = 1.0 - sample @ centroids[0]
    for i in range(1, k):
        probs = np.maximum(closest, 1e-12)
        probs = probs / probs.sum()
        idx = rng.choice(n, p=probs)
        centroids[i] = sample[idx]
        dist = 1.0 - sample @ centroids[i]
        closest = np.minimum(closest, dist)
    return normalize_rows(centroids)


def train_centroids(train: np.ndarray, k: int, sample_size: int, iters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = train.shape[0]
    if sample_size < n:
        indices = rng.choice(n, size=sample_size, replace=False)
        sample = train[indices]
    else:
        sample = train
    sample = normalize_rows(sample.astype(np.float32, copy=False))
    centroids = kmeans_pp_init(sample, k, rng)
    for _ in range(iters):
        scores = sample @ centroids.T
        assign = np.argmax(scores, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        for c in range(k):
            mask = assign == c
            if np.any(mask):
                new_centroids[c] = sample[mask].mean(axis=0)
                counts[c] = mask.sum()
            else:
                new_centroids[c] = centroids[c]
                counts[c] = 1
        centroids = normalize_rows(new_centroids)
    return centroids


def create_collection(base_url: str, name: str, dim: int, m: int, ef_construct: int) -> None:
    request_json(
        base_url,
        "PUT",
        f"/collections/{urllib.parse.quote(name, safe='')}",
        body={
            "vectors": {"size": dim, "distance": "Cosine"},
            "shard_number": 1,
            "replication_factor": 1,
            "write_consistency_factor": 1,
            "hnsw_config": {
                "m": m,
                "ef_construct": ef_construct,
                "full_scan_threshold": 10,
                "max_indexing_threads": 0,
            },
            "optimizers_config": {"default_segment_number": 1, "indexing_threshold": 10},
        },
    )


def delete_collection_if_exists(base_url: str, name: str) -> None:
    try:
        request_json(base_url, "DELETE", f"/collections/{urllib.parse.quote(name, safe='')}")
    except RuntimeError as exc:
        if "404" not in str(exc):
            raise


def upsert_batch(base_url: str, name: str, ids: list[int], vectors: list[list[float]]) -> None:
    request_json(
        base_url,
        "PUT",
        f"/collections/{urllib.parse.quote(name, safe='')}/points?wait=true",
        body={"batch": {"ids": ids, "vectors": vectors}},
        timeout=600.0,
    )


def collection_info(base_url: str, name: str) -> dict:
    return request_json(base_url, "GET", f"/collections/{urllib.parse.quote(name, safe='')}")["result"]


def wait_collection_indexed(base_url: str, name: str, expected_points: int, timeout_sec: float = 3600.0) -> dict:
    start = time.perf_counter()
    while True:
        info = collection_info(base_url, name)
        if int(info.get("indexed_vectors_count") or 0) >= expected_points and int(info.get("points_count") or 0) == expected_points:
            return info
        if time.perf_counter() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for {name} to finish indexing")
        time.sleep(1.0)


def batch_search_collection(
    base_url: str,
    collection: str,
    query_vectors: np.ndarray,
    query_indices: list[int],
    top_k: int,
    hnsw_ef: int,
    batch_size: int = 100,
) -> dict[int, list[tuple[float, int]]]:
    results: dict[int, list[tuple[float, int]]] = {}
    for start_idx in range(0, len(query_indices), batch_size):
        idx_chunk = query_indices[start_idx : start_idx + batch_size]
        body = {
            "searches": [
                {
                    "vector": query_vectors[q_idx].tolist(),
                    "limit": top_k,
                    "params": {"hnsw_ef": hnsw_ef},
                    "with_payload": False,
                    "with_vector": False,
                }
                for q_idx in idx_chunk
            ]
        }
        payload = request_json(
            base_url,
            "POST",
            f"/collections/{urllib.parse.quote(collection, safe='')}/points/search/batch",
            body=body,
            timeout=600.0,
        )
        for q_idx, per_query in zip(idx_chunk, payload["result"]):
            merged = [(float(item["score"]), int(item["id"]) - 1) for item in per_query]
            results[q_idx] = merged
    return results


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5_path, "r") as handle:
        train = handle["train"][:].astype(np.float32, copy=True)
        queries = handle["test"][: args.query_count].astype(np.float32, copy=True)
        neighbors = handle["neighbors"][: args.query_count, : args.top_k].astype(np.int32, copy=True)

    train = normalize_rows(train)
    queries = normalize_rows(queries)

    centroids = train_centroids(train, args.num_shards, args.sample_size, args.kmeans_iters, args.seed)
    centroid_scores = train @ centroids.T
    assignments = np.argmax(centroid_scores, axis=1)

    # Recompute exact centroids for the built partition.
    final_centroids = np.zeros_like(centroids)
    counts = np.zeros(args.num_shards, dtype=np.int64)
    for shard_id in range(args.num_shards):
        mask = assignments == shard_id
        counts[shard_id] = int(mask.sum())
        if counts[shard_id] > 0:
            final_centroids[shard_id] = train[mask].mean(axis=0)
        else:
            final_centroids[shard_id] = centroids[shard_id]
    final_centroids = normalize_rows(final_centroids)

    shard_names = [f"{args.collection_prefix}_shard_{i:02d}" for i in range(args.num_shards)]

    build_rows = []
    for shard_id, shard_name in enumerate(shard_names):
        if not args.reuse_existing:
            delete_collection_if_exists(args.base_url, shard_name)
            create_collection(args.base_url, shard_name, train.shape[1], args.hnsw_m, args.ef_construct)

        shard_indices = np.where(assignments == shard_id)[0]
        if not args.reuse_existing:
            for start_idx in range(0, len(shard_indices), args.upload_batch_size):
                idx_chunk = shard_indices[start_idx : start_idx + args.upload_batch_size]
                ids = (idx_chunk + 1).tolist()
                vectors = train[idx_chunk].tolist()
                upsert_batch(args.base_url, shard_name, ids, vectors)
        info = wait_collection_indexed(args.base_url, shard_name, len(shard_indices))
        build_rows.append(
            {
                "shard_id": shard_id,
                "collection": shard_name,
                "points_count": len(shard_indices),
                "indexed_vectors_count": int(info.get("indexed_vectors_count") or 0),
                "segments_count": int(info.get("segments_count") or 0),
            }
        )

    with (output_dir / "builds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(build_rows[0].keys()))
        writer.writeheader()
        for row in build_rows:
            writer.writerow(row)

    query_centroid_scores = queries @ final_centroids.T
    nprobe_rows = []
    for nprobe in args.nprobes:
        nprobe = min(nprobe, args.num_shards)
        nearest = np.argpartition(-query_centroid_scores, kth=nprobe - 1, axis=1)[:, :nprobe]

        per_shard_queries: dict[int, list[int]] = defaultdict(list)
        for q_idx in range(len(queries)):
            for shard_id in nearest[q_idx]:
                per_shard_queries[int(shard_id)].append(q_idx)

        per_query_candidates: list[list[tuple[float, int]]] = [[] for _ in range(len(queries))]
        start = time.perf_counter()
        for shard_id, query_indices in per_shard_queries.items():
            shard_results = batch_search_collection(
                args.base_url,
                shard_names[shard_id],
                queries,
                query_indices,
                args.top_k,
                args.hnsw_ef,
            )
            for q_idx, items in shard_results.items():
                per_query_candidates[q_idx].extend(items)

        hits = 0
        for q_idx, candidates in enumerate(per_query_candidates):
            candidates.sort(reverse=True)
            top = []
            seen = set()
            for score, idx in candidates:
                if idx in seen:
                    continue
                seen.add(idx)
                top.append(idx)
                if len(top) == args.top_k:
                    break
            gt = set(map(int, neighbors[q_idx]))
            hits += len(set(top) & gt)

        wall = time.perf_counter() - start
        row = {
            "nprobe": nprobe,
            "hnsw_ef": args.hnsw_ef,
            "query_count": len(queries),
            "top_k": args.top_k,
            "recall_at_k": hits / (len(queries) * args.top_k),
            "qps": len(queries) / wall,
            "wall_s": wall,
            "avg_selected_shards": float(nprobe),
        }
        nprobe_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    with (output_dir / "nprobe_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(nprobe_rows[0].keys()))
        writer.writeheader()
        for row in nprobe_rows:
            writer.writerow(row)

    summary = {
        "base_url": args.base_url,
        "hdf5_path": args.hdf5_path,
        "num_shards": args.num_shards,
        "hnsw_m": args.hnsw_m,
        "ef_construct": args.ef_construct,
        "hnsw_ef": args.hnsw_ef,
        "nprobe_rows": nprobe_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
