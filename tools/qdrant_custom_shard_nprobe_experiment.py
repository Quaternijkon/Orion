#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single custom-sharded Qdrant collection using centroid-based shard keys, "
            "then benchmark selective nprobe search over shard_key subsets."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6335")
    parser.add_argument("--collection", default="qdrant_custom_centroid44")
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--num-shards", type=int, default=44)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--upload-batch-size", type=int, default=1024)
    parser.add_argument("--query-count", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=100)
    parser.add_argument("--hnsw-ef", type=int, default=32)
    parser.add_argument("--nprobes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 44])
    parser.add_argument("--output-dir", default="results/qdrant_custom_nprobe")
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
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.read().decode()}") from exc


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
    if sample_size < len(train):
        indices = rng.choice(len(train), size=sample_size, replace=False)
        sample = train[indices]
    else:
        sample = train
    sample = normalize_rows(sample.astype(np.float32, copy=False))
    centroids = kmeans_pp_init(sample, k, rng)
    for _ in range(iters):
        scores = sample @ centroids.T
        assignments = np.argmax(scores, axis=1)
        new_centroids = np.zeros_like(centroids)
        for shard_id in range(k):
            mask = assignments == shard_id
            if np.any(mask):
                new_centroids[shard_id] = sample[mask].mean(axis=0)
            else:
                new_centroids[shard_id] = centroids[shard_id]
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
            "sharding_method": "custom",
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


def create_shard_key(base_url: str, collection: str, shard_key: str) -> None:
    request_json(
        base_url,
        "PUT",
        f"/collections/{urllib.parse.quote(collection, safe='')}/shards",
        body={"shard_key": shard_key, "shards_number": 1, "replication_factor": 1},
    )


def delete_collection_if_exists(base_url: str, collection: str) -> None:
    try:
        request_json(base_url, "DELETE", f"/collections/{urllib.parse.quote(collection, safe='')}")
    except RuntimeError as exc:
        if "404" not in str(exc):
            raise


def upsert_points(base_url: str, collection: str, shard_key: str, ids: list[int], vectors: list[list[float]]) -> None:
    points = [{"id": idx, "vector": vec} for idx, vec in zip(ids, vectors)]
    request_json(
        base_url,
        "PUT",
        f"/collections/{urllib.parse.quote(collection, safe='')}/points?wait=true",
        body={"points": points, "shard_key": shard_key},
        timeout=600.0,
    )


def collection_info(base_url: str, collection: str) -> dict:
    return request_json(base_url, "GET", f"/collections/{urllib.parse.quote(collection, safe='')}")["result"]


def wait_collection_indexed(base_url: str, collection: str, expected_points: int, timeout_sec: float = 7200.0) -> dict:
    start = time.perf_counter()
    while True:
        info = collection_info(base_url, collection)
        if int(info.get("indexed_vectors_count") or 0) >= expected_points and int(info.get("points_count") or 0) == expected_points:
            return info
        if time.perf_counter() - start > timeout_sec:
            raise TimeoutError("Timed out waiting for custom-sharded collection to finish indexing")
        time.sleep(1.0)


def batch_search(base_url: str, collection: str, query_vectors: np.ndarray, neighbors: np.ndarray, top_k: int, hnsw_ef: int, shard_keys: list[list[str]], batch_size: int = 100) -> dict:
    total_hits = 0
    total_queries = len(query_vectors)
    start = time.perf_counter()

    for start_idx in range(0, total_queries, batch_size):
        chunk = query_vectors[start_idx : start_idx + batch_size]
        key_chunk = shard_keys[start_idx : start_idx + batch_size]
        body = {
            "searches": [
                {
                    "vector": q.tolist(),
                    "limit": top_k,
                    "params": {"hnsw_ef": hnsw_ef},
                    "with_payload": False,
                    "with_vector": False,
                    "shard_key": keys,
                }
                for q, keys in zip(chunk, key_chunk)
            ]
        }
        payload = request_json(
            base_url,
            "POST",
            f"/collections/{urllib.parse.quote(collection, safe='')}/points/search/batch",
            body=body,
            timeout=600.0,
        )
        for local_idx, result in enumerate(payload["result"]):
            ids = {int(item["id"]) - 1 for item in result}
            gt = set(map(int, neighbors[start_idx + local_idx]))
            total_hits += len(ids & gt)

    wall = time.perf_counter() - start
    return {
        "query_count": total_queries,
        "recall_at_k": total_hits / (total_queries * top_k),
        "qps": total_queries / wall,
        "wall_s": wall,
    }


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

    scores = train @ centroids.T
    assignments = np.argmax(scores, axis=1)
    shard_keys = [f"centroid_{i:02d}" for i in range(args.num_shards)]

    if not args.reuse_existing:
        delete_collection_if_exists(args.base_url, args.collection)
        create_collection(args.base_url, args.collection, train.shape[1], args.hnsw_m, args.ef_construct)
        for shard_key in shard_keys:
            create_shard_key(args.base_url, args.collection, shard_key)

        for shard_id, shard_key in enumerate(shard_keys):
            point_indices = np.where(assignments == shard_id)[0]
            for start_idx in range(0, len(point_indices), args.upload_batch_size):
                idx_chunk = point_indices[start_idx : start_idx + args.upload_batch_size]
                ids = (idx_chunk + 1).tolist()
                vectors = train[idx_chunk].tolist()
                upsert_points(args.base_url, args.collection, shard_key, ids, vectors)

    info = wait_collection_indexed(args.base_url, args.collection, len(train))
    build_row = {
        "collection": args.collection,
        "points_count": int(info.get("points_count") or 0),
        "indexed_vectors_count": int(info.get("indexed_vectors_count") or 0),
        "segments_count": int(info.get("segments_count") or 0),
    }
    with (output_dir / "builds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(build_row.keys()))
        writer.writeheader()
        writer.writerow(build_row)

    query_scores = queries @ centroids.T
    rows = []
    for nprobe in args.nprobes:
        nprobe = min(nprobe, args.num_shards)
        nearest = np.argpartition(-query_scores, kth=nprobe - 1, axis=1)[:, :nprobe]
        selected_keys = [[shard_keys[int(shard_id)] for shard_id in row] for row in nearest]
        result = batch_search(
            args.base_url,
            args.collection,
            queries,
            neighbors,
            args.top_k,
            args.hnsw_ef,
            selected_keys,
        )
        row = {
            "nprobe": nprobe,
            "hnsw_ef": args.hnsw_ef,
            "query_count": args.query_count,
            "top_k": args.top_k,
            "recall_at_k": result["recall_at_k"],
            "qps": result["qps"],
            "wall_s": result["wall_s"],
            "avg_selected_shards": float(nprobe),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    with (output_dir / "nprobe_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "collection": args.collection,
        "num_shards": args.num_shards,
        "hnsw_m": args.hnsw_m,
        "ef_construct": args.ef_construct,
        "hnsw_ef": args.hnsw_ef,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
