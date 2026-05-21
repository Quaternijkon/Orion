#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare shard partition schemes inside one Qdrant custom-sharding framework "
            "under matched-recall full-search and selective-search settings."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6336")
    parser.add_argument("--collection-prefix", default="partcmp44")
    parser.add_argument(
        "--hdf5-path",
        default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5",
    )
    parser.add_argument("--num-shards", type=int, default=44)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--upload-batch-size", type=int, default=1024)
    parser.add_argument("--tuning-query-count", type=int, default=2000)
    parser.add_argument("--eval-query-count", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=100)
    parser.add_argument(
        "--ef-candidates",
        type=int,
        nargs="+",
        default=[24, 32, 40, 48, 64, 80, 96],
    )
    parser.add_argument(
        "--nprobe-candidates",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 44],
    )
    parser.add_argument("--target-recall", type=float, default=0.88)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--stability-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/qdrant_partition_scheme")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def request_json(
    base_url: str,
    method: str,
    path: str,
    body: dict | None = None,
    timeout: float = 300.0,
) -> dict:
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


def stable_hash_bucket(point_id: int, num_shards: int) -> int:
    digest = hashlib.blake2b(str(point_id).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little", signed=False)
    return value % num_shards


def compute_rankings_and_priorities(
    train: np.ndarray,
    centroids: np.ndarray,
    batch_size: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    order_parts: list[np.ndarray] = []
    priority_parts: list[np.ndarray] = []
    for start_idx in range(0, len(train), batch_size):
        batch = train[start_idx : start_idx + batch_size]
        scores = batch @ centroids.T
        order = np.argsort(-scores, axis=1).astype(np.uint8, copy=False)
        top1 = scores[np.arange(len(batch)), order[:, 0]]
        top2 = scores[np.arange(len(batch)), order[:, 1]]
        priority = (top1 - top2).astype(np.float32, copy=False)
        order_parts.append(order)
        priority_parts.append(priority)
    return np.vstack(order_parts), np.concatenate(priority_parts)


def balanced_assign_from_rankings(
    order: np.ndarray,
    priorities: np.ndarray,
    capacities: list[int],
) -> np.ndarray:
    remaining = capacities[:]
    assignments = np.full(order.shape[0], -1, dtype=np.int32)
    processing_order = np.argsort(-priorities)
    for point_idx in processing_order:
        for shard_id in order[point_idx]:
            shard_id_int = int(shard_id)
            if remaining[shard_id_int] > 0:
                assignments[point_idx] = shard_id_int
                remaining[shard_id_int] -= 1
                break
        if assignments[point_idx] < 0:
            raise RuntimeError("balanced assignment failed to place a point")
    return assignments


def capacities_for_points(num_points: int, num_shards: int) -> list[int]:
    base = num_points // num_shards
    extra = num_points % num_shards
    return [base + (1 if shard_id < extra else 0) for shard_id in range(num_shards)]


def build_scheme_assignments(
    scheme: str,
    train: np.ndarray,
    num_shards: int,
    sample_size: int,
    kmeans_iters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if scheme == "hash":
        assignments = np.array(
            [stable_hash_bucket(point_id + 1, num_shards) for point_id in range(len(train))],
            dtype=np.int32,
        )
        centroids = np.zeros((num_shards, train.shape[1]), dtype=np.float32)
        return assignments, centroids

    centroids = train_centroids(train, num_shards, sample_size, kmeans_iters, seed)
    if scheme == "kmeans":
        assignments = np.argmax(train @ centroids.T, axis=1).astype(np.int32, copy=False)
        return assignments, centroids

    if scheme == "balanced_kmeans":
        order, priorities = compute_rankings_and_priorities(train, centroids)
        capacities = capacities_for_points(len(train), num_shards)
        assignments = balanced_assign_from_rankings(order, priorities, capacities)
        return assignments, centroids

    raise ValueError(f"unknown scheme: {scheme}")


def build_final_centroids(
    train: np.ndarray,
    assignments: np.ndarray,
    num_shards: int,
    fallback_centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    final_centroids = np.zeros((num_shards, train.shape[1]), dtype=np.float32)
    counts = np.zeros(num_shards, dtype=np.int64)
    for shard_id in range(num_shards):
        mask = assignments == shard_id
        counts[shard_id] = int(mask.sum())
        if counts[shard_id] > 0:
            final_centroids[shard_id] = train[mask].mean(axis=0)
        elif fallback_centroids is not None:
            final_centroids[shard_id] = fallback_centroids[shard_id]
    return normalize_rows(final_centroids), counts


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


def upsert_points(
    base_url: str,
    collection: str,
    shard_key: str,
    ids: list[int],
    vectors: list[list[float]],
) -> None:
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


def collection_exists(base_url: str, collection: str) -> bool:
    try:
        collection_info(base_url, collection)
        return True
    except RuntimeError as exc:
        message = str(exc)
        if "404" in message or "Not found:" in message or "doesn't exist" in message:
            return False
        raise


def wait_collection_indexed(
    base_url: str,
    collection: str,
    expected_points: int,
    timeout_sec: float = 7200.0,
) -> dict:
    start = time.perf_counter()
    last_indexed: int | None = None
    last_change = start
    while True:
        info = collection_info(base_url, collection)
        indexed = int(info.get("indexed_vectors_count") or 0)
        points = int(info.get("points_count") or 0)
        if indexed != last_indexed:
            last_indexed = indexed
            last_change = time.perf_counter()
        if (
            indexed >= expected_points
            and points == expected_points
        ):
            return info
        # In practice, custom-sharded collections can leave a tiny plateau of
        # non-indexed tail points in small trailing segments. If the indexed
        # count stops changing for a while after all points arrived, proceed.
        if points == expected_points and time.perf_counter() - last_change >= 30.0:
            return info
        if time.perf_counter() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for {collection} to finish indexing")
        time.sleep(1.0)


def ranked_shards(query_centroid_scores: np.ndarray, nprobe: int) -> np.ndarray:
    order = np.argsort(-query_centroid_scores, axis=1)
    return order[:, :nprobe]


def search_batch(
    base_url: str,
    collection: str,
    searches: list[dict[str, Any]],
    timeout: float = 600.0,
) -> list[list[tuple[float, int]]]:
    payload = request_json(
        base_url,
        "POST",
        f"/collections/{urllib.parse.quote(collection, safe='')}/points/search/batch",
        body={"searches": searches},
        timeout=timeout,
    )
    return [
        [(float(item["score"]), int(item["id"]) - 1) for item in per_query]
        for per_query in payload["result"]
    ]


def batch_search_uniform(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    hnsw_ef: int,
    selected_keys: list[list[str]],
    batch_size: int,
) -> dict[str, float]:
    total_hits = 0
    total_queries = len(queries)
    start = time.perf_counter()

    for start_idx in range(0, total_queries, batch_size):
        chunk = queries[start_idx : start_idx + batch_size]
        key_chunk = selected_keys[start_idx : start_idx + batch_size]
        searches = [
            {
                "vector": query.tolist(),
                "limit": top_k,
                "params": {"hnsw_ef": hnsw_ef},
                "with_payload": False,
                "with_vector": False,
                "shard_key": keys,
            }
            for query, keys in zip(chunk, key_chunk)
        ]
        results = search_batch(base_url, collection, searches)
        for local_idx, result in enumerate(results):
            ids = {point_id for _score, point_id in result}
            gt = set(map(int, neighbors[start_idx + local_idx]))
            total_hits += len(ids & gt)

    wall = time.perf_counter() - start
    return {
        "recall_at_k": total_hits / (total_queries * top_k),
        "qps": total_queries / wall,
        "wall_s": wall,
    }


def choose_best_matched_recall(rows: list[dict[str, Any]], target_recall: float) -> dict[str, Any]:
    valid = [row for row in rows if row["recall_at_k"] >= target_recall]
    if not valid:
        best = max(rows, key=lambda row: row["recall_at_k"])
        raise RuntimeError(
            f"No candidate met target recall {target_recall:.4f}. "
            f"Best recall was {best['recall_at_k']:.4f} for {best}"
        )
    return max(valid, key=lambda row: row["qps"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_collection(
    base_url: str,
    collection: str,
    train: np.ndarray,
    assignments: np.ndarray,
    shard_keys: list[str],
    hnsw_m: int,
    ef_construct: int,
    upload_batch_size: int,
    reuse_existing: bool,
) -> dict[str, Any]:
    if not reuse_existing or not collection_exists(base_url, collection):
        delete_collection_if_exists(base_url, collection)
        create_collection(base_url, collection, train.shape[1], hnsw_m, ef_construct)
        for shard_key in shard_keys:
            create_shard_key(base_url, collection, shard_key)
        for shard_id, shard_key in enumerate(shard_keys):
            point_indices = np.where(assignments == shard_id)[0]
            for start_idx in range(0, len(point_indices), upload_batch_size):
                idx_chunk = point_indices[start_idx : start_idx + upload_batch_size]
                ids = (idx_chunk + 1).tolist()
                vectors = train[idx_chunk].tolist()
                upsert_points(base_url, collection, shard_key, ids, vectors)

    info = wait_collection_indexed(base_url, collection, len(train))
    return {
        "collection": collection,
        "points_count": int(info.get("points_count") or 0),
        "indexed_vectors_count": int(info.get("indexed_vectors_count") or 0),
        "segments_count": int(info.get("segments_count") or 0),
    }


def evaluate_full(
    base_url: str,
    collection: str,
    shard_keys: list[str],
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    ef_value: int,
    batch_size: int,
) -> dict[str, float]:
    selected_keys = [shard_keys[:] for _ in range(len(queries))]
    return batch_search_uniform(
        base_url,
        collection,
        queries,
        neighbors,
        top_k,
        ef_value,
        selected_keys,
        batch_size,
    )


def evaluate_selective(
    base_url: str,
    collection: str,
    shard_keys: list[str],
    shard_centroids: np.ndarray,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    ef_value: int,
    nprobe: int,
    batch_size: int,
) -> dict[str, float]:
    nearest = ranked_shards(queries @ shard_centroids.T, nprobe)
    selected_keys = [[shard_keys[int(shard_id)] for shard_id in row] for row in nearest]
    return batch_search_uniform(
        base_url,
        collection,
        queries,
        neighbors,
        top_k,
        ef_value,
        selected_keys,
        batch_size,
    )


def partition_summary_rows(
    scheme: str,
    shard_keys: list[str],
    shard_sizes: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mean_size = float(np.mean(shard_sizes))
    std_size = float(np.std(shard_sizes))
    cv = (std_size / mean_size) if mean_size > 0 else 0.0
    rows = [
        {
            "scheme": scheme,
            "shard_id": shard_id,
            "shard_key": shard_keys[shard_id],
            "points_count": int(shard_sizes[shard_id]),
        }
        for shard_id in range(len(shard_keys))
    ]
    summary = {
        "scheme": scheme,
        "mean_points": mean_size,
        "std_points": std_size,
        "cv": cv,
        "min_points": int(np.min(shard_sizes)),
        "max_points": int(np.max(shard_sizes)),
    }
    return rows, summary


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5_path, "r") as handle:
        train = handle["train"][:].astype(np.float32, copy=True)
        tuning_queries = handle["test"][: args.tuning_query_count].astype(np.float32, copy=True)
        tuning_neighbors = handle["neighbors"][: args.tuning_query_count, : args.top_k].astype(np.int32, copy=True)
        eval_queries = handle["test"][: args.eval_query_count].astype(np.float32, copy=True)
        eval_neighbors = handle["neighbors"][: args.eval_query_count, : args.top_k].astype(np.int32, copy=True)

    train = normalize_rows(train)
    tuning_queries = normalize_rows(tuning_queries)
    eval_queries = normalize_rows(eval_queries)

    build_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    partition_summary: list[dict[str, Any]] = []
    full_tuning_rows: list[dict[str, Any]] = []
    selective_tuning_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

    schemes = ["hash", "kmeans", "balanced_kmeans"]
    for scheme in schemes:
        print(f"Preparing scheme: {scheme}", flush=True)
        assignments, seed_centroids = build_scheme_assignments(
            scheme,
            train,
            args.num_shards,
            args.sample_size,
            args.kmeans_iters,
            args.seed,
        )
        shard_centroids, shard_sizes = build_final_centroids(
            train,
            assignments,
            args.num_shards,
            fallback_centroids=seed_centroids if len(seed_centroids) else None,
        )
        shard_keys = [f"{scheme}_{i:02d}" for i in range(args.num_shards)]
        collection = f"{args.collection_prefix}_{scheme}_d{train.shape[1]}_n{len(train)}_s{args.num_shards}_m{args.hnsw_m}_efc{args.ef_construct}"

        build_row = ensure_collection(
            args.base_url,
            collection,
            train,
            assignments,
            shard_keys,
            args.hnsw_m,
            args.ef_construct,
            args.upload_batch_size,
            args.reuse_existing,
        )
        build_row["scheme"] = scheme
        build_rows.append(build_row)

        shard_rows, shard_summary = partition_summary_rows(scheme, shard_keys, shard_sizes)
        partition_rows.extend(shard_rows)
        partition_summary.append(shard_summary)

        for ef_value in args.ef_candidates:
            result = evaluate_full(
                args.base_url,
                collection,
                shard_keys,
                tuning_queries,
                tuning_neighbors,
                args.top_k,
                ef_value,
                args.batch_size,
            )
            row = {
                "scheme": scheme,
                "scenario": "full",
                "nprobe": args.num_shards,
                "hnsw_ef": ef_value,
                "query_count": args.tuning_query_count,
                "recall_at_k": result["recall_at_k"],
                "qps": result["qps"],
                "wall_s": result["wall_s"],
            }
            full_tuning_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

        for nprobe in args.nprobe_candidates:
            nprobe = min(nprobe, args.num_shards)
            for ef_value in args.ef_candidates:
                result = evaluate_selective(
                    args.base_url,
                    collection,
                    shard_keys,
                    shard_centroids,
                    tuning_queries,
                    tuning_neighbors,
                    args.top_k,
                    ef_value,
                    nprobe,
                    args.batch_size,
                )
                row = {
                    "scheme": scheme,
                    "scenario": "selective",
                    "nprobe": nprobe,
                    "hnsw_ef": ef_value,
                    "query_count": args.tuning_query_count,
                    "recall_at_k": result["recall_at_k"],
                    "qps": result["qps"],
                    "wall_s": result["wall_s"],
                }
                selective_tuning_rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

        scheme_full_rows = [row for row in full_tuning_rows if row["scheme"] == scheme]
        best_full = choose_best_matched_recall(scheme_full_rows, args.target_recall)
        best_full_eval = evaluate_full(
            args.base_url,
            collection,
            shard_keys,
            eval_queries,
            eval_neighbors,
            args.top_k,
            int(best_full["hnsw_ef"]),
            args.batch_size,
        )
        best_rows.append(
            {
                "scheme": scheme,
                "scenario": "full",
                "nprobe": args.num_shards,
                "hnsw_ef": int(best_full["hnsw_ef"]),
                "query_count": args.eval_query_count,
                "recall_at_k": best_full_eval["recall_at_k"],
                "qps": best_full_eval["qps"],
                "wall_s": best_full_eval["wall_s"],
            }
        )

        scheme_selective_rows = [row for row in selective_tuning_rows if row["scheme"] == scheme]
        best_selective = choose_best_matched_recall(scheme_selective_rows, args.target_recall)
        best_selective_eval = evaluate_selective(
            args.base_url,
            collection,
            shard_keys,
            shard_centroids,
            eval_queries,
            eval_neighbors,
            args.top_k,
            int(best_selective["hnsw_ef"]),
            int(best_selective["nprobe"]),
            args.batch_size,
        )
        best_rows.append(
            {
                "scheme": scheme,
                "scenario": "selective",
                "nprobe": int(best_selective["nprobe"]),
                "hnsw_ef": int(best_selective["hnsw_ef"]),
                "query_count": args.eval_query_count,
                "recall_at_k": best_selective_eval["recall_at_k"],
                "qps": best_selective_eval["qps"],
                "wall_s": best_selective_eval["wall_s"],
            }
        )

        # Warm up once for both best points before stability runs.
        _ = evaluate_full(
            args.base_url,
            collection,
            shard_keys,
            eval_queries,
            eval_neighbors,
            args.top_k,
            int(best_full["hnsw_ef"]),
            args.batch_size,
        )
        _ = evaluate_selective(
            args.base_url,
            collection,
            shard_keys,
            shard_centroids,
            eval_queries,
            eval_neighbors,
            args.top_k,
            int(best_selective["hnsw_ef"]),
            int(best_selective["nprobe"]),
            args.batch_size,
        )

        for run_idx in range(1, args.stability_repeats + 1):
            result = evaluate_full(
                args.base_url,
                collection,
                shard_keys,
                eval_queries,
                eval_neighbors,
                args.top_k,
                int(best_full["hnsw_ef"]),
                args.batch_size,
            )
            stability_rows.append(
                {
                    "scheme": scheme,
                    "scenario": "full",
                    "run": run_idx,
                    "nprobe": args.num_shards,
                    "hnsw_ef": int(best_full["hnsw_ef"]),
                    "query_count": args.eval_query_count,
                    "recall_at_k": result["recall_at_k"],
                    "qps": result["qps"],
                    "wall_s": result["wall_s"],
                }
            )

        for run_idx in range(1, args.stability_repeats + 1):
            result = evaluate_selective(
                args.base_url,
                collection,
                shard_keys,
                shard_centroids,
                eval_queries,
                eval_neighbors,
                args.top_k,
                int(best_selective["hnsw_ef"]),
                int(best_selective["nprobe"]),
                args.batch_size,
            )
            stability_rows.append(
                {
                    "scheme": scheme,
                    "scenario": "selective",
                    "run": run_idx,
                    "nprobe": int(best_selective["nprobe"]),
                    "hnsw_ef": int(best_selective["hnsw_ef"]),
                    "query_count": args.eval_query_count,
                    "recall_at_k": result["recall_at_k"],
                    "qps": result["qps"],
                    "wall_s": result["wall_s"],
                }
            )

    write_csv(output_dir / "builds.csv", build_rows)
    write_csv(output_dir / "partition_stats.csv", partition_rows)
    write_csv(output_dir / "partition_summary.csv", partition_summary)
    write_csv(output_dir / "full_tuning.csv", full_tuning_rows)
    write_csv(output_dir / "selective_tuning.csv", selective_tuning_rows)
    write_csv(output_dir / "best_points.csv", best_rows)
    write_csv(output_dir / "stability_runs.csv", stability_rows)

    stability_summary: list[dict[str, Any]] = []
    for scheme in schemes:
        for scenario in ("full", "selective"):
            rows = [
                row
                for row in stability_rows
                if row["scheme"] == scheme and row["scenario"] == scenario
            ]
            qps_values = [row["qps"] for row in rows]
            recall_values = [row["recall_at_k"] for row in rows]
            stability_summary.append(
                {
                    "scheme": scheme,
                    "scenario": scenario,
                    "runs": len(rows),
                    "qps_mean": statistics.mean(qps_values),
                    "qps_stdev": statistics.stdev(qps_values) if len(qps_values) > 1 else 0.0,
                    "qps_min": min(qps_values),
                    "qps_max": max(qps_values),
                    "recall_mean": statistics.mean(recall_values),
                    "recall_stdev": statistics.stdev(recall_values) if len(recall_values) > 1 else 0.0,
                }
            )

    summary = {
        "base_url": args.base_url,
        "hdf5_path": args.hdf5_path,
        "num_shards": args.num_shards,
        "tuning_query_count": args.tuning_query_count,
        "eval_query_count": args.eval_query_count,
        "top_k": args.top_k,
        "hnsw_m": args.hnsw_m,
        "ef_construct": args.ef_construct,
        "target_recall": args.target_recall,
        "ef_candidates": args.ef_candidates,
        "nprobe_candidates": args.nprobe_candidates,
        "best_rows": best_rows,
        "partition_summary": partition_summary,
        "stability_summary": stability_summary,
        "note": (
            "Compare these controlled custom-sharding results against the existing auto-hash "
            "reference under results/hnsw/20260415_120959 and results/hash44_stability/20260415_121142 "
            "to separate architecture effects from partition effects."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
