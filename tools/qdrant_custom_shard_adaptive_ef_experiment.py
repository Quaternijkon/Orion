#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
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
            "Build or reuse a KMeans custom-sharded Qdrant collection, then compare "
            "uniform-hnsw_ef search against an application-layer adaptive per-shard hnsw_ef policy."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6336")
    parser.add_argument("--collection", default="qdrant_custom_centroid44_cluster")
    parser.add_argument(
        "--hdf5-path",
        default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5",
    )
    parser.add_argument("--num-shards", type=int, default=44)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--upload-batch-size", type=int, default=1024)
    parser.add_argument("--query-count", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=100)
    parser.add_argument(
        "--uniform-ef-candidates",
        type=int,
        nargs="+",
        default=[48, 64, 80, 96, 128],
    )
    parser.add_argument(
        "--nprobes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 44],
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--target-recall", type=float, default=0.88)
    parser.add_argument("--output-dir", default="results/qdrant_custom_adaptive_ef")
    parser.add_argument("--seed", type=int, default=42)
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


def train_centroids(
    train: np.ndarray,
    k: int,
    sample_size: int,
    iters: int,
    seed: int,
) -> np.ndarray:
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


def wait_collection_indexed(
    base_url: str,
    collection: str,
    expected_points: int,
    timeout_sec: float = 7200.0,
) -> dict:
    start = time.perf_counter()
    while True:
        info = collection_info(base_url, collection)
        if (
            int(info.get("indexed_vectors_count") or 0) >= expected_points
            and int(info.get("points_count") or 0) == expected_points
        ):
            return info
        if time.perf_counter() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for {collection} to finish indexing")
        time.sleep(1.0)


def collection_exists(base_url: str, collection: str) -> bool:
    try:
        collection_info(base_url, collection)
        return True
    except RuntimeError as exc:
        if "404" in str(exc):
            return False
        raise


def build_final_centroids(train: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    final_centroids = np.zeros_like(centroids)
    counts = np.zeros(len(centroids), dtype=np.int64)
    for shard_id in range(len(centroids)):
        mask = assignments == shard_id
        counts[shard_id] = int(mask.sum())
        if counts[shard_id] > 0:
            final_centroids[shard_id] = train[mask].mean(axis=0)
        else:
            final_centroids[shard_id] = centroids[shard_id]
    return normalize_rows(final_centroids), counts


def ranked_shards(query_centroid_scores: np.ndarray, nprobe: int) -> np.ndarray:
    order = np.argsort(-query_centroid_scores, axis=1)
    return order[:, :nprobe]


def ef_schedule_for_nprobe(schedule: list[tuple[int, int]], nprobe: int) -> list[int]:
    values: list[int] = []
    previous_limit = 0
    for limit, ef_value in schedule:
        if limit <= previous_limit:
            raise ValueError("schedule limits must be strictly increasing")
        span = max(0, min(limit, nprobe) - previous_limit)
        values.extend([ef_value] * span)
        previous_limit += span
        if previous_limit >= nprobe:
            break
    if len(values) < nprobe:
        raise ValueError(f"schedule does not cover nprobe={nprobe}")
    return values[:nprobe]


def group_selected_keys_by_ef(keys: list[str], ef_values: list[int]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for key, ef_value in zip(keys, ef_values):
        grouped.setdefault(ef_value, []).append(key)
    return grouped


def merge_topk_candidates(candidate_groups: list[list[tuple[float, int]]], top_k: int) -> list[int]:
    best_by_id: dict[int, float] = {}
    for group in candidate_groups:
        for score, point_id in group:
            current = best_by_id.get(point_id)
            if current is None or score > current:
                best_by_id[point_id] = score
    ordered = sorted(best_by_id.items(), key=lambda item: item[1], reverse=True)
    return [point_id for point_id, _score in ordered[:top_k]]


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


def batch_search_adaptive(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    selected_keys: list[list[str]],
    selected_efs: list[list[int]],
    batch_size: int,
) -> dict[str, float]:
    total_hits = 0
    total_queries = len(queries)
    start = time.perf_counter()

    for start_idx in range(0, total_queries, batch_size):
        chunk = queries[start_idx : start_idx + batch_size]
        key_chunk = selected_keys[start_idx : start_idx + batch_size]
        ef_chunk = selected_efs[start_idx : start_idx + batch_size]

        per_query_candidates: list[list[list[tuple[float, int]]]] = [[] for _ in range(len(chunk))]
        ef_to_searches: dict[int, list[dict[str, Any]]] = defaultdict(list)
        ef_to_query_positions: dict[int, list[int]] = defaultdict(list)

        for local_idx, (keys, ef_values) in enumerate(zip(key_chunk, ef_chunk)):
            grouped = group_selected_keys_by_ef(keys, ef_values)
            for ef_value, shard_keys in grouped.items():
                ef_to_query_positions[ef_value].append(local_idx)
                ef_to_searches[ef_value].append(
                    {
                        "vector": chunk[local_idx].tolist(),
                        "limit": top_k,
                        "params": {"hnsw_ef": ef_value},
                        "with_payload": False,
                        "with_vector": False,
                        "shard_key": shard_keys,
                    }
                )

        for ef_value, searches in ef_to_searches.items():
            results = search_batch(base_url, collection, searches)
            for local_idx, result in zip(ef_to_query_positions[ef_value], results):
                per_query_candidates[local_idx].append(result)

        for local_idx, candidate_groups in enumerate(per_query_candidates):
            top_ids = merge_topk_candidates(candidate_groups, top_k)
            gt = set(map(int, neighbors[start_idx + local_idx]))
            total_hits += len(set(top_ids) & gt)

    wall = time.perf_counter() - start
    return {
        "recall_at_k": total_hits / (total_queries * top_k),
        "qps": total_queries / wall,
        "wall_s": wall,
    }


def default_adaptive_schedule_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "sched_a_4_28_44__128_64_48",
            "schedule": [(4, 128), (28, 64), (44, 48)],
        },
        {
            "name": "sched_b_8_28_44__96_64_48",
            "schedule": [(8, 96), (28, 64), (44, 48)],
        },
        {
            "name": "sched_c_4_20_44__128_64_56",
            "schedule": [(4, 128), (20, 64), (44, 56)],
        },
        {
            "name": "sched_d_8_32_44__96_64_56",
            "schedule": [(8, 96), (32, 64), (44, 56)],
        },
        {
            "name": "sched_e_4_16_32_44__160_80_64_48",
            "schedule": [(4, 160), (16, 80), (32, 64), (44, 48)],
        },
    ]


def choose_candidate(rows: list[dict[str, Any]], target_recall: float, score_key: str = "recall_at_k") -> dict[str, Any]:
    meeting = [row for row in rows if row[score_key] >= target_recall]
    if not meeting:
        best = max(rows, key=lambda row: row[score_key])
        raise RuntimeError(
            f"No candidate reached target recall {target_recall:.4f}; "
            f"best was {best[score_key]:.4f} with {best}"
        )
    return max(meeting, key=lambda row: row["qps"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_collection(
    args: argparse.Namespace,
    train: np.ndarray,
    assignments: np.ndarray,
    shard_keys: list[str],
) -> dict[str, Any]:
    if not args.reuse_existing or not collection_exists(args.base_url, args.collection):
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
    return {
        "collection": args.collection,
        "points_count": int(info.get("points_count") or 0),
        "indexed_vectors_count": int(info.get("indexed_vectors_count") or 0),
        "segments_count": int(info.get("segments_count") or 0),
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
    assignments = np.argmax(train @ centroids.T, axis=1)
    final_centroids, shard_sizes = build_final_centroids(train, assignments, centroids)
    shard_keys = [f"centroid_{i:02d}" for i in range(args.num_shards)]

    build_row = ensure_collection(args, train, assignments, shard_keys)
    write_csv(output_dir / "builds.csv", [build_row])

    shard_rows = [
        {
            "shard_id": shard_id,
            "shard_key": shard_keys[shard_id],
            "points_count": int(size),
        }
        for shard_id, size in enumerate(shard_sizes)
    ]
    write_csv(output_dir / "shard_sizes.csv", shard_rows)

    query_scores = queries @ final_centroids.T

    uniform_tuning_rows: list[dict[str, Any]] = []
    for ef_value in args.uniform_ef_candidates:
        nearest = ranked_shards(query_scores, args.num_shards)
        selected_keys = [[shard_keys[int(shard_id)] for shard_id in row] for row in nearest]
        result = batch_search_uniform(
            args.base_url,
            args.collection,
            queries,
            neighbors,
            args.top_k,
            ef_value,
            selected_keys,
            args.batch_size,
        )
        row = {
            "candidate_type": "uniform_ef",
            "nprobe": args.num_shards,
            "hnsw_ef": ef_value,
            "recall_at_k": result["recall_at_k"],
            "qps": result["qps"],
            "wall_s": result["wall_s"],
        }
        uniform_tuning_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    write_csv(output_dir / "uniform_ef_tuning.csv", uniform_tuning_rows)
    chosen_uniform = choose_candidate(uniform_tuning_rows, args.target_recall)

    uniform_rows: list[dict[str, Any]] = []
    for nprobe in args.nprobes:
        nprobe = min(nprobe, args.num_shards)
        nearest = ranked_shards(query_scores, nprobe)
        selected_keys = [[shard_keys[int(shard_id)] for shard_id in row] for row in nearest]
        result = batch_search_uniform(
            args.base_url,
            args.collection,
            queries,
            neighbors,
            args.top_k,
            int(chosen_uniform["hnsw_ef"]),
            selected_keys,
            args.batch_size,
        )
        row = {
            "strategy": "uniform",
            "nprobe": nprobe,
            "hnsw_ef": int(chosen_uniform["hnsw_ef"]),
            "schedule_name": "",
            "schedule": "",
            "query_count": args.query_count,
            "top_k": args.top_k,
            "recall_at_k": result["recall_at_k"],
            "qps": result["qps"],
            "wall_s": result["wall_s"],
            "avg_selected_shards": float(nprobe),
        }
        uniform_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    write_csv(output_dir / "uniform_ef_nprobe.csv", uniform_rows)

    adaptive_tuning_rows: list[dict[str, Any]] = []
    adaptive_candidates = default_adaptive_schedule_candidates()
    nearest_all = ranked_shards(query_scores, args.num_shards)
    selected_keys_all = [[shard_keys[int(shard_id)] for shard_id in row] for row in nearest_all]
    for candidate in adaptive_candidates:
        ef_list = ef_schedule_for_nprobe(candidate["schedule"], args.num_shards)
        selected_efs = [ef_list[:] for _ in range(args.query_count)]
        result = batch_search_adaptive(
            args.base_url,
            args.collection,
            queries,
            neighbors,
            args.top_k,
            selected_keys_all,
            selected_efs,
            args.batch_size,
        )
        row = {
            "candidate_type": "adaptive_schedule",
            "nprobe": args.num_shards,
            "schedule_name": candidate["name"],
            "schedule": json.dumps(candidate["schedule"]),
            "recall_at_k": result["recall_at_k"],
            "qps": result["qps"],
            "wall_s": result["wall_s"],
        }
        adaptive_tuning_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    write_csv(output_dir / "adaptive_ef_tuning.csv", adaptive_tuning_rows)
    chosen_adaptive = choose_candidate(adaptive_tuning_rows, args.target_recall)
    chosen_schedule = json.loads(chosen_adaptive["schedule"])

    adaptive_rows: list[dict[str, Any]] = []
    for nprobe in args.nprobes:
        nprobe = min(nprobe, args.num_shards)
        nearest = ranked_shards(query_scores, nprobe)
        selected_keys = [[shard_keys[int(shard_id)] for shard_id in row] for row in nearest]
        ef_list = ef_schedule_for_nprobe(chosen_schedule, nprobe)
        selected_efs = [ef_list[:] for _ in range(args.query_count)]
        result = batch_search_adaptive(
            args.base_url,
            args.collection,
            queries,
            neighbors,
            args.top_k,
            selected_keys,
            selected_efs,
            args.batch_size,
        )
        row = {
            "strategy": "adaptive",
            "nprobe": nprobe,
            "hnsw_ef": "",
            "schedule_name": chosen_adaptive["schedule_name"],
            "schedule": json.dumps(chosen_schedule),
            "query_count": args.query_count,
            "top_k": args.top_k,
            "recall_at_k": result["recall_at_k"],
            "qps": result["qps"],
            "wall_s": result["wall_s"],
            "avg_selected_shards": float(nprobe),
        }
        adaptive_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    write_csv(output_dir / "adaptive_ef_nprobe.csv", adaptive_rows)

    summary = {
        "base_url": args.base_url,
        "collection": args.collection,
        "hdf5_path": args.hdf5_path,
        "num_shards": args.num_shards,
        "query_count": args.query_count,
        "top_k": args.top_k,
        "hnsw_m": args.hnsw_m,
        "ef_construct": args.ef_construct,
        "target_recall": args.target_recall,
        "chosen_uniform": chosen_uniform,
        "chosen_adaptive": chosen_adaptive,
        "uniform_tuning_rows": uniform_tuning_rows,
        "adaptive_tuning_rows": adaptive_tuning_rows,
        "uniform_rows": uniform_rows,
        "adaptive_rows": adaptive_rows,
        "build_row": build_row,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
