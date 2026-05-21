#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible HNSW sweep against a live Qdrant instance and measure "
            "build latency, search latency, and recall@k."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:6333",
        help="Qdrant base URL. Default: %(default)s",
    )
    parser.add_argument(
        "--collection-prefix",
        default="hnsw_exp",
        help="Prefix for temporary collection names. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hnsw",
        help="Directory for JSON/CSV outputs. Default: %(default)s",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["synthetic", "hdf5"],
        default="synthetic",
        help="Dataset source. Default: %(default)s",
    )
    parser.add_argument(
        "--hdf5-path",
        default="",
        help="HDF5 dataset path when --dataset-source hdf5 is used.",
    )
    parser.add_argument(
        "--hdf5-train-key",
        default="train",
        help="Train dataset key inside HDF5. Default: %(default)s",
    )
    parser.add_argument(
        "--hdf5-query-key",
        default="test",
        help="Query dataset key inside HDF5. Default: %(default)s",
    )
    parser.add_argument(
        "--hdf5-neighbors-key",
        default="neighbors",
        help="Ground-truth neighbors key inside HDF5. Default: %(default)s",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=5000,
        help=(
            "Number of vectors to insert. For HDF5, use the full train size by passing "
            "the dataset row count explicitly. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=50,
        help=(
            "Number of query vectors to evaluate. For HDF5, use the full test size by "
            "passing the dataset row count explicitly. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--exact-num-queries",
        type=int,
        default=0,
        help=(
            "Number of queries used only for exact-search latency measurement. "
            "0 means: all queries for synthetic data, or min(100, num_queries) for HDF5 data."
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Dense vector dimension. Default: %(default)s",
    )
    parser.add_argument(
        "--distance",
        choices=["Cosine", "Dot", "Euclid", "Manhattan"],
        default="Cosine",
        help="Distance metric for the collection. Default: %(default)s",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=20,
        help="Synthetic cluster count used to generate structured vectors. Default: %(default)s",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for upserts. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic data generation. Default: %(default)s",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K used for exact and approximate search. Default: %(default)s",
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=5,
        help="Warm-up queries per search mode. Default: %(default)s",
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Collection-level HNSW M values to sweep. Default: %(default)s",
    )
    parser.add_argument(
        "--ef-construct-values",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Collection-level HNSW ef_construct values to sweep. Default: %(default)s",
    )
    parser.add_argument(
        "--query-ef-values",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Query-time hnsw_ef values to sweep. Default: %(default)s",
    )
    parser.add_argument(
        "--full-scan-threshold",
        type=int,
        default=10,
        help=(
            "Collection full_scan_threshold in KB. Small values force HNSW on "
            "small datasets. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--indexing-threshold",
        type=int,
        default=10,
        help=(
            "Collection indexing_threshold in KB. Use a value >= 10 to build HNSW. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--max-indexing-threads",
        type=int,
        default=0,
        help="HNSW max_indexing_threads. 0 lets Qdrant choose automatically. Default: %(default)s",
    )
    parser.add_argument(
        "--collection-shard-number",
        type=int,
        default=1,
        help=(
            "Number of Qdrant shards for the collection. Points are distributed across "
            "these shards and searched across all shards unless shard_key is specified. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=180.0,
        help="Timeout for waiting on indexing completion. Default: %(default)s",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=0.5,
        help="Polling interval while waiting for indexing. Default: %(default)s",
    )
    parser.add_argument(
        "--keep-collections",
        action="store_true",
        help="Do not delete experiment collections after the run.",
    )
    return parser.parse_args()


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percent / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = rank - lower
    return lower_value + (upper_value - lower_value) * weight


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector[:]
    return [value / norm for value in vector]


def chunked(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def slugify(text: str) -> str:
    cleaned = []
    for char in text:
        if char.isalnum():
            cleaned.append(char.lower())
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "dataset"


@dataclass
class DatasetBundle:
    name: str
    num_points: int
    num_queries: int
    dim: int
    queries: list[list[float]]
    points: list[dict[str, Any]] | None
    ground_truth_ids: list[list[int | str]] | None
    ground_truth_source: str
    hdf5_path: str | None = None
    hdf5_train_key: str | None = None


@dataclass
class BuildStats:
    collection_name: str
    create_secs: float
    upsert_secs: float
    wait_index_secs: float
    total_build_secs: float
    points_count: int
    indexed_vectors_count: int
    segments_count: int
    ram_data_size: int | None
    disk_data_size: int | None
    optimizer_status: str | None


@dataclass
class SearchStats:
    collection_name: str
    ground_truth_source: str
    evaluated_queries: int
    exact_latency_queries: int
    m: int
    ef_construct: int
    query_ef: int
    full_scan_threshold: int
    indexing_threshold: int
    top_k: int
    exact_latency_mean_ms: float
    exact_latency_p50_ms: float
    exact_latency_p95_ms: float
    approx_latency_mean_ms: float
    approx_latency_p50_ms: float
    approx_latency_p95_ms: float
    approx_server_time_mean_ms: float
    recall_at_k_mean: float
    recall_at_k_stddev: float
    hit_at_1_mean: float
    build_secs: float
    wait_index_secs: float
    indexed_vectors_count: int
    points_count: int
    segments_count: int
    ram_data_size: int | None
    disk_data_size: int | None


class QdrantHttpClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{method} {url} returned non-JSON body: {raw}") from exc

        status = parsed.get("status")
        if status not in (None, "ok"):
            raise RuntimeError(f"{method} {url} returned unexpected status: {parsed}")
        return parsed

    def delete_collection_if_exists(self, collection_name: str) -> None:
        encoded = urllib.parse.quote(collection_name, safe="")
        try:
            self.request_json("DELETE", f"/collections/{encoded}", timeout=30.0)
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise

    def create_collection(
        self,
        collection_name: str,
        dim: int,
        distance: str,
        shard_number: int,
        m: int,
        ef_construct: int,
        full_scan_threshold: int,
        indexing_threshold: int,
        max_indexing_threads: int,
    ) -> None:
        encoded = urllib.parse.quote(collection_name, safe="")
        body = {
            "vectors": {
                "size": dim,
                "distance": distance,
            },
            "shard_number": shard_number,
            "replication_factor": 1,
            "write_consistency_factor": 1,
            "hnsw_config": {
                "m": m,
                "ef_construct": ef_construct,
                "full_scan_threshold": full_scan_threshold,
                "max_indexing_threads": max_indexing_threads,
            },
            "optimizers_config": {
                "default_segment_number": 1,
                "indexing_threshold": indexing_threshold,
            },
        }
        self.request_json("PUT", f"/collections/{encoded}", body=body, timeout=60.0)

    def upsert_points(self, collection_name: str, points: Sequence[dict[str, Any]]) -> None:
        encoded = urllib.parse.quote(collection_name, safe="")
        self.request_json(
            "PUT",
            f"/collections/{encoded}/points?wait=true",
            body={"points": list(points)},
            timeout=120.0,
        )

    def upsert_batch(
        self,
        collection_name: str,
        ids: Sequence[int | str],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        encoded = urllib.parse.quote(collection_name, safe="")
        self.request_json(
            "PUT",
            f"/collections/{encoded}/points?wait=true",
            body={"batch": {"ids": list(ids), "vectors": list(vectors)}},
            timeout=240.0,
        )

    def collection_info(self, collection_name: str) -> dict[str, Any]:
        encoded = urllib.parse.quote(collection_name, safe="")
        return self.request_json("GET", f"/collections/{encoded}", timeout=30.0)["result"]

    def collection_optimizations(self, collection_name: str) -> dict[str, Any]:
        encoded = urllib.parse.quote(collection_name, safe="")
        return self.request_json(
            "GET",
            f"/collections/{encoded}/optimizations?with=queued,completed,idle_segments",
            timeout=30.0,
        )["result"]

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
        params: dict[str, Any] | None = None,
    ) -> tuple[list[int | str], float, float]:
        encoded = urllib.parse.quote(collection_name, safe="")
        body: dict[str, Any] = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": False,
            "with_vector": False,
        }
        if params is not None:
            body["params"] = params
        started = time.perf_counter()
        response = self.request_json(
            "POST",
            f"/collections/{encoded}/points/search",
            body=body,
            timeout=120.0,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        result = response.get("result", [])
        ids = [point["id"] for point in result]
        server_time_ms = float(response.get("time", 0.0)) * 1000.0
        return ids, latency_ms, server_time_ms


def build_synthetic_dataset(
    num_points: int,
    num_queries: int,
    dim: int,
    num_clusters: int,
    seed: int,
    distance: str,
) -> DatasetBundle:
    rng = random.Random(seed)

    centroids: list[list[float]] = []
    for _ in range(num_clusters):
        centroid = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        if distance == "Cosine":
            centroid = normalize(centroid)
        centroids.append(centroid)

    def jitter(base: list[float], noise_stddev: float) -> list[float]:
        vector = [value + rng.gauss(0.0, noise_stddev) for value in base]
        if distance == "Cosine":
            return normalize(vector)
        return vector

    points: list[dict[str, Any]] = []
    for point_id in range(num_points):
        cluster = point_id % num_clusters
        points.append(
            {
                "id": point_id + 1,
                "vector": jitter(centroids[cluster], noise_stddev=0.12),
                "payload": {"cluster": cluster},
            }
        )

    queries: list[list[float]] = []
    for query_id in range(num_queries):
        cluster = query_id % num_clusters
        queries.append(jitter(centroids[cluster], noise_stddev=0.08))

    return DatasetBundle(
        name="synthetic",
        num_points=num_points,
        num_queries=num_queries,
        dim=dim,
        queries=queries,
        points=points,
        ground_truth_ids=None,
        ground_truth_source="qdrant_exact",
    )


def load_hdf5_dataset(
    hdf5_path: str,
    train_key: str,
    query_key: str,
    neighbors_key: str,
    num_points: int,
    num_queries: int,
    top_k: int,
    distance: str,
) -> DatasetBundle:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for --dataset-source hdf5") from exc

    path = Path(hdf5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 dataset does not exist: {path}")

    with h5py.File(path, "r") as handle:
        train = handle[train_key]
        query = handle[query_key]
        neighbors = handle[neighbors_key] if neighbors_key in handle else None

        train_rows, dim = int(train.shape[0]), int(train.shape[1])
        query_rows = int(query.shape[0])

        if num_points <= 0 or num_points > train_rows:
            raise ValueError(
                f"--num-points must be within 1..{train_rows} for this HDF5 dataset."
            )
        if num_queries <= 0 or num_queries > query_rows:
            raise ValueError(
                f"--num-queries must be within 1..{query_rows} for this HDF5 dataset."
            )

        query_matrix = query[:num_queries].astype("float32", copy=True)
        if distance == "Cosine":
            norms = (query_matrix * query_matrix).sum(axis=1) ** 0.5
            norms[norms < 1e-12] = 1.0
            query_matrix = query_matrix / norms[:, None]

        queries = [row.tolist() for row in query_matrix]

        ground_truth_ids: list[list[int | str]] | None = None
        ground_truth_source = "qdrant_exact"
        if neighbors is not None:
            if num_points != train_rows:
                raise ValueError(
                    "HDF5 neighbors are defined against the full train split. "
                    "Use the full train size when relying on HDF5 ground truth."
                )
            ground_truth_ids = [
                [int(point_id) + 1 for point_id in row[:top_k]]
                for row in neighbors[:num_queries]
            ]
            ground_truth_source = "hdf5_neighbors"

    return DatasetBundle(
        name=path.stem,
        num_points=num_points,
        num_queries=num_queries,
        dim=dim,
        queries=queries,
        points=None,
        ground_truth_ids=ground_truth_ids,
        ground_truth_source=ground_truth_source,
        hdf5_path=str(path),
        hdf5_train_key=train_key,
    )


def iter_hdf5_train_batches(
    hdf5_path: str,
    train_key: str,
    num_points: int,
    batch_size: int,
    distance: str,
) -> Iterable[tuple[list[int], list[list[float]]]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for --dataset-source hdf5") from exc

    with h5py.File(hdf5_path, "r") as handle:
        train = handle[train_key]
        for start in range(0, num_points, batch_size):
            stop = min(start + batch_size, num_points)
            vectors = train[start:stop].astype("float32", copy=True)
            if distance == "Cosine":
                norms = (vectors * vectors).sum(axis=1) ** 0.5
                norms[norms < 1e-12] = 1.0
                vectors = vectors / norms[:, None]
            ids = list(range(start + 1, stop + 1))
            yield ids, vectors.tolist()


def load_dataset(args: argparse.Namespace) -> DatasetBundle:
    if args.dataset_source == "synthetic":
        return build_synthetic_dataset(
            num_points=args.num_points,
            num_queries=args.num_queries,
            dim=args.dim,
            num_clusters=args.num_clusters,
            seed=args.seed,
            distance=args.distance,
        )

    if not args.hdf5_path:
        raise ValueError("--hdf5-path is required when --dataset-source hdf5 is used")

    return load_hdf5_dataset(
        hdf5_path=args.hdf5_path,
        train_key=args.hdf5_train_key,
        query_key=args.hdf5_query_key,
        neighbors_key=args.hdf5_neighbors_key,
        num_points=args.num_points,
        num_queries=args.num_queries,
        top_k=args.top_k,
        distance=args.distance,
    )


def upload_dataset(
    client: QdrantHttpClient,
    collection_name: str,
    dataset: DatasetBundle,
    batch_size: int,
    distance: str,
) -> None:
    if dataset.points is not None:
        for batch in chunked(dataset.points, batch_size):
            client.upsert_points(collection_name, batch)
        return

    if not dataset.hdf5_path or not dataset.hdf5_train_key:
        raise RuntimeError("Missing HDF5 metadata for streamed upload.")

    for ids, vectors in iter_hdf5_train_batches(
        hdf5_path=dataset.hdf5_path,
        train_key=dataset.hdf5_train_key,
        num_points=dataset.num_points,
        batch_size=batch_size,
        distance=distance,
    ):
        client.upsert_batch(collection_name, ids, vectors)


def indexing_wait_is_complete(
    points_count: int,
    expected_points: int,
    indexed_vectors_count: int,
    running: Sequence[Any],
    queued: Sequence[Any],
) -> bool:
    # Tiny multi-shard runs can leave residual segments unindexed but still searchable.
    # Once all points arrived and optimizer queues are empty, Qdrant serves the tail by full scan.
    _ = indexed_vectors_count
    return points_count == expected_points and not running and not queued


def wait_until_indexed(
    client: QdrantHttpClient,
    collection_name: str,
    expected_points: int,
    timeout_sec: float,
    poll_interval_sec: float,
) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    last_state: dict[str, Any] | None = None

    while True:
        info = client.collection_info(collection_name)
        optimizations = client.collection_optimizations(collection_name)
        indexed_vectors_count = int(info.get("indexed_vectors_count") or 0)
        points_count = int(info.get("points_count") or 0)
        running = optimizations.get("running", [])
        queued = optimizations.get("queued", [])

        if indexing_wait_is_complete(
            points_count=points_count,
            expected_points=expected_points,
            indexed_vectors_count=indexed_vectors_count,
            running=running,
            queued=queued,
        ):
            elapsed = time.perf_counter() - started
            return elapsed, info

        last_state = {
            "points_count": points_count,
            "indexed_vectors_count": indexed_vectors_count,
            "running": running,
            "queued": queued,
        }

        if time.perf_counter() - started > timeout_sec:
            raise TimeoutError(
                f"Timed out waiting for indexing on {collection_name}. Last state: {last_state}"
            )
        time.sleep(poll_interval_sec)


def collect_exact_baseline(
    client: QdrantHttpClient,
    collection_name: str,
    queries: Sequence[list[float]],
    top_k: int,
    warmup_queries: int,
) -> tuple[list[list[int | str]], list[float]]:
    warmup_count = min(warmup_queries, len(queries))
    for query in queries[:warmup_count]:
        client.search(collection_name, query, top_k, params={"exact": True})

    exact_ids: list[list[int | str]] = []
    exact_latencies_ms: list[float] = []
    for query in queries:
        ids, latency_ms, _server_time_ms = client.search(
            collection_name,
            query,
            top_k,
            params={"exact": True},
        )
        exact_ids.append(ids)
        exact_latencies_ms.append(latency_ms)
    return exact_ids, exact_latencies_ms


def measure_exact_latency(
    client: QdrantHttpClient,
    collection_name: str,
    queries: Sequence[list[float]],
    top_k: int,
    warmup_queries: int,
) -> list[float]:
    warmup_count = min(warmup_queries, len(queries))
    for query in queries[:warmup_count]:
        client.search(collection_name, query, top_k, params={"exact": True})

    exact_latencies_ms: list[float] = []
    for query in queries:
        _ids, latency_ms, _server_time_ms = client.search(
            collection_name,
            query,
            top_k,
            params={"exact": True},
        )
        exact_latencies_ms.append(latency_ms)
    return exact_latencies_ms


def evaluate_query_ef(
    client: QdrantHttpClient,
    collection_name: str,
    queries: Sequence[list[float]],
    reference_ids: Sequence[Sequence[int | str]],
    top_k: int,
    query_ef: int,
    warmup_queries: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    warmup_count = min(warmup_queries, len(queries))
    for query in queries[:warmup_count]:
        client.search(collection_name, query, top_k, params={"hnsw_ef": query_ef})

    approx_latencies_ms: list[float] = []
    approx_server_times_ms: list[float] = []
    recalls: list[float] = []
    hits_at_1: list[float] = []

    for query, baseline in zip(queries, reference_ids):
        approx_ids, latency_ms, server_time_ms = client.search(
            collection_name,
            query,
            top_k,
            params={"hnsw_ef": query_ef},
        )
        approx_latencies_ms.append(latency_ms)
        approx_server_times_ms.append(server_time_ms)

        baseline_set = set(baseline)
        approx_set = set(approx_ids)
        denominator = max(1, min(top_k, len(baseline)))
        recall = len(baseline_set & approx_set) / denominator
        recalls.append(recall)

        baseline_top1 = baseline[0] if baseline else None
        approx_top1 = approx_ids[0] if approx_ids else None
        hits_at_1.append(1.0 if baseline_top1 == approx_top1 else 0.0)

    return approx_latencies_ms, approx_server_times_ms, recalls, hits_at_1


def create_build_stats(
    collection_name: str,
    create_secs: float,
    upsert_secs: float,
    wait_index_secs: float,
    info: dict[str, Any],
) -> BuildStats:
    return BuildStats(
        collection_name=collection_name,
        create_secs=create_secs,
        upsert_secs=upsert_secs,
        wait_index_secs=wait_index_secs,
        total_build_secs=create_secs + upsert_secs + wait_index_secs,
        points_count=int(info.get("points_count") or 0),
        indexed_vectors_count=int(info.get("indexed_vectors_count") or 0),
        segments_count=int(info.get("segments_count") or 0),
        ram_data_size=info.get("ram_data_size"),
        disk_data_size=info.get("disk_data_size"),
        optimizer_status=info.get("optimizer_status"),
    )


def render_search_stats(
    build_stats: BuildStats,
    dataset: DatasetBundle,
    exact_latency_queries: int,
    m: int,
    ef_construct: int,
    query_ef: int,
    full_scan_threshold: int,
    indexing_threshold: int,
    top_k: int,
    exact_latencies_ms: Sequence[float],
    approx_latencies_ms: Sequence[float],
    approx_server_times_ms: Sequence[float],
    recalls: Sequence[float],
    hits_at_1: Sequence[float],
) -> SearchStats:
    return SearchStats(
        collection_name=build_stats.collection_name,
        ground_truth_source=dataset.ground_truth_source,
        evaluated_queries=dataset.num_queries,
        exact_latency_queries=exact_latency_queries,
        m=m,
        ef_construct=ef_construct,
        query_ef=query_ef,
        full_scan_threshold=full_scan_threshold,
        indexing_threshold=indexing_threshold,
        top_k=top_k,
        exact_latency_mean_ms=mean(exact_latencies_ms),
        exact_latency_p50_ms=percentile(exact_latencies_ms, 50),
        exact_latency_p95_ms=percentile(exact_latencies_ms, 95),
        approx_latency_mean_ms=mean(approx_latencies_ms),
        approx_latency_p50_ms=percentile(approx_latencies_ms, 50),
        approx_latency_p95_ms=percentile(approx_latencies_ms, 95),
        approx_server_time_mean_ms=mean(approx_server_times_ms),
        recall_at_k_mean=mean(recalls),
        recall_at_k_stddev=statistics.pstdev(recalls) if len(recalls) > 1 else 0.0,
        hit_at_1_mean=mean(hits_at_1),
        build_secs=build_stats.total_build_secs,
        wait_index_secs=build_stats.wait_index_secs,
        indexed_vectors_count=build_stats.indexed_vectors_count,
        points_count=build_stats.points_count,
        segments_count=build_stats.segments_count,
        ram_data_size=build_stats.ram_data_size,
        disk_data_size=build_stats.disk_data_size,
    )


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    dataset: DatasetBundle,
    build_rows: Sequence[BuildStats],
    search_rows: Sequence[SearchStats],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(search_rows[0]).keys()))
        writer.writeheader()
        for row in search_rows:
            writer.writerow(asdict(row))

    build_path = output_dir / "builds.csv"
    with build_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(build_rows[0]).keys()))
        writer.writeheader()
        for row in build_rows:
            writer.writerow(asdict(row))

    details = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "dataset": {
            "name": dataset.name,
            "num_points": dataset.num_points,
            "num_queries": dataset.num_queries,
            "dim": dataset.dim,
            "ground_truth_source": dataset.ground_truth_source,
            "hdf5_path": dataset.hdf5_path,
            "hdf5_train_key": dataset.hdf5_train_key,
        },
        "builds": [asdict(row) for row in build_rows],
        "search_results": [asdict(row) for row in search_rows],
    }
    details_path = output_dir / "details.json"
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    print("")
    print(f"Wrote CSV summary to: {summary_path}")
    print(f"Wrote build summary to: {build_path}")
    print(f"Wrote full JSON details to: {details_path}")


def resolve_exact_latency_queries(args: argparse.Namespace, dataset: DatasetBundle) -> int:
    if args.exact_num_queries > 0:
        return min(args.exact_num_queries, dataset.num_queries)
    if dataset.ground_truth_ids is not None:
        return min(100, dataset.num_queries)
    return dataset.num_queries


def main() -> int:
    args = parse_args()
    if args.dataset_source == "synthetic" and args.num_clusters <= 0:
        raise ValueError("--num-clusters must be greater than 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")
    if args.collection_shard_number <= 0:
        raise ValueError("--collection-shard-number must be greater than 0")
    if args.full_scan_threshold < 10:
        raise ValueError("--full-scan-threshold must be 10 or greater for this Qdrant version.")
    if args.indexing_threshold < 10:
        raise ValueError(
            "--indexing-threshold must be 10 or greater for this Qdrant version."
        )

    dataset = load_dataset(args)
    if dataset.dim != args.dim and args.dataset_source == "hdf5":
        print(
            f"[note] Overriding --dim={args.dim} with dataset dim={dataset.dim} from {dataset.name}",
            flush=True,
        )
    args.dim = dataset.dim

    exact_latency_queries = resolve_exact_latency_queries(args, dataset)
    exact_latency_probe_queries = dataset.queries[:exact_latency_queries]

    print(
        f"Dataset: {dataset.name} | source={args.dataset_source} | "
        f"points={dataset.num_points} | queries={dataset.num_queries} | dim={dataset.dim} | "
        f"ground_truth={dataset.ground_truth_source}",
        flush=True,
    )

    client = QdrantHttpClient(args.base_url)

    build_rows: list[BuildStats] = []
    search_rows: list[SearchStats] = []

    total_collections = len(args.m_values) * len(args.ef_construct_values)
    collection_index = 0
    dataset_slug = slugify(dataset.name)

    for m in args.m_values:
        for ef_construct in args.ef_construct_values:
            collection_index += 1
            collection_name = (
                f"{args.collection_prefix}_{dataset_slug}_d{dataset.dim}_n{dataset.num_points}_"
                f"s{args.collection_shard_number}_m{m}_efc{ef_construct}_fs{args.full_scan_threshold}"
            )

            print(
                f"[{collection_index}/{total_collections}] Building {collection_name} "
                f"(m={m}, ef_construct={ef_construct})",
                flush=True,
            )

            client.delete_collection_if_exists(collection_name)

            create_started = time.perf_counter()
            client.create_collection(
                collection_name=collection_name,
                dim=dataset.dim,
                distance=args.distance,
                shard_number=args.collection_shard_number,
                m=m,
                ef_construct=ef_construct,
                full_scan_threshold=args.full_scan_threshold,
                indexing_threshold=args.indexing_threshold,
                max_indexing_threads=args.max_indexing_threads,
            )
            create_secs = time.perf_counter() - create_started

            upsert_started = time.perf_counter()
            upload_dataset(
                client=client,
                collection_name=collection_name,
                dataset=dataset,
                batch_size=args.batch_size,
                distance=args.distance,
            )
            upsert_secs = time.perf_counter() - upsert_started

            wait_index_secs, info = wait_until_indexed(
                client=client,
                collection_name=collection_name,
                expected_points=dataset.num_points,
                timeout_sec=args.timeout_sec,
                poll_interval_sec=args.poll_interval_sec,
            )

            build_stats = create_build_stats(
                collection_name=collection_name,
                create_secs=create_secs,
                upsert_secs=upsert_secs,
                wait_index_secs=wait_index_secs,
                info=info,
            )
            build_rows.append(build_stats)

            print(
                f"  indexed_vectors={build_stats.indexed_vectors_count} "
                f"segments={build_stats.segments_count} "
                f"build={build_stats.total_build_secs:.2f}s",
                flush=True,
            )

            if dataset.ground_truth_ids is None:
                reference_ids, exact_latencies_ms = collect_exact_baseline(
                    client=client,
                    collection_name=collection_name,
                    queries=dataset.queries,
                    top_k=args.top_k,
                    warmup_queries=args.warmup_queries,
                )
            else:
                reference_ids = dataset.ground_truth_ids
                exact_latencies_ms = measure_exact_latency(
                    client=client,
                    collection_name=collection_name,
                    queries=exact_latency_probe_queries,
                    top_k=args.top_k,
                    warmup_queries=args.warmup_queries,
                )

            print(
                f"  exact latency sample mean={mean(exact_latencies_ms):.2f} ms "
                f"p95={percentile(exact_latencies_ms, 95):.2f} ms "
                f"(queries={len(exact_latencies_ms)})",
                flush=True,
            )

            for query_ef in args.query_ef_values:
                (
                    approx_latencies_ms,
                    approx_server_times_ms,
                    recalls,
                    hits_at_1,
                ) = evaluate_query_ef(
                    client=client,
                    collection_name=collection_name,
                    queries=dataset.queries,
                    reference_ids=reference_ids,
                    top_k=args.top_k,
                    query_ef=query_ef,
                    warmup_queries=args.warmup_queries,
                )

                search_stats = render_search_stats(
                    build_stats=build_stats,
                    dataset=dataset,
                    exact_latency_queries=len(exact_latencies_ms),
                    m=m,
                    ef_construct=ef_construct,
                    query_ef=query_ef,
                    full_scan_threshold=args.full_scan_threshold,
                    indexing_threshold=args.indexing_threshold,
                    top_k=args.top_k,
                    exact_latencies_ms=exact_latencies_ms,
                    approx_latencies_ms=approx_latencies_ms,
                    approx_server_times_ms=approx_server_times_ms,
                    recalls=recalls,
                    hits_at_1=hits_at_1,
                )
                search_rows.append(search_stats)

                print(
                    f"  query_ef={query_ef:<4} "
                    f"recall@{args.top_k}={search_stats.recall_at_k_mean:.4f} "
                    f"hit@1={search_stats.hit_at_1_mean:.4f} "
                    f"latency_p50={search_stats.approx_latency_p50_ms:.2f} ms "
                    f"latency_p95={search_stats.approx_latency_p95_ms:.2f} ms",
                    flush=True,
                )

            if not args.keep_collections:
                client.delete_collection_if_exists(collection_name)

    if not build_rows or not search_rows:
        raise RuntimeError("No experiment rows were generated.")

    run_output_dir = Path(args.output_dir) / timestamp_slug()
    write_outputs(run_output_dir, args, dataset, build_rows, search_rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
