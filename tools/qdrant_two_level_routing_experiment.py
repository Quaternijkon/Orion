#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from math import ceil
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Qdrant two-level routing experiment. The default mode mirrors the original "
            "C++ idea construction/routing path using the patched Qdrant MultiEP search path."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6336")
    parser.add_argument("--collection", default="qdrant_original_idea_cluster")
    parser.add_argument(
        "--hdf5-path",
        default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5",
    )
    parser.add_argument(
        "--routing-mode",
        choices=["faithful_original_rest", "legacy_centroid", "naive_hash_all_shards"],
        default="faithful_original_rest",
    )
    parser.add_argument("--num-shards", type=int, default=31)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--upload-batch-size", type=int, default=1024)
    parser.add_argument("--tuning-query-count", type=int, default=1000)
    parser.add_argument("--eval-query-count", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=100)
    parser.add_argument(
        "--upper-k-candidates",
        type=int,
        nargs="+",
        default=[100],
    )
    parser.add_argument(
        "--base-ef-candidates",
        type=int,
        nargs="+",
        default=[20],
    )
    parser.add_argument(
        "--factor-candidates",
        type=int,
        nargs="+",
        default=[4],
    )
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--k-overlap", type=int, default=10)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--kmeans-rand-seed", type=int, default=1)
    parser.add_argument("--topology-iters", type=int, default=50)
    parser.add_argument("--upper-build-batch-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--target-recall", type=float, default=0.88)
    parser.add_argument("--stability-repeats", type=int, default=3)
    parser.add_argument(
        "--concurrency-candidates",
        type=int,
        nargs="+",
        default=[],
        help=(
            "Optional client-side concurrent benchmark with the requested worker counts."
        ),
    )
    parser.add_argument(
        "--concurrency-evaluation-mode",
        choices=["preplanned_search", "end_to_end"],
        default="preplanned_search",
        help=(
            "preplanned_search precomputes query routing/search plans before timing and "
            "isolates Qdrant search pressure. end_to_end includes per-batch upper routing "
            "and search-plan construction in the timed QPS path."
        ),
    )
    parser.add_argument(
        "--search-dispatch-mode",
        choices=["coordinator", "direct_peer"],
        default="coordinator",
        help=(
            "coordinator sends batch search requests to --base-url and lets Qdrant fan out. "
            "direct_peer splits each batch by shard owner and sends peer-local searches "
            "directly to the provided bottom-node HTTP URLs."
        ),
    )
    parser.add_argument(
        "--direct-peer-http-urls",
        nargs="+",
        default=None,
        help="Peer HTTP endpoints for direct_peer mode, as PEER_ID=URL entries.",
    )
    parser.add_argument(
        "--routed-execution-mode",
        choices=["grouped_by_ef", "compact_query_ef", "per_shard_multi_ep", "compact_multi_ep"],
        default="grouped_by_ef",
        help=(
            "How routed idea searches are sent to Qdrant. grouped_by_ef preserves "
            "per-shard dynamic EF grouped by equal EF. compact_query_ef sends one "
            "search object per query with all routed shards and one compact EF. "
            "per_shard_multi_ep sends one search object per routed shard with the "
            "original upper-tier point IDs as HNSW entry points. compact_multi_ep "
            "sends one search object per query while preserving per-shard entry "
            "points and dynamic EF in shard-key maps."
        ),
    )
    parser.add_argument(
        "--compact-ef-mode",
        choices=["max", "mean_ceil"],
        default="max",
        help="How to reduce per-shard dynamic EF values to one query-level EF in compact_query_ef mode.",
    )
    parser.add_argument(
        "--routed-result-limit-mode",
        choices=["top_k", "per_shard_top_k", "fixed_multiplier"],
        default="top_k",
        help=(
            "Result limit for routed searches. top_k asks Qdrant for final top-k only. "
            "per_shard_top_k asks for top_k times the number of selected shards, "
            "preserving the wider candidate pool of separate shard searches. "
            "fixed_multiplier asks for top_k times --routed-result-limit-multiplier."
        ),
    )
    parser.add_argument(
        "--routed-result-limit-multiplier",
        type=float,
        default=1,
        help="Candidate multiplier used when --routed-result-limit-mode=fixed_multiplier.",
    )
    parser.add_argument(
        "--routed-planning-mode",
        choices=["per_batch", "materialized"],
        default="per_batch",
        help=(
            "per_batch preserves the original benchmark implementation and builds route "
            "plans inside each search batch. materialized computes upper routing and "
            "route plans once for the evaluation set inside the timed path, then reuses "
            "the same method4 plans for batched lower-tier searches. This keeps method4 "
            "semantics and timing coverage while reducing Python batch-loop overhead."
        ),
    )
    parser.add_argument(
        "--source-id-dedup-block-size",
        type=int,
        default=None,
        help=(
            "Experimental server-side merge de-dup block size for copied point IDs "
            "encoded as shard_id * block_size + source_id + 1. Defaults to "
            "train_size + 1 when copied source_id payloads are used."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/qdrant_two_level")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--recover-routing-from-collection",
        action="store_true",
        help=(
            "For faithful_original_rest with --reuse-existing, recover upper-point shard "
            "membership from the already-built Qdrant collection instead of recomputing "
            "the full original partitioning pipeline. This is for evaluation-only reruns."
        ),
    )
    parser.add_argument("--disable-multi-assign", action="store_true")
    parser.add_argument("--disable-fission", action="store_true")
    parser.add_argument("--search-all-shards", action="store_true")
    parser.add_argument(
        "--shard-placement",
        choices=["auto", "round_robin", "none"],
        default="auto",
        help=(
            "Physical placement for custom shard keys. Does not change the "
            "KMeans/routing algorithm; it only selects which Qdrant peer owns "
            "each custom shard in cluster deployments."
        ),
    )
    parser.add_argument(
        "--placement-peer-uri-contains",
        nargs="+",
        default=None,
        help=(
            "Optional URI substrings used to choose which Qdrant peers may own custom "
            "shards. In controller+shards topologies, pass the shard-node hostname "
            "prefix, for example: --placement-peer-uri-contains qdrant_shard_."
        ),
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help=(
            "Use only the first N training vectors for deployment smoke tests. "
            "Leave unset for benchmark runs."
        ),
    )
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


def slice_train_rows(rows: Any, train_limit: int | None) -> Any:
    if train_limit is None:
        return rows
    if train_limit <= 0:
        raise ValueError("train_limit must be positive")
    return rows[:train_limit]


def cluster_peer_ids(base_url: str, uri_filters: list[str] | None = None) -> list[int]:
    try:
        result = request_json(base_url, "GET", "/cluster")["result"]
    except RuntimeError:
        return []

    peer_ids: set[int] = set()
    if result.get("peer_id") is not None and not uri_filters:
        peer_ids.add(int(result["peer_id"]))
    for peer_id, peer_info in (result.get("peers") or {}).items():
        uri = str((peer_info or {}).get("uri") or "")
        if uri_filters and not any(uri_filter in uri for uri_filter in uri_filters):
            continue
        peer_ids.add(int(peer_id))
    return sorted(peer_ids)


def parse_peer_http_urls(items: list[str] | None) -> dict[int, str]:
    peer_urls: dict[int, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("expected PEER_ID=URL for --direct-peer-http-urls")
        peer_id_text, url = item.split("=", 1)
        try:
            peer_id = int(peer_id_text)
        except ValueError as exc:
            raise ValueError("expected PEER_ID=URL for --direct-peer-http-urls") from exc
        if not url:
            raise ValueError("expected PEER_ID=URL for --direct-peer-http-urls")
        peer_urls[peer_id] = url.rstrip("/")
    return peer_urls


def placement_for_shard_key(
    shard_index: int,
    peer_ids: list[int],
    mode: str,
) -> list[int] | None:
    if mode == "none":
        return None
    if mode == "auto" and len(peer_ids) <= 1:
        return None
    if mode not in {"auto", "round_robin"}:
        raise ValueError(f"unsupported shard placement mode: {mode}")
    if not peer_ids:
        return None
    return [int(peer_ids[shard_index % len(peer_ids)])]


def shard_create_body(
    shard_key: str,
    placement: list[int] | None = None,
    shards_number: int = 1,
    replication_factor: int = 1,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "shard_key": shard_key,
        "shards_number": shards_number,
        "replication_factor": replication_factor,
    }
    if placement is not None:
        body["placement"] = placement
    return body


def shard_key_for_id(shard_id: int) -> str:
    return f"centroid_{shard_id:02d}"


class OriginalRoutingState:
    def __init__(
        self,
        initial_num_shards: int,
        num_shards: int,
        upper_indices: np.ndarray,
        point_to_l1s: list[list[int]],
        l1_to_shard: list[int],
        primary_shards: np.ndarray,
        point_to_shards: list[list[int]],
        shard_counts: np.ndarray,
        total_assigned: int,
        expansion_ratio: float,
        topology_iterations: int,
        fission_events: list[dict[str, Any]],
    ) -> None:
        self.initial_num_shards = initial_num_shards
        self.num_shards = num_shards
        self.upper_indices = upper_indices
        self.point_to_l1s = point_to_l1s
        self.l1_to_shard = l1_to_shard
        self.primary_shards = primary_shards
        self.point_to_shards = point_to_shards
        self.shard_counts = shard_counts
        self.total_assigned = total_assigned
        self.expansion_ratio = expansion_ratio
        self.topology_iterations = topology_iterations
        self.fission_events = fission_events


def global_upper_indices(num_points: int, denominator: int, seed: int) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    subset_size = num_points // denominator
    if subset_size <= 0:
        raise ValueError("num_points // denominator must be positive")
    rng = np.random.default_rng(seed)
    return rng.permutation(num_points)[:subset_size].astype(np.int64, copy=False)


def squared_l2_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    point_norms = np.sum(points * points, axis=1, keepdims=True)
    centroid_norms = np.sum(centroids * centroids, axis=1, keepdims=True).T
    distances = point_norms + centroid_norms - 2.0 * (points @ centroids.T)
    return np.maximum(distances, 0.0)


def cpp_style_kmeans_train(
    data: np.ndarray,
    indices: np.ndarray | list[int],
    k: int,
    max_iter: int = 10,
    seed: int = 1,
) -> np.ndarray:
    indices_arr = np.asarray(indices, dtype=np.int64)
    if k <= 0:
        raise ValueError("k must be positive")
    if len(indices_arr) < k:
        raise ValueError("k must not exceed number of indexed points")

    rng = np.random.default_rng(seed)
    points = data[indices_arr].astype(np.float32, copy=False)
    centroids = np.zeros((k, data.shape[1]), dtype=np.float32)

    first_pos = int(rng.integers(0, len(indices_arr)))
    centroids[0] = points[first_pos]
    min_dists = np.full(len(indices_arr), np.finfo(np.float32).max, dtype=np.float32)

    for centroid_id in range(1, k):
        dists = np.sum((points - centroids[centroid_id - 1]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        total_dist = float(np.sum(min_dists))
        if total_dist <= 0.0:
            next_pos = 0
        else:
            threshold = float(rng.random()) * total_dist
            next_pos = int(np.searchsorted(np.cumsum(min_dists), threshold, side="left"))
            if next_pos >= len(indices_arr):
                next_pos = len(indices_arr) - 1
        centroids[centroid_id] = points[next_pos]

    for _ in range(max_iter):
        assignments = np.argmin(squared_l2_distances(points, centroids), axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(assignments, minlength=k)
        for centroid_id in range(k):
            if counts[centroid_id] > 0:
                new_centroids[centroid_id] = points[assignments == centroid_id].mean(axis=0)
            else:
                new_centroids[centroid_id] = centroids[centroid_id]
        centroids = new_centroids
    return centroids


def cpp_style_predict_balanced(
    data: np.ndarray,
    indices: np.ndarray | list[int],
    point_weights: np.ndarray | list[int],
    centroids: np.ndarray,
) -> np.ndarray:
    indices_arr = np.asarray(indices, dtype=np.int64)
    weights = np.asarray(point_weights, dtype=np.int64)
    k = len(centroids)
    if len(indices_arr) != len(weights):
        raise ValueError("indices and point_weights must have the same length")
    if k <= 0:
        raise ValueError("centroids must not be empty")

    distances = squared_l2_distances(data[indices_arr].astype(np.float32, copy=False), centroids)
    assignments = np.full(len(indices_arr), -1, dtype=np.int32)
    cluster_weight = np.zeros(k, dtype=np.int64)
    target_weight = int(np.sum(weights) // k + 1)
    max_cluster_weight = target_weight * 1.5

    for flat_pos in np.argsort(distances.ravel(), kind="stable"):
        point_pos = int(flat_pos // k)
        cluster_id = int(flat_pos % k)
        if assignments[point_pos] != -1:
            continue
        if cluster_weight[cluster_id] + weights[point_pos] <= max_cluster_weight:
            assignments[point_pos] = cluster_id
            cluster_weight[cluster_id] += weights[point_pos]

    for point_pos in range(len(assignments)):
        if assignments[point_pos] == -1:
            cluster_id = int(np.argmin(cluster_weight))
            assignments[point_pos] = cluster_id
            cluster_weight[cluster_id] += weights[point_pos]
    return assignments


def compute_point_to_l1s(
    upper_index: Any,
    train: np.ndarray,
    k_overlap: int,
    batch_size: int,
) -> list[list[int]]:
    if k_overlap <= 0:
        raise ValueError("k_overlap must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    labels_rows: list[list[int]] = []
    k = min(k_overlap, upper_index.get_current_count())
    for start in range(0, len(train), batch_size):
        labels, _distances = upper_index.knn_query(train[start : start + batch_size], k=k)
        labels_rows.extend([[int(label) for label in row] for row in labels.tolist()])
    return labels_rows


def initial_l1_shards_by_balanced_kmeans(
    train: np.ndarray,
    upper_indices: np.ndarray,
    up_tier_weights: np.ndarray,
    num_shards: int,
    kmeans_iters: int,
    seed: int,
) -> list[int]:
    centroids = cpp_style_kmeans_train(
        train,
        upper_indices,
        num_shards,
        max_iter=kmeans_iters,
        seed=seed,
    )
    assignments = cpp_style_predict_balanced(train, upper_indices, up_tier_weights, centroids)
    l1_to_shard = [-1] * len(train)
    for point_id, shard_id in zip(upper_indices.tolist(), assignments.tolist()):
        l1_to_shard[int(point_id)] = int(shard_id)
    return l1_to_shard


def converge_l1_topology(
    point_to_l1s: list[list[int]],
    upper_indices: np.ndarray,
    up_tier_weights: np.ndarray,
    l1_to_shard: list[int],
    num_points: int,
    num_shards: int,
    max_iters: int,
) -> tuple[list[int], int]:
    current_shard = list(l1_to_shard)
    ideal_l0_weight = float(num_points) / float(num_shards)
    min_allowed_l1_size = ideal_l0_weight * 0.2
    iteration_count = 0

    for iteration in range(max_iters):
        current_sizes = np.zeros(num_shards, dtype=np.int64)
        for local_idx, l1_idx in enumerate(upper_indices.tolist()):
            shard_id = current_shard[int(l1_idx)]
            if shard_id != -1:
                current_sizes[shard_id] += int(up_tier_weights[local_idx])

        next_shard = list(current_shard)
        changed_count = 0
        for l1_idx in upper_indices.tolist():
            l1_idx = int(l1_idx)
            source_shard = current_shard[l1_idx]
            if source_shard != -1 and current_sizes[source_shard] < min_allowed_l1_size:
                continue

            votes: Counter[int] = Counter()
            for ep_l1 in point_to_l1s[l1_idx]:
                shard_id = current_shard[int(ep_l1)]
                if shard_id != -1:
                    votes[shard_id] += 1
            if not votes:
                continue

            max_vote = max(votes.values())
            best_shards = [shard_id for shard_id in sorted(votes) if votes[shard_id] == max_vote]
            target_shard = source_shard
            if source_shard not in best_shards:
                target_shard = best_shards[0]

            if target_shard != source_shard:
                next_shard[l1_idx] = target_shard
                changed_count += 1

        current_shard = next_shard
        iteration_count = iteration + 1
        if changed_count == 0:
            break

    return current_shard, iteration_count


def target_shards_from_votes(
    l1s: list[int],
    reference_l1_shard: list[int],
    point_index: int,
    num_shards: int,
    use_multi_assign: bool,
) -> list[int]:
    votes: Counter[int] = Counter()
    for ep_l1 in l1s:
        shard_id = reference_l1_shard[int(ep_l1)] if int(ep_l1) < len(reference_l1_shard) else -1
        if shard_id != -1:
            votes[int(shard_id)] += 1

    target_shards: list[int] = []
    if votes:
        max_vote = max(votes.values())
        if use_multi_assign and max_vote >= 2:
            target_shards = [shard_id for shard_id in sorted(votes) if votes[shard_id] == max_vote]

    if not target_shards:
        for ep_l1 in l1s:
            shard_id = reference_l1_shard[int(ep_l1)] if int(ep_l1) < len(reference_l1_shard) else -1
            if shard_id != -1:
                target_shards.append(int(shard_id))
                break
        if not target_shards:
            target_shards.append(int(point_index % num_shards))
    return target_shards


def recalibrate_l1_weights_by_voting(
    point_to_l1s: list[list[int]],
    upper_indices: np.ndarray,
    l1_to_shard: list[int],
    num_shards: int,
    use_multi_assign: bool,
) -> np.ndarray:
    l1_global_to_local = {int(point_id): local_idx for local_idx, point_id in enumerate(upper_indices.tolist())}
    weights = np.zeros(len(upper_indices), dtype=np.int64)
    for point_index, l1s in enumerate(point_to_l1s):
        target_shards = target_shards_from_votes(
            l1s,
            l1_to_shard,
            point_index,
            num_shards,
            use_multi_assign,
        )
        for target_shard in target_shards:
            for ep_l1 in l1s:
                if l1_to_shard[int(ep_l1)] == target_shard:
                    local_idx = l1_global_to_local.get(int(ep_l1))
                    if local_idx is not None:
                        weights[local_idx] += 1
                    break
    return weights


def apply_fission_simulator(
    train: np.ndarray,
    upper_indices: np.ndarray,
    up_tier_weights: np.ndarray,
    l1_to_shard: list[int],
    initial_num_shards: int,
    kmeans_iters: int,
    seed: int,
) -> tuple[list[int], int, list[dict[str, Any]]]:
    ideal_l0_weight = float(len(train)) / float(initial_num_shards)
    max_allowed_weight = ideal_l0_weight * 1.5
    fission_min_allowed = ideal_l0_weight * 0.3
    new_num_shards = initial_num_shards
    post_fission_shard = list(l1_to_shard)
    events: list[dict[str, Any]] = []

    for shard_id in range(initial_num_shards):
        shard_l1_nodes: list[int] = []
        shard_l1_weights: list[int] = []
        expected_weight = 0.0
        for local_idx, l1_idx in enumerate(upper_indices.tolist()):
            if l1_to_shard[int(l1_idx)] == shard_id:
                weight = int(up_tier_weights[local_idx])
                expected_weight += weight
                shard_l1_nodes.append(int(l1_idx))
                shard_l1_weights.append(weight)

        if expected_weight <= max_allowed_weight or len(shard_l1_nodes) <= 1:
            continue

        initial_split_k = int(ceil(expected_weight / ideal_l0_weight))
        split_accepted = False
        best_assignments: np.ndarray | None = None
        accepted_split_k = 0
        for split_k in range(initial_split_k, 1, -1):
            if len(shard_l1_nodes) < split_k:
                continue
            centroids = cpp_style_kmeans_train(
                train,
                shard_l1_nodes,
                split_k,
                max_iter=kmeans_iters,
                seed=seed,
            )
            assignments = cpp_style_predict_balanced(train, shard_l1_nodes, shard_l1_weights, centroids)
            sub_weights = np.zeros(split_k, dtype=np.float64)
            for assignment, weight in zip(assignments.tolist(), shard_l1_weights):
                sub_weights[int(assignment)] += float(weight)
            if not np.any(sub_weights < fission_min_allowed):
                split_accepted = True
                best_assignments = assignments
                accepted_split_k = split_k
                break

        event = {
            "source_shard": shard_id,
            "expected_weight": int(expected_weight),
            "accepted": split_accepted,
            "split_k": accepted_split_k,
        }
        events.append(event)
        if split_accepted and best_assignments is not None:
            for l1_idx, sub_cluster in zip(shard_l1_nodes, best_assignments.tolist()):
                if int(sub_cluster) == 0:
                    post_fission_shard[l1_idx] = shard_id
                else:
                    post_fission_shard[l1_idx] = new_num_shards + int(sub_cluster) - 1
            new_num_shards += accepted_split_k - 1

    return post_fission_shard, new_num_shards, events


def assign_points_by_l1_vote(
    point_to_l1s: list[list[int]],
    reference_l1_shard: list[int],
    num_shards: int,
    use_multi_assign: bool,
) -> tuple[np.ndarray, list[list[int]]]:
    primary_shards = np.full(len(point_to_l1s), -1, dtype=np.int32)
    point_to_shards: list[list[int]] = []
    for point_index, l1s in enumerate(point_to_l1s):
        target_shards = target_shards_from_votes(
            l1s,
            reference_l1_shard,
            point_index,
            num_shards,
            use_multi_assign,
        )
        primary_shards[point_index] = int(target_shards[0])
        point_to_shards.append([int(shard_id) for shard_id in target_shards])
    return primary_shards, point_to_shards


def total_assigned_points(point_to_shards: list[list[int]]) -> int:
    return sum(len(shards) for shards in point_to_shards)


def point_indices_by_shard(
    point_to_shards: list[list[int]],
    num_shards: int,
    upper_indices: np.ndarray | None = None,
) -> list[np.ndarray]:
    by_shard: list[list[int]] = [[] for _ in range(num_shards)]
    if upper_indices is None:
        ordered_points = list(range(len(point_to_shards)))
    else:
        is_l1 = np.zeros(len(point_to_shards), dtype=bool)
        is_l1[np.asarray(upper_indices, dtype=np.int64)] = True
        ordered_points = [int(point_id) for point_id in upper_indices.tolist()]
        ordered_points.extend(int(point_id) for point_id in np.flatnonzero(~is_l1).tolist())

    for point_id in ordered_points:
        for shard_id in point_to_shards[point_id]:
            if 0 <= int(shard_id) < num_shards:
                by_shard[int(shard_id)].append(point_id)
    return [np.asarray(indices, dtype=np.int64) for indices in by_shard]


def hash_point_to_shards(num_points: int, num_shards: int) -> list[list[int]]:
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    return [[point_id % num_shards] for point_id in range(num_points)]


def all_shard_keys_and_ef(num_shards: int, hnsw_ef: int) -> tuple[list[str], list[int]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if hnsw_ef <= 0:
        raise ValueError("hnsw_ef must be positive")
    return [shard_key_for_id(shard_id) for shard_id in range(num_shards)], [hnsw_ef] * num_shards


def build_original_routing_state(
    train: np.ndarray,
    upper_indices: np.ndarray,
    point_to_l1s: list[list[int]],
    initial_num_shards: int,
    kmeans_iters: int,
    kmeans_seed: int,
    topology_iters: int,
    use_multi_assign: bool,
    enable_fission: bool,
) -> OriginalRoutingState:
    nearest_l1 = np.asarray([l1s[0] for l1s in point_to_l1s], dtype=np.int64)
    l1_weights_map = np.bincount(nearest_l1, minlength=len(train))
    up_tier_weights = l1_weights_map[upper_indices].astype(np.int64, copy=False)

    l1_to_shard = initial_l1_shards_by_balanced_kmeans(
        train,
        upper_indices,
        up_tier_weights,
        initial_num_shards,
        kmeans_iters,
        kmeans_seed,
    )
    l1_to_shard, topology_iteration_count = converge_l1_topology(
        point_to_l1s,
        upper_indices,
        up_tier_weights,
        l1_to_shard,
        len(train),
        initial_num_shards,
        topology_iters,
    )

    recalibrated_weights = recalibrate_l1_weights_by_voting(
        point_to_l1s,
        upper_indices,
        l1_to_shard,
        initial_num_shards,
        use_multi_assign,
    )

    fission_events: list[dict[str, Any]] = []
    num_shards = initial_num_shards
    if enable_fission:
        l1_to_shard, num_shards, fission_events = apply_fission_simulator(
            train,
            upper_indices,
            recalibrated_weights,
            l1_to_shard,
            initial_num_shards,
            kmeans_iters,
            kmeans_seed,
        )

    primary_shards, point_to_shards = assign_points_by_l1_vote(
        point_to_l1s,
        l1_to_shard,
        num_shards,
        use_multi_assign,
    )
    shard_counts = np.zeros(num_shards, dtype=np.int64)
    for shards in point_to_shards:
        for shard_id in shards:
            shard_counts[int(shard_id)] += 1

    total_assigned = int(np.sum(shard_counts))
    return OriginalRoutingState(
        initial_num_shards=initial_num_shards,
        num_shards=num_shards,
        upper_indices=upper_indices,
        point_to_l1s=point_to_l1s,
        l1_to_shard=l1_to_shard,
        primary_shards=primary_shards,
        point_to_shards=point_to_shards,
        shard_counts=shard_counts,
        total_assigned=total_assigned,
        expansion_ratio=float(total_assigned / len(train)),
        topology_iterations=topology_iteration_count,
        fission_events=fission_events,
    )


def route_upper_labels_to_shard_eps(
    labels: list[int] | np.ndarray,
    point_to_shards: list[list[int]],
) -> dict[int, list[int]]:
    routed: dict[int, list[int]] = defaultdict(list)
    for label in labels:
        point_id = int(label)
        if 0 <= point_id < len(point_to_shards):
            for shard_id in point_to_shards[point_id]:
                routed[int(shard_id)].append(point_id)
    return dict(sorted(routed.items()))


def shard_efs_from_routed_eps(
    shard_to_eps: dict[int, list[int]],
    num_shards: int,
    base_ef: int,
    factor: int,
    search_all_shards: bool = False,
) -> tuple[list[str], list[int]]:
    if search_all_shards:
        shard_ids = list(range(num_shards))
    else:
        shard_ids = sorted(shard_to_eps)
    shard_keys = [shard_key_for_id(shard_id) for shard_id in shard_ids]
    ef_values = [base_ef + factor * len(shard_to_eps.get(shard_id, [])) for shard_id in shard_ids]
    return shard_keys, ef_values


def original_upper_sample_rows(routing: OriginalRoutingState) -> list[dict[str, Any]]:
    l1_counts = np.zeros(routing.num_shards, dtype=np.int64)
    for l1_idx in routing.upper_indices.tolist():
        shard_id = routing.l1_to_shard[int(l1_idx)]
        if 0 <= shard_id < routing.num_shards:
            l1_counts[shard_id] += 1
    return [
        {
            "shard_id": shard_id,
            "shard_key": shard_key_for_id(shard_id),
            "points_count": int(routing.shard_counts[shard_id]),
            "sample_count": int(l1_counts[shard_id]),
        }
        for shard_id in range(routing.num_shards)
    ]


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


def build_assignments_and_centroids(
    train: np.ndarray,
    num_shards: int,
    sample_size: int,
    kmeans_iters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    centroids = train_centroids(train, num_shards, sample_size, kmeans_iters, seed)
    assignments = np.argmax(train @ centroids.T, axis=1).astype(np.int32, copy=False)
    final_centroids = np.zeros_like(centroids)
    for shard_id in range(num_shards):
        mask = assignments == shard_id
        if np.any(mask):
            final_centroids[shard_id] = train[mask].mean(axis=0)
        else:
            final_centroids[shard_id] = centroids[shard_id]
    return assignments, normalize_rows(final_centroids)


def compute_sample_sizes(shard_sizes: list[int], denominator: int) -> list[int]:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sample_sizes = []
    for size in shard_sizes:
        if size <= 0:
            sample_sizes.append(0)
        else:
            sample_sizes.append(max(1, size // denominator))
    return sample_sizes


def sample_upper_points(
    train: np.ndarray,
    assignments: np.ndarray,
    num_shards: int,
    denominator: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    sampled_vectors: list[np.ndarray] = []
    sampled_ids: list[int] = []
    label_to_shard: dict[int, str] = {}
    rows: list[dict[str, Any]] = []

    shard_indices_list = [np.where(assignments == shard_id)[0] for shard_id in range(num_shards)]
    sample_sizes = compute_sample_sizes([len(indices) for indices in shard_indices_list], denominator)

    for shard_id, (indices, sample_size) in enumerate(zip(shard_indices_list, sample_sizes)):
        shard_key = shard_key_for_id(shard_id)
        if len(indices) == 0 or sample_size == 0:
            rows.append(
                {
                    "shard_id": shard_id,
                    "shard_key": shard_key,
                    "points_count": int(len(indices)),
                    "sample_count": 0,
                }
            )
            continue
        sampled_local = rng.choice(indices, size=sample_size, replace=False)
        sampled_vectors.append(train[sampled_local])
        point_ids = (sampled_local + 1).astype(np.int64)
        sampled_ids.extend(point_ids.tolist())
        for point_id in point_ids.tolist():
            label_to_shard[point_id] = shard_key
        rows.append(
            {
                "shard_id": shard_id,
                "shard_key": shard_key,
                "points_count": int(len(indices)),
                "sample_count": int(sample_size),
            }
        )

    merged_vectors = np.vstack(sampled_vectors).astype(np.float32, copy=False)
    merged_ids = np.array(sampled_ids, dtype=np.int64)
    return merged_vectors, merged_ids, label_to_shard, rows


def build_upper_index(
    vectors: np.ndarray,
    labels: np.ndarray,
    dim: int,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> Any:
    import hnswlib

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(labels), ef_construction=ef_construction, M=m)
    index.add_items(vectors, labels)
    index.set_ef(ef_search)
    return index


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


def create_shard_key(
    base_url: str,
    collection: str,
    shard_key: str,
    placement: list[int] | None = None,
) -> None:
    request_json(
        base_url,
        "PUT",
        f"/collections/{urllib.parse.quote(collection, safe='')}/shards",
        body=shard_create_body(shard_key, placement=placement),
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
    source_ids: list[int] | None = None,
) -> None:
    points = []
    for offset, (idx, vec) in enumerate(zip(ids, vectors)):
        point: dict[str, Any] = {"id": idx, "vector": vec}
        if source_ids is not None:
            point["payload"] = {"source_id": int(source_ids[offset])}
        points.append(point)
    request_json(
        base_url,
        "PUT",
        f"/collections/{urllib.parse.quote(collection, safe='')}/points?wait=true",
        body={"points": points, "shard_key": shard_key},
        timeout=600.0,
    )


def collection_info(base_url: str, collection: str) -> dict:
    return request_json(base_url, "GET", f"/collections/{urllib.parse.quote(collection, safe='')}")["result"]


def collection_cluster_info(base_url: str, collection: str) -> dict | None:
    try:
        return request_json(
            base_url,
            "GET",
            f"/collections/{urllib.parse.quote(collection, safe='')}/cluster",
        )["result"]
    except RuntimeError:
        return None


def collection_shard_key_to_peer(base_url: str, collection: str) -> dict[str, int]:
    cluster_info = collection_cluster_info(base_url, collection)
    if not cluster_info:
        return {}

    mapping: dict[str, int] = {}
    for shard in (cluster_info.get("local_shards") or []) + (cluster_info.get("remote_shards") or []):
        shard_key = shard.get("shard_key")
        peer_id = shard.get("peer_id", cluster_info.get("peer_id"))
        if shard_key is not None and peer_id is not None:
            mapping[str(shard_key)] = int(peer_id)
    return mapping


def collection_cluster_summary(cluster_info: dict | None) -> dict[str, Any]:
    if not cluster_info:
        return {
            "cluster_shard_count": 0,
            "cluster_peer_count": 0,
            "cluster_local_shards": 0,
            "cluster_remote_shards": 0,
            "cluster_active_shards": 0,
        }

    local_shards = cluster_info.get("local_shards") or []
    remote_shards = cluster_info.get("remote_shards") or []
    peer_ids: set[int] = set()
    if cluster_info.get("peer_id") is not None:
        peer_ids.add(int(cluster_info["peer_id"]))
    for shard in remote_shards:
        if shard.get("peer_id") is not None:
            peer_ids.add(int(shard["peer_id"]))

    active = sum(1 for shard in local_shards + remote_shards if shard.get("state") == "Active")
    return {
        "cluster_shard_count": int(cluster_info.get("shard_count") or 0),
        "cluster_peer_count": len(peer_ids),
        "cluster_local_shards": len(local_shards),
        "cluster_remote_shards": len(remote_shards),
        "cluster_active_shards": active,
    }


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
        if indexed >= expected_points and points == expected_points:
            return info
        if points == expected_points and time.perf_counter() - last_change >= 30.0:
            return info
        if time.perf_counter() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for {collection} to finish indexing")
        time.sleep(1.0)


def ensure_collection(
    base_url: str,
    collection: str,
    train: np.ndarray,
    assignments: np.ndarray,
    num_shards: int,
    hnsw_m: int,
    ef_construct: int,
    upload_batch_size: int,
    reuse_existing: bool,
    shard_placement: str,
    peer_ids: list[int],
) -> dict[str, Any]:
    shard_keys = [shard_key_for_id(shard_id) for shard_id in range(num_shards)]
    if not reuse_existing or not collection_exists(base_url, collection):
        delete_collection_if_exists(base_url, collection)
        create_collection(base_url, collection, train.shape[1], hnsw_m, ef_construct)
        for shard_id, shard_key in enumerate(shard_keys):
            placement = placement_for_shard_key(shard_id, peer_ids, shard_placement)
            create_shard_key(base_url, collection, shard_key, placement=placement)
        for shard_id, shard_key in enumerate(shard_keys):
            point_indices = np.where(assignments == shard_id)[0]
            for start_idx in range(0, len(point_indices), upload_batch_size):
                idx_chunk = point_indices[start_idx : start_idx + upload_batch_size]
                ids = (idx_chunk + 1).tolist()
                vectors = train[idx_chunk].tolist()
                upsert_points(base_url, collection, shard_key, ids, vectors)

    info = wait_collection_indexed(base_url, collection, len(train))
    cluster_summary = collection_cluster_summary(collection_cluster_info(base_url, collection))
    return {
        "collection": collection,
        "points_count": int(info.get("points_count") or 0),
        "indexed_vectors_count": int(info.get("indexed_vectors_count") or 0),
        "segments_count": int(info.get("segments_count") or 0),
        "shard_placement": shard_placement,
        "discovered_peer_count": len(peer_ids),
        **cluster_summary,
    }


def encode_copy_id(point_index: int, shard_id: int, num_points: int) -> int:
    return int(shard_id) * (int(num_points) + 1) + int(point_index) + 1


def decode_copy_id(copy_id: int, num_points: int) -> tuple[int, int]:
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if copy_id <= 0:
        raise ValueError("copy_id must be positive")
    block_size = int(num_points) + 1
    zero_based = int(copy_id) - 1
    shard_id = zero_based // block_size
    point_index = zero_based % block_size
    if point_index >= num_points:
        raise ValueError(f"copy_id {copy_id} does not encode a valid source point")
    return int(point_index), int(shard_id)


def encode_entry_point_id(source_id: int, shard_key: str, source_id_dedup_block_size: int | None) -> int:
    if source_id_dedup_block_size is None:
        return int(source_id)
    shard_id = int(shard_key.rsplit("_", 1)[1])
    return shard_id * int(source_id_dedup_block_size) + int(source_id) + 1


def source_id_from_scrolled_point(point: dict[str, Any], num_points: int) -> int:
    payload = point.get("payload") or {}
    if "source_id" in payload:
        return int(payload["source_id"])
    source_id, _shard_id = decode_copy_id(int(point["id"]), num_points)
    return source_id


def add_scrolled_points_to_upper_shards(
    point_to_shards: list[list[int]],
    upper_set: set[int],
    shard_id: int,
    points: list[dict[str, Any]],
    num_points: int,
) -> None:
    for point in points:
        source_id = source_id_from_scrolled_point(point, num_points)
        if source_id in upper_set:
            point_to_shards[source_id].append(int(shard_id))


def recover_upper_point_to_shards_from_collection(
    base_url: str,
    collection: str,
    upper_indices: np.ndarray,
    num_points: int,
    num_shards: int,
    page_size: int = 10000,
) -> list[list[int]]:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    upper_set = {int(point_id) for point_id in upper_indices.tolist()}
    point_to_shards: list[list[int]] = [[] for _ in range(num_points)]

    for shard_id in range(num_shards):
        shard_key = shard_key_for_id(shard_id)
        offset: Any = None
        while True:
            body: dict[str, Any] = {
                "limit": page_size,
                "with_payload": ["source_id"],
                "with_vector": False,
                "shard_key": shard_key,
            }
            if offset is not None:
                body["offset"] = offset
            result = request_json(
                base_url,
                "POST",
                f"/collections/{urllib.parse.quote(collection, safe='')}/points/scroll",
                body=body,
                timeout=300.0,
            )["result"]
            points = result.get("points") or []
            add_scrolled_points_to_upper_shards(
                point_to_shards,
                upper_set,
                shard_id,
                points,
                num_points,
            )
            offset = result.get("next_page_offset")
            if offset is None:
                break

    missing = [int(point_id) for point_id in upper_indices.tolist() if not point_to_shards[int(point_id)]]
    if missing:
        preview = ", ".join(str(point_id) for point_id in missing[:10])
        raise RuntimeError(
            f"Recovered no shard membership for {len(missing)} upper points; first missing: {preview}"
        )
    return point_to_shards


def ensure_collection_from_point_shards(
    base_url: str,
    collection: str,
    train: np.ndarray,
    point_to_shards: list[list[int]],
    upper_indices: np.ndarray,
    num_shards: int,
    hnsw_m: int,
    ef_construct: int,
    upload_batch_size: int,
    reuse_existing: bool,
    shard_placement: str,
    peer_ids: list[int],
) -> dict[str, Any]:
    shard_keys = [shard_key_for_id(shard_id) for shard_id in range(num_shards)]
    expected_points = total_assigned_points(point_to_shards)
    if not reuse_existing or not collection_exists(base_url, collection):
        delete_collection_if_exists(base_url, collection)
        create_collection(base_url, collection, train.shape[1], hnsw_m, ef_construct)
        for shard_id, shard_key in enumerate(shard_keys):
            placement = placement_for_shard_key(shard_id, peer_ids, shard_placement)
            create_shard_key(base_url, collection, shard_key, placement=placement)

        points_by_shard = point_indices_by_shard(point_to_shards, num_shards, upper_indices)
        for shard_id, shard_key in enumerate(shard_keys):
            point_indices = points_by_shard[shard_id]
            for start_idx in range(0, len(point_indices), upload_batch_size):
                idx_chunk = point_indices[start_idx : start_idx + upload_batch_size]
                ids = [encode_copy_id(int(point_idx), shard_id, len(train)) for point_idx in idx_chunk.tolist()]
                source_ids = [int(point_idx) for point_idx in idx_chunk.tolist()]
                vectors = train[idx_chunk].tolist()
                upsert_points(base_url, collection, shard_key, ids, vectors, source_ids=source_ids)

    info = wait_collection_indexed(base_url, collection, expected_points)
    cluster_summary = collection_cluster_summary(collection_cluster_info(base_url, collection))
    return {
        "collection": collection,
        "points_count": int(info.get("points_count") or 0),
        "indexed_vectors_count": int(info.get("indexed_vectors_count") or 0),
        "segments_count": int(info.get("segments_count") or 0),
        "shard_placement": shard_placement,
        "discovered_peer_count": len(peer_ids),
        "logical_points_count": int(len(train)),
        "assigned_points_count": int(expected_points),
        **cluster_summary,
    }


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
    rows: list[list[tuple[float, int]]] = []
    for per_query in payload["result"]:
        row: list[tuple[float, int]] = []
        for item in per_query:
            payload_value = item.get("payload") or {}
            if "source_id" in payload_value:
                point_id = int(payload_value["source_id"])
            else:
                point_id = int(item["id"]) - 1
            row.append((float(item["score"]), point_id))
        rows.append(row)
    return rows


def compute_upper_labels(
    upper_index: Any,
    queries: np.ndarray,
    upper_k: int,
) -> np.ndarray:
    k = min(upper_k, upper_index.get_current_count())
    labels, _distances = upper_index.knn_query(queries, k=k)
    return labels


def shard_efs_from_upper_hits(
    shard_hit_counts: dict[str, int],
    base_ef: int,
    factor: int,
) -> tuple[list[str], list[int]]:
    ranked = [
        (shard_key, hit_count)
        for shard_key, hit_count in shard_hit_counts.items()
        if hit_count > 0
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    shard_keys = [shard_key for shard_key, _count in ranked]
    ef_values = [base_ef + factor * count for _shard_key, count in ranked]
    return shard_keys, ef_values


def group_selected_keys_by_ef(keys: list[str], ef_values: list[int]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for key, ef_value in zip(keys, ef_values):
        grouped.setdefault(ef_value, []).append(key)
    return grouped


def compact_ef_value(ef_values: list[int], mode: str) -> int:
    if not ef_values:
        raise ValueError("ef_values must not be empty")
    if mode == "max":
        return int(max(ef_values))
    if mode == "mean_ceil":
        return int(ceil(sum(ef_values) / len(ef_values)))
    raise ValueError(f"unsupported compact EF mode: {mode}")


def routed_result_limit(top_k: int, shard_count: int, mode: str, multiplier: float = 1) -> int:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if mode == "top_k":
        return int(top_k)
    if mode == "per_shard_top_k":
        return int(top_k) * int(shard_count)
    if mode == "fixed_multiplier":
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        return int(ceil(int(top_k) * float(multiplier)))
    raise ValueError(f"unsupported routed result limit mode: {mode}")


def merge_topk_candidates(candidate_groups: list[list[tuple[float, int]]], top_k: int) -> list[int]:
    best_by_id: dict[int, float] = {}
    for group in candidate_groups:
        for score, point_id in group:
            current = best_by_id.get(point_id)
            if current is None or score > current:
                best_by_id[point_id] = score
    ordered = sorted(best_by_id.items(), key=lambda item: item[1], reverse=True)
    return [point_id for point_id, _score in ordered[:top_k]]


def evaluate_config(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    upper_index: Any,
    upper_k: int,
    base_ef: int,
    factor: int,
    batch_size: int,
    label_to_shard: dict[int, str] | None = None,
    point_to_shards: list[list[int]] | None = None,
    num_shards: int | None = None,
    search_all_shards: bool = False,
    use_payload_source_id: bool = False,
    routed_execution_mode: str = "grouped_by_ef",
    compact_ef_mode: str = "max",
    routed_result_limit_mode: str = "top_k",
    routed_result_limit_multiplier: float = 1,
    source_id_dedup_block_size: int | None = None,
) -> dict[str, Any]:
    if label_to_shard is None and point_to_shards is None:
        raise ValueError("either label_to_shard or point_to_shards must be provided")
    if point_to_shards is not None and num_shards is None:
        raise ValueError("num_shards is required with point_to_shards")
    if routed_execution_mode not in {"grouped_by_ef", "compact_query_ef", "per_shard_multi_ep", "compact_multi_ep"}:
        raise ValueError(f"unsupported routed execution mode: {routed_execution_mode}")

    total_hits = 0
    total_queries = len(queries)
    total_visited_shards = 0
    total_upper_hits = 0
    total_assigned_ef = 0
    total_assigned_ef_count = 0
    search_batch_calls = 0
    search_request_count = 0
    start = time.perf_counter()

    for start_idx in range(0, total_queries, batch_size):
        chunk = queries[start_idx : start_idx + batch_size]
        upper_labels = compute_upper_labels(upper_index, chunk, upper_k)

        per_query_candidates: list[list[list[tuple[float, int]]]] = [[] for _ in range(len(chunk))]
        flat_searches: list[dict[str, Any]] = []
        flat_query_positions: list[int] = []

        for local_idx, labels_row in enumerate(upper_labels):
            if point_to_shards is not None:
                shard_to_eps = route_upper_labels_to_shard_eps(labels_row.tolist(), point_to_shards)
                shard_keys, ef_values = shard_efs_from_routed_eps(
                    shard_to_eps,
                    int(num_shards),
                    base_ef,
                    factor,
                    search_all_shards=search_all_shards,
                )
                total_upper_hits += int(sum(len(eps) for eps in shard_to_eps.values()))
            else:
                shard_hit_counts = Counter()
                assert label_to_shard is not None
                for label in labels_row.tolist():
                    shard_key = label_to_shard.get(int(label))
                    if shard_key is not None:
                        shard_hit_counts[shard_key] += 1

                shard_keys, ef_values = shard_efs_from_upper_hits(dict(shard_hit_counts), base_ef, factor)
                total_upper_hits += int(sum(shard_hit_counts.values()))
            total_visited_shards += len(shard_keys)

            if not shard_keys:
                continue
            if routed_execution_mode == "per_shard_multi_ep" and point_to_shards is not None:
                total_assigned_ef += sum(ef_values)
                total_assigned_ef_count += len(ef_values)
                for shard_key, ef_value in zip(shard_keys, ef_values):
                    shard_id = int(shard_key.rsplit("_", 1)[1])
                    flat_query_positions.append(local_idx)
                    flat_searches.append(
                        search_request(
                            chunk[local_idx].tolist(),
                            top_k,
                            ef_value,
                            [shard_key],
                        use_payload_source_id,
                        shard_to_eps.get(shard_id, []),
                        source_id_dedup_block_size=source_id_dedup_block_size,
                    )
                )
            elif routed_execution_mode == "compact_multi_ep" and point_to_shards is not None:
                total_assigned_ef += sum(ef_values)
                total_assigned_ef_count += len(ef_values)
                entry_points_by_shard: dict[str, list[int]] = {}
                ef_by_shard: dict[str, int] = {}
                for shard_key, ef_value in zip(shard_keys, ef_values):
                    shard_id = int(shard_key.rsplit("_", 1)[1])
                    entry_points_by_shard[shard_key] = shard_to_eps.get(shard_id, [])
                    ef_by_shard[shard_key] = int(ef_value)
                flat_query_positions.append(local_idx)
                flat_searches.append(
                    search_request(
                        chunk[local_idx].tolist(),
                        routed_result_limit(
                            top_k,
                            len(shard_keys),
                            routed_result_limit_mode,
                            routed_result_limit_multiplier,
                        ),
                        max(ef_values),
                        shard_keys,
                        use_payload_source_id,
                        hnsw_entry_points_by_shard=entry_points_by_shard,
                        hnsw_ef_by_shard=ef_by_shard,
                        source_id_dedup_block_size=source_id_dedup_block_size,
                    )
                )
            elif routed_execution_mode == "compact_query_ef":
                compact_ef = compact_ef_value(ef_values, compact_ef_mode)
                total_assigned_ef += compact_ef * len(shard_keys)
                total_assigned_ef_count += len(shard_keys)
                flat_query_positions.append(local_idx)
                flat_searches.append(
                    search_request(
                        chunk[local_idx].tolist(),
                        routed_result_limit(
                            top_k,
                            len(shard_keys),
                            routed_result_limit_mode,
                            routed_result_limit_multiplier,
                        ),
                        compact_ef,
                        shard_keys,
                        use_payload_source_id,
                        source_id_dedup_block_size=source_id_dedup_block_size,
                    )
                )
            else:
                total_assigned_ef += sum(ef_values)
                total_assigned_ef_count += len(ef_values)
                grouped = group_selected_keys_by_ef(shard_keys, ef_values)
                for ef_value, grouped_keys in grouped.items():
                    flat_query_positions.append(local_idx)
                    flat_searches.append(
                        search_request(
                            chunk[local_idx].tolist(),
                            top_k,
                            ef_value,
                            grouped_keys,
                            use_payload_source_id,
                            source_id_dedup_block_size=source_id_dedup_block_size,
                        )
                    )

        if flat_searches:
            search_batch_calls += 1
            search_request_count += len(flat_searches)
            results = search_batch(base_url, collection, flat_searches)
            for local_idx, result in zip(flat_query_positions, results):
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
        "avg_visited_shards": total_visited_shards / total_queries,
        "avg_upper_hits": total_upper_hits / total_queries,
        "avg_assigned_ef_per_visited_shard": (
            total_assigned_ef / total_assigned_ef_count if total_assigned_ef_count > 0 else 0.0
        ),
        "search_batch_calls": search_batch_calls,
        "avg_search_requests_per_query": search_request_count / total_queries,
    }


def evaluate_config_materialized_routing(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    upper_index: Any,
    upper_k: int,
    base_ef: int,
    factor: int,
    batch_size: int,
    label_to_shard: dict[int, str] | None = None,
    point_to_shards: list[list[int]] | None = None,
    num_shards: int | None = None,
    search_all_shards: bool = False,
    use_payload_source_id: bool = False,
    routed_execution_mode: str = "grouped_by_ef",
    compact_ef_mode: str = "max",
    routed_result_limit_mode: str = "top_k",
    routed_result_limit_multiplier: float = 1,
    source_id_dedup_block_size: int | None = None,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if label_to_shard is None and point_to_shards is None:
        raise ValueError("either label_to_shard or point_to_shards must be provided")
    if point_to_shards is not None and num_shards is None:
        raise ValueError("num_shards is required with point_to_shards")

    total_queries = len(queries)
    start = time.perf_counter()
    upper_labels = compute_upper_labels(upper_index, queries, upper_k)
    if point_to_shards is not None:
        query_plans = build_routed_search_plans(
            queries,
            upper_labels,
            point_to_shards,
            int(num_shards),
            top_k,
            base_ef,
            factor,
            search_all_shards,
            use_payload_source_id,
            routed_execution_mode,
            compact_ef_mode,
            routed_result_limit_mode,
            routed_result_limit_multiplier,
            source_id_dedup_block_size,
        )
    else:
        assert label_to_shard is not None
        query_plans = [
            legacy_routed_search_plan(
                query.tolist(),
                labels_row,
                label_to_shard,
                top_k,
                base_ef,
                factor,
                use_payload_source_id,
                routed_execution_mode,
                compact_ef_mode,
                routed_result_limit_mode,
                routed_result_limit_multiplier,
                source_id_dedup_block_size,
            )
            for query, labels_row in zip(queries, upper_labels)
        ]

    total_hits = 0
    total_executed_queries = 0
    total_visited_shards = 0
    total_upper_hits = 0
    total_assigned_ef = 0
    total_assigned_ef_count = 0
    search_batch_calls = 0
    search_request_count = 0

    for start_idx in range(0, total_queries, batch_size):
        end_idx = min(start_idx + batch_size, total_queries)
        result = execute_query_plans_once(
            base_url,
            collection,
            query_plans[start_idx:end_idx],
            neighbors[start_idx:end_idx],
            top_k,
            direct_peer_urls=direct_peer_urls,
            shard_key_to_peer=shard_key_to_peer,
        )
        total_hits += int(result["hits"])
        total_executed_queries += int(result["query_count"])
        total_visited_shards += int(result["visited_shards"])
        total_upper_hits += int(result["upper_hits"])
        total_assigned_ef += int(result["assigned_ef_sum"])
        total_assigned_ef_count += int(result["assigned_ef_count"])
        search_batch_calls += int(result["search_batch_calls"])
        search_request_count += int(result["search_request_count"])

    wall = time.perf_counter() - start
    return {
        "recall_at_k": total_hits / (total_executed_queries * top_k),
        "qps": total_executed_queries / wall,
        "wall_s": wall,
        "avg_visited_shards": total_visited_shards / total_executed_queries,
        "avg_upper_hits": total_upper_hits / total_executed_queries,
        "avg_assigned_ef_per_visited_shard": (
            total_assigned_ef / total_assigned_ef_count if total_assigned_ef_count > 0 else 0.0
        ),
        "search_batch_calls": search_batch_calls,
        "avg_search_requests_per_query": search_request_count / total_executed_queries,
    }


def evaluate_all_shards_config(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    num_shards: int,
    hnsw_ef: int,
    batch_size: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
) -> dict[str, Any]:
    total_hits = 0
    total_queries = len(queries)
    shard_keys, _ef_values = all_shard_keys_and_ef(num_shards, hnsw_ef)
    start = time.perf_counter()

    for start_idx in range(0, total_queries, batch_size):
        chunk = queries[start_idx : start_idx + batch_size]
        searches = []
        for query in chunk:
            request = {
                "vector": query.tolist(),
                "limit": top_k,
                "params": {"hnsw_ef": hnsw_ef},
                "with_payload": ["source_id"] if use_payload_source_id else False,
                "with_vector": False,
                "shard_key": shard_keys,
            }
            if source_id_dedup_block_size is not None:
                request["source_id_dedup_block_size"] = int(source_id_dedup_block_size)
            searches.append(request)
        results = search_batch(base_url, collection, searches)
        for local_idx, result in enumerate(results):
            top_ids = merge_topk_candidates([result], top_k)
            gt = set(map(int, neighbors[start_idx + local_idx]))
            total_hits += len(set(top_ids) & gt)

    wall = time.perf_counter() - start
    return {
        "recall_at_k": total_hits / (total_queries * top_k),
        "qps": total_queries / wall,
        "wall_s": wall,
        "avg_visited_shards": float(num_shards),
        "avg_upper_hits": 0.0,
        "avg_assigned_ef_per_visited_shard": float(hnsw_ef),
        "search_batch_calls": int(ceil(total_queries / batch_size)),
        "avg_search_requests_per_query": 1.0,
    }


def search_request(
    query: list[float],
    top_k: int,
    hnsw_ef: int,
    shard_keys: list[str],
    use_payload_source_id: bool,
    hnsw_entry_points: list[int] | None = None,
    hnsw_entry_points_by_shard: dict[str, list[int]] | None = None,
    hnsw_ef_by_shard: dict[str, int] | None = None,
    source_id_dedup_block_size: int | None = None,
) -> dict[str, Any]:
    request = {
        "vector": query,
        "limit": top_k,
        "params": {"hnsw_ef": int(hnsw_ef)},
        "with_payload": ["source_id"] if use_payload_source_id else False,
        "with_vector": False,
        "shard_key": shard_keys,
    }
    if hnsw_entry_points:
        if source_id_dedup_block_size is not None and len(shard_keys) != 1:
            raise ValueError("single-shard hnsw_entry_points are required when encoding copied point IDs")
        shard_key = shard_keys[0] if source_id_dedup_block_size is not None else ""
        request["hnsw_entry_points"] = [
            encode_entry_point_id(point_id, shard_key, source_id_dedup_block_size)
            for point_id in hnsw_entry_points
        ]
    if hnsw_entry_points_by_shard:
        request["hnsw_entry_points_by_shard"] = {
            str(shard_key): [
                encode_entry_point_id(point_id, str(shard_key), source_id_dedup_block_size)
                for point_id in entry_points
            ]
            for shard_key, entry_points in hnsw_entry_points_by_shard.items()
        }
    if hnsw_ef_by_shard:
        request["hnsw_ef_by_shard"] = {
            str(shard_key): int(ef_value)
            for shard_key, ef_value in hnsw_ef_by_shard.items()
        }
    if source_id_dedup_block_size is not None:
        request["source_id_dedup_block_size"] = int(source_id_dedup_block_size)
    return request


def all_shard_search_plan(
    query: list[float],
    top_k: int,
    num_shards: int,
    hnsw_ef: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
) -> dict[str, Any]:
    shard_keys, _ef_values = all_shard_keys_and_ef(num_shards, hnsw_ef)
    return {
        "searches": [
            search_request(
                query,
                top_k,
                hnsw_ef,
                shard_keys,
                use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
            )
        ],
        "visited_shards": int(num_shards),
        "upper_hits": 0,
        "assigned_ef_sum": int(hnsw_ef) * int(num_shards),
        "assigned_ef_count": int(num_shards),
    }


def routed_search_plan(
    query: list[float],
    upper_labels: list[int] | np.ndarray,
    point_to_shards: list[list[int]],
    num_shards: int,
    top_k: int,
    base_ef: int,
    factor: int,
    search_all_shards: bool,
    use_payload_source_id: bool,
    routed_execution_mode: str = "grouped_by_ef",
    compact_ef_mode: str = "max",
    routed_result_limit_mode: str = "top_k",
    routed_result_limit_multiplier: float = 1,
    source_id_dedup_block_size: int | None = None,
) -> dict[str, Any]:
    if routed_execution_mode not in {"grouped_by_ef", "compact_query_ef", "per_shard_multi_ep", "compact_multi_ep"}:
        raise ValueError(f"unsupported routed execution mode: {routed_execution_mode}")
    shard_to_eps = route_upper_labels_to_shard_eps(upper_labels, point_to_shards)
    shard_keys, ef_values = shard_efs_from_routed_eps(
        shard_to_eps,
        num_shards,
        base_ef,
        factor,
        search_all_shards=search_all_shards,
    )

    searches = []
    assigned_ef_sum = int(sum(ef_values))
    if routed_execution_mode == "per_shard_multi_ep":
        for shard_key, ef_value in zip(shard_keys, ef_values):
            shard_id = int(shard_key.rsplit("_", 1)[1])
            searches.append(
                search_request(
                    query,
                    top_k,
                    ef_value,
                    [shard_key],
                    use_payload_source_id,
                    shard_to_eps.get(shard_id, []),
                    source_id_dedup_block_size=source_id_dedup_block_size,
                )
            )
    elif routed_execution_mode == "compact_multi_ep" and shard_keys:
        entry_points_by_shard: dict[str, list[int]] = {}
        ef_by_shard: dict[str, int] = {}
        for shard_key, ef_value in zip(shard_keys, ef_values):
            shard_id = int(shard_key.rsplit("_", 1)[1])
            entry_points_by_shard[shard_key] = shard_to_eps.get(shard_id, [])
            ef_by_shard[shard_key] = int(ef_value)
        searches.append(
            search_request(
                query,
                routed_result_limit(
                    top_k,
                    len(shard_keys),
                    routed_result_limit_mode,
                    routed_result_limit_multiplier,
                ),
                max(ef_values),
                shard_keys,
                use_payload_source_id,
                hnsw_entry_points_by_shard=entry_points_by_shard,
                hnsw_ef_by_shard=ef_by_shard,
                source_id_dedup_block_size=source_id_dedup_block_size,
            )
        )
    elif routed_execution_mode == "compact_query_ef" and shard_keys:
        compact_ef = compact_ef_value(ef_values, compact_ef_mode)
        assigned_ef_sum = compact_ef * len(shard_keys)
        searches.append(
            search_request(
                query,
                routed_result_limit(
                    top_k,
                    len(shard_keys),
                    routed_result_limit_mode,
                    routed_result_limit_multiplier,
                ),
                compact_ef,
                shard_keys,
                use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
            )
        )
    else:
        for ef_value, grouped_keys in group_selected_keys_by_ef(shard_keys, ef_values).items():
            searches.append(
                search_request(
                    query,
                    top_k,
                    ef_value,
                grouped_keys,
                use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
            )
        )

    return {
        "searches": searches,
        "visited_shards": len(shard_keys),
        "upper_hits": int(sum(len(eps) for eps in shard_to_eps.values())),
        "assigned_ef_sum": assigned_ef_sum,
        "assigned_ef_count": len(ef_values),
    }


def build_all_shard_search_plans(
    queries: np.ndarray,
    top_k: int,
    num_shards: int,
    hnsw_ef: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
) -> list[dict[str, Any]]:
    return [
        all_shard_search_plan(
            query.tolist(),
            top_k,
            num_shards,
            hnsw_ef,
            use_payload_source_id,
            source_id_dedup_block_size,
        )
        for query in queries
    ]


def build_routed_search_plans(
    queries: np.ndarray,
    upper_labels: np.ndarray,
    point_to_shards: list[list[int]],
    num_shards: int,
    top_k: int,
    base_ef: int,
    factor: int,
    search_all_shards: bool,
    use_payload_source_id: bool,
    routed_execution_mode: str = "grouped_by_ef",
    compact_ef_mode: str = "max",
    routed_result_limit_mode: str = "top_k",
    routed_result_limit_multiplier: float = 1,
    source_id_dedup_block_size: int | None = None,
) -> list[dict[str, Any]]:
    return [
        routed_search_plan(
            query.tolist(),
            labels_row,
            point_to_shards,
            num_shards,
            top_k,
            base_ef,
            factor,
            search_all_shards,
            use_payload_source_id,
            routed_execution_mode,
            compact_ef_mode,
            routed_result_limit_mode,
            routed_result_limit_multiplier,
            source_id_dedup_block_size,
        )
        for query, labels_row in zip(queries, upper_labels)
    ]


def execute_query_plans_once(
    base_url: str,
    collection: str,
    query_plans: list[dict[str, Any]],
    neighbors: Any,
    top_k: int,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
) -> dict[str, Any]:
    per_query_candidates: list[list[list[tuple[float, int]]]] = [
        [] for _ in range(len(query_plans))
    ]

    search_batch_calls = 0
    search_request_count = 0

    if direct_peer_urls is not None:
        if shard_key_to_peer is None:
            raise ValueError("shard_key_to_peer is required with direct_peer_urls")

        peer_batches: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for query_idx, plan in enumerate(query_plans):
            for search in plan["searches"]:
                keys_by_peer: dict[int, list[str]] = defaultdict(list)
                for shard_key in search.get("shard_key") or []:
                    if shard_key not in shard_key_to_peer:
                        raise ValueError(f"missing peer mapping for shard key {shard_key}")
                    keys_by_peer[int(shard_key_to_peer[shard_key])].append(shard_key)
                for peer_id, shard_keys in keys_by_peer.items():
                    if peer_id not in direct_peer_urls:
                        raise ValueError(f"missing direct HTTP URL for peer {peer_id}")
                    split_search = dict(search)
                    split_search["shard_key"] = shard_keys
                    peer_batches[peer_id].append((query_idx, split_search))

        search_batch_calls = len(peer_batches)
        search_request_count = sum(len(items) for items in peer_batches.values())

        def run_peer_batch(peer_id: int, items: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, list[tuple[float, int]]]]:
            results = search_batch(
                direct_peer_urls[peer_id],
                collection,
                [search for _query_idx, search in items],
            )
            return [
                (query_idx, result)
                for (query_idx, _search), result in zip(items, results)
            ]

        if len(peer_batches) == 1:
            peer_results = [
                result
                for peer_id, items in peer_batches.items()
                for result in run_peer_batch(peer_id, items)
            ]
        else:
            with ThreadPoolExecutor(max_workers=len(peer_batches)) as pool:
                futures = [
                    pool.submit(run_peer_batch, peer_id, items)
                    for peer_id, items in peer_batches.items()
                ]
                peer_results = [
                    result
                    for future in futures
                    for result in future.result()
                ]

        for local_idx, result in peer_results:
            per_query_candidates[local_idx].append(result)
    else:
        flat_searches: list[dict[str, Any]] = []
        query_positions: list[int] = []
        for query_idx, plan in enumerate(query_plans):
            for search in plan["searches"]:
                query_positions.append(query_idx)
                flat_searches.append(search)

        search_request_count = len(flat_searches)
        if flat_searches:
            search_batch_calls = 1
            results = search_batch(base_url, collection, flat_searches)
            for local_idx, result in zip(query_positions, results):
                per_query_candidates[local_idx].append(result)

    hits = 0
    for offset, candidate_groups in enumerate(per_query_candidates):
        top_ids = merge_topk_candidates(candidate_groups, top_k)
        gt = set(map(int, neighbors[offset]))
        hits += len(set(top_ids) & gt)

    return {
        "hits": hits,
        "query_count": len(query_plans),
        "visited_shards": sum(int(plan["visited_shards"]) for plan in query_plans),
        "upper_hits": sum(int(plan["upper_hits"]) for plan in query_plans),
        "assigned_ef_sum": sum(int(plan["assigned_ef_sum"]) for plan in query_plans),
        "assigned_ef_count": sum(int(plan["assigned_ef_count"]) for plan in query_plans),
        "search_batch_calls": search_batch_calls,
        "search_request_count": search_request_count,
    }


def legacy_routed_search_plan(
    query: list[float],
    upper_labels: list[int] | np.ndarray,
    label_to_shard: dict[int, str],
    top_k: int,
    base_ef: int,
    factor: int,
    use_payload_source_id: bool,
    routed_execution_mode: str = "grouped_by_ef",
    compact_ef_mode: str = "max",
    routed_result_limit_mode: str = "top_k",
    routed_result_limit_multiplier: float = 1,
    source_id_dedup_block_size: int | None = None,
) -> dict[str, Any]:
    if routed_execution_mode not in {"grouped_by_ef", "compact_query_ef", "per_shard_multi_ep", "compact_multi_ep"}:
        raise ValueError(f"unsupported routed execution mode: {routed_execution_mode}")

    shard_hit_counts = Counter()
    shard_to_eps: dict[str, list[int]] = defaultdict(list)
    for label in list(upper_labels):
        shard_key = label_to_shard.get(int(label))
        if shard_key is not None:
            shard_hit_counts[shard_key] += 1
            shard_to_eps[shard_key].append(int(label))

    shard_keys, ef_values = shard_efs_from_upper_hits(dict(shard_hit_counts), base_ef, factor)
    searches: list[dict[str, Any]] = []
    assigned_ef_sum = int(sum(ef_values))

    if routed_execution_mode == "per_shard_multi_ep":
        for shard_key, ef_value in zip(shard_keys, ef_values):
            searches.append(
                search_request(
                    query,
                    top_k,
                    ef_value,
                    [shard_key],
                    use_payload_source_id,
                    shard_to_eps.get(shard_key, []),
                    source_id_dedup_block_size=source_id_dedup_block_size,
                )
            )
    elif routed_execution_mode == "compact_multi_ep" and shard_keys:
        searches.append(
            search_request(
                query,
                routed_result_limit(
                    top_k,
                    len(shard_keys),
                    routed_result_limit_mode,
                    routed_result_limit_multiplier,
                ),
                max(ef_values),
                shard_keys,
                use_payload_source_id,
                hnsw_entry_points_by_shard={
                    shard_key: shard_to_eps.get(shard_key, [])
                    for shard_key in shard_keys
                },
                hnsw_ef_by_shard={
                    shard_key: int(ef_value)
                    for shard_key, ef_value in zip(shard_keys, ef_values)
                },
                source_id_dedup_block_size=source_id_dedup_block_size,
            )
        )
    elif routed_execution_mode == "compact_query_ef" and shard_keys:
        compact_ef = compact_ef_value(ef_values, compact_ef_mode)
        assigned_ef_sum = compact_ef * len(shard_keys)
        searches.append(
            search_request(
                query,
                routed_result_limit(
                    top_k,
                    len(shard_keys),
                    routed_result_limit_mode,
                    routed_result_limit_multiplier,
                ),
                compact_ef,
                shard_keys,
                use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
            )
        )
    else:
        for ef_value, grouped_keys in group_selected_keys_by_ef(shard_keys, ef_values).items():
            searches.append(
                search_request(
                    query,
                    top_k,
                    ef_value,
                grouped_keys,
                use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
            )
        )

    return {
        "searches": searches,
        "visited_shards": len(shard_keys),
        "upper_hits": int(sum(shard_hit_counts.values())),
        "assigned_ef_sum": assigned_ef_sum,
        "assigned_ef_count": len(ef_values),
    }


def evaluate_routed_config_batch(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: Any,
    top_k: int,
    upper_index: Any,
    upper_k: int,
    base_ef: int,
    factor: int,
    label_to_shard: dict[int, str] | None = None,
    point_to_shards: list[list[int]] | None = None,
    num_shards: int | None = None,
    search_all_shards: bool = False,
    use_payload_source_id: bool = False,
    routed_execution_mode: str = "grouped_by_ef",
    compact_ef_mode: str = "max",
    routed_result_limit_mode: str = "top_k",
    routed_result_limit_multiplier: float = 1,
    source_id_dedup_block_size: int | None = None,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
) -> dict[str, Any]:
    if label_to_shard is None and point_to_shards is None:
        raise ValueError("either label_to_shard or point_to_shards must be provided")
    if point_to_shards is not None and num_shards is None:
        raise ValueError("num_shards is required with point_to_shards")

    upper_labels = compute_upper_labels(upper_index, queries, upper_k)
    if point_to_shards is not None:
        query_plans = build_routed_search_plans(
            queries,
            upper_labels,
            point_to_shards,
            int(num_shards),
            top_k,
            base_ef,
            factor,
            search_all_shards,
            use_payload_source_id,
            routed_execution_mode,
            compact_ef_mode,
            routed_result_limit_mode,
            routed_result_limit_multiplier,
            source_id_dedup_block_size,
        )
    else:
        assert label_to_shard is not None
        query_plans = [
            legacy_routed_search_plan(
                query.tolist(),
                labels_row,
                label_to_shard,
                top_k,
                base_ef,
                factor,
                use_payload_source_id,
                routed_execution_mode,
                compact_ef_mode,
                routed_result_limit_mode,
                routed_result_limit_multiplier,
                source_id_dedup_block_size,
            )
            for query, labels_row in zip(queries, upper_labels)
        ]

    return execute_query_plans_once(
        base_url,
        collection,
        query_plans,
        neighbors,
        top_k,
        direct_peer_urls=direct_peer_urls,
        shard_key_to_peer=shard_key_to_peer,
    )


def evaluate_all_shards_config_batch(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: Any,
    top_k: int,
    num_shards: int,
    hnsw_ef: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
) -> dict[str, Any]:
    query_plans = build_all_shard_search_plans(
        queries,
        top_k,
        num_shards,
        hnsw_ef,
        use_payload_source_id,
        source_id_dedup_block_size,
    )
    return execute_query_plans_once(
        base_url,
        collection,
        query_plans,
        neighbors,
        top_k,
        direct_peer_urls=direct_peer_urls,
        shard_key_to_peer=shard_key_to_peer,
    )


def aggregate_timed_batch_results(
    batch_results: list[dict[str, Any]],
    wall: float,
    top_k: int,
) -> dict[str, Any]:
    total_hits = sum(int(result["hits"]) for result in batch_results)
    total_queries = sum(int(result["query_count"]) for result in batch_results)
    total_visited_shards = sum(int(result["visited_shards"]) for result in batch_results)
    total_upper_hits = sum(int(result["upper_hits"]) for result in batch_results)
    total_assigned_ef = sum(int(result["assigned_ef_sum"]) for result in batch_results)
    total_assigned_ef_count = sum(int(result["assigned_ef_count"]) for result in batch_results)
    search_batch_calls = sum(int(result["search_batch_calls"]) for result in batch_results)
    search_request_count = sum(int(result["search_request_count"]) for result in batch_results)

    return {
        "recall_at_k": total_hits / (total_queries * top_k),
        "qps": total_queries / wall,
        "wall_s": wall,
        "avg_visited_shards": total_visited_shards / total_queries,
        "avg_upper_hits": total_upper_hits / total_queries,
        "avg_assigned_ef_per_visited_shard": (
            total_assigned_ef / total_assigned_ef_count if total_assigned_ef_count > 0 else 0.0
        ),
        "search_batch_calls": search_batch_calls,
        "avg_search_requests_per_query": search_request_count / total_queries,
    }


def evaluate_config_concurrent(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    upper_index: Any,
    upper_k: int,
    base_ef: int,
    factor: int,
    batch_size: int,
    concurrency: int,
    label_to_shard: dict[int, str] | None = None,
    point_to_shards: list[list[int]] | None = None,
    num_shards: int | None = None,
    search_all_shards: bool = False,
    use_payload_source_id: bool = False,
    routed_execution_mode: str = "grouped_by_ef",
    compact_ef_mode: str = "max",
    routed_result_limit_mode: str = "top_k",
    routed_result_limit_multiplier: float = 1,
    source_id_dedup_block_size: int | None = None,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if len(queries) == 0:
        raise ValueError("queries must not be empty")

    ranges = [
        (start_idx, min(start_idx + batch_size, len(queries)))
        for start_idx in range(0, len(queries), batch_size)
    ]

    def run_range(start_idx: int, end_idx: int) -> dict[str, Any]:
        return evaluate_routed_config_batch(
            base_url,
            collection,
            queries[start_idx:end_idx],
            neighbors[start_idx:end_idx],
            top_k,
            upper_index,
            upper_k,
            base_ef,
            factor,
            label_to_shard=label_to_shard,
            point_to_shards=point_to_shards,
            num_shards=num_shards,
            search_all_shards=search_all_shards,
            use_payload_source_id=use_payload_source_id,
            routed_execution_mode=routed_execution_mode,
            compact_ef_mode=compact_ef_mode,
            routed_result_limit_mode=routed_result_limit_mode,
            routed_result_limit_multiplier=routed_result_limit_multiplier,
            source_id_dedup_block_size=source_id_dedup_block_size,
            direct_peer_urls=direct_peer_urls,
            shard_key_to_peer=shard_key_to_peer,
        )

    start = time.perf_counter()
    if concurrency == 1:
        batch_results = [run_range(start_idx, end_idx) for start_idx, end_idx in ranges]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(run_range, start_idx, end_idx)
                for start_idx, end_idx in ranges
            ]
            batch_results = [future.result() for future in futures]

    wall = time.perf_counter() - start
    return aggregate_timed_batch_results(batch_results, wall, top_k)


def evaluate_all_shards_config_concurrent(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    num_shards: int,
    hnsw_ef: int,
    batch_size: int,
    concurrency: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if len(queries) == 0:
        raise ValueError("queries must not be empty")

    ranges = [
        (start_idx, min(start_idx + batch_size, len(queries)))
        for start_idx in range(0, len(queries), batch_size)
    ]

    def run_range(start_idx: int, end_idx: int) -> dict[str, Any]:
        return evaluate_all_shards_config_batch(
            base_url,
            collection,
            queries[start_idx:end_idx],
            neighbors[start_idx:end_idx],
            top_k,
            num_shards,
            hnsw_ef,
            use_payload_source_id,
            source_id_dedup_block_size,
            direct_peer_urls=direct_peer_urls,
            shard_key_to_peer=shard_key_to_peer,
        )

    start = time.perf_counter()
    if concurrency == 1:
        batch_results = [run_range(start_idx, end_idx) for start_idx, end_idx in ranges]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(run_range, start_idx, end_idx)
                for start_idx, end_idx in ranges
            ]
            batch_results = [future.result() for future in futures]

    wall = time.perf_counter() - start
    return aggregate_timed_batch_results(batch_results, wall, top_k)


def evaluate_preplanned_search_batch(
    base_url: str,
    collection: str,
    query_plans: list[dict[str, Any]],
    neighbors: Any,
    top_k: int,
    start_idx: int,
    end_idx: int,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
) -> tuple[int, int]:
    result = execute_query_plans_once(
        base_url,
        collection,
        query_plans[start_idx:end_idx],
        neighbors[start_idx:end_idx],
        top_k,
        direct_peer_urls=direct_peer_urls,
        shard_key_to_peer=shard_key_to_peer,
    )
    return int(result["hits"]), int(result["query_count"])


def evaluate_preplanned_searches_concurrent(
    base_url: str,
    collection: str,
    query_plans: list[dict[str, Any]],
    neighbors: Any,
    top_k: int,
    batch_size: int,
    concurrency: int,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if not query_plans:
        raise ValueError("query_plans must not be empty")

    ranges = [
        (start_idx, min(start_idx + batch_size, len(query_plans)))
        for start_idx in range(0, len(query_plans), batch_size)
    ]

    total_hits = 0
    total_queries = 0
    start = time.perf_counter()
    if concurrency == 1:
        for start_idx, end_idx in ranges:
            hits, count = evaluate_preplanned_search_batch(
                base_url,
                collection,
                query_plans,
                neighbors,
                top_k,
                start_idx,
                end_idx,
                direct_peer_urls,
                shard_key_to_peer,
            )
            total_hits += hits
            total_queries += count
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    evaluate_preplanned_search_batch,
                    base_url,
                    collection,
                    query_plans,
                    neighbors,
                    top_k,
                    start_idx,
                    end_idx,
                    direct_peer_urls,
                    shard_key_to_peer,
                )
                for start_idx, end_idx in ranges
            ]
            for future in futures:
                hits, count = future.result()
                total_hits += hits
                total_queries += count

    wall = time.perf_counter() - start
    total_visited_shards = sum(int(plan["visited_shards"]) for plan in query_plans)
    total_upper_hits = sum(int(plan["upper_hits"]) for plan in query_plans)
    total_assigned_ef = sum(int(plan["assigned_ef_sum"]) for plan in query_plans)
    total_assigned_ef_count = sum(int(plan["assigned_ef_count"]) for plan in query_plans)
    return {
        "recall_at_k": total_hits / (total_queries * top_k),
        "qps": total_queries / wall,
        "wall_s": wall,
        "avg_visited_shards": total_visited_shards / total_queries,
        "avg_upper_hits": total_upper_hits / total_queries,
        "avg_assigned_ef_per_visited_shard": (
            total_assigned_ef / total_assigned_ef_count if total_assigned_ef_count > 0 else 0.0
        ),
        "search_batch_calls": len(ranges),
        "avg_search_requests_per_query": (
            sum(len(plan["searches"]) for plan in query_plans) / total_queries
        ),
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


def main() -> int:
    args = parse_args()
    if args.recover_routing_from_collection and (
        args.routing_mode != "faithful_original_rest" or not args.reuse_existing
    ):
        raise ValueError(
            "--recover-routing-from-collection requires --routing-mode faithful_original_rest "
            "and --reuse-existing"
        )
    output_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5_path, "r") as handle:
        train = slice_train_rows(handle["train"], args.train_limit)[:].astype(np.float32, copy=True)
        tuning_queries = handle["test"][: args.tuning_query_count].astype(np.float32, copy=True)
        tuning_neighbors = handle["neighbors"][: args.tuning_query_count, : args.top_k].astype(np.int32, copy=True)
        eval_queries = handle["test"][: args.eval_query_count].astype(np.float32, copy=True)
        eval_neighbors = handle["neighbors"][: args.eval_query_count, : args.top_k].astype(np.int32, copy=True)

    train = normalize_rows(train)
    tuning_queries = normalize_rows(tuning_queries)
    eval_queries = normalize_rows(eval_queries)

    peers = cluster_peer_ids(args.base_url, args.placement_peer_uri_contains)
    label_to_shard: dict[int, str] | None = None
    point_to_shards: list[list[int]] | None = None
    routing_state: OriginalRoutingState | None = None
    use_payload_source_id = False

    if args.routing_mode == "legacy_centroid":
        assignments, _shard_centroids = build_assignments_and_centroids(
            train,
            args.num_shards,
            args.sample_size,
            args.kmeans_iters,
            args.seed,
        )
        build_row = ensure_collection(
            args.base_url,
            args.collection,
            train,
            assignments,
            args.num_shards,
            args.hnsw_m,
            args.ef_construct,
            args.upload_batch_size,
            args.reuse_existing,
            args.shard_placement,
            peers,
        )

        upper_vectors, upper_ids, label_to_shard, sample_rows = sample_upper_points(
            train,
            assignments,
            args.num_shards,
            args.sample_denominator,
            args.seed,
        )
        upper_index = build_upper_index(
            upper_vectors,
            upper_ids,
            train.shape[1],
            args.upper_m,
            args.upper_ef_construction,
            args.upper_search_ef,
        )
        effective_num_shards = args.num_shards
    elif args.routing_mode == "naive_hash_all_shards":
        point_to_shards = hash_point_to_shards(len(train), args.num_shards)
        effective_num_shards = args.num_shards
        empty_upper_indices = np.asarray([], dtype=np.int64)
        build_row = ensure_collection_from_point_shards(
            args.base_url,
            args.collection,
            train,
            point_to_shards,
            empty_upper_indices,
            effective_num_shards,
            args.hnsw_m,
            args.ef_construct,
            args.upload_batch_size,
            args.reuse_existing,
            args.shard_placement,
            peers,
        )
        shard_counts = np.bincount(
            np.asarray([shards[0] for shards in point_to_shards], dtype=np.int64),
            minlength=effective_num_shards,
        )
        sample_rows = [
            {
                "shard_id": shard_id,
                "shard_key": shard_key_for_id(shard_id),
                "points_count": int(shard_counts[shard_id]),
                "sample_count": 0,
            }
            for shard_id in range(effective_num_shards)
        ]
        upper_ids = empty_upper_indices
        upper_index = None
        use_payload_source_id = True
    else:
        upper_indices = global_upper_indices(
            len(train),
            args.sample_denominator,
            args.upper_sample_seed,
        )
        upper_index = build_upper_index(
            train[upper_indices],
            upper_indices.astype(np.int64, copy=False),
            train.shape[1],
            args.upper_m,
            args.upper_ef_construction,
            100,
        )
        if args.recover_routing_from_collection:
            collection_cluster = collection_cluster_info(args.base_url, args.collection)
            cluster_summary = collection_cluster_summary(collection_cluster)
            effective_num_shards = int(cluster_summary["cluster_shard_count"] or args.num_shards)
            point_to_shards = recover_upper_point_to_shards_from_collection(
                args.base_url,
                args.collection,
                upper_indices,
                len(train),
                effective_num_shards,
            )
            info = collection_info(args.base_url, args.collection)
            l1_counts = np.zeros(effective_num_shards, dtype=np.int64)
            for point_id in upper_indices.tolist():
                for shard_id in point_to_shards[int(point_id)]:
                    if 0 <= int(shard_id) < effective_num_shards:
                        l1_counts[int(shard_id)] += 1
            build_row = {
                "collection": args.collection,
                "points_count": int(info.get("points_count") or 0),
                "indexed_vectors_count": int(info.get("indexed_vectors_count") or 0),
                "segments_count": int(info.get("segments_count") or 0),
                "shard_placement": args.shard_placement,
                "discovered_peer_count": len(peers),
                "logical_points_count": int(len(train)),
                "assigned_points_count": int(info.get("points_count") or 0),
                **cluster_summary,
            }
            sample_rows = [
                {
                    "shard_id": shard_id,
                    "shard_key": shard_key_for_id(shard_id),
                    "points_count": "",
                    "sample_count": int(l1_counts[shard_id]),
                }
                for shard_id in range(effective_num_shards)
            ]
        else:
            point_to_l1s = compute_point_to_l1s(
                upper_index,
                train,
                args.k_overlap,
                args.upper_build_batch_size,
            )
            routing_state = build_original_routing_state(
                train,
                upper_indices,
                point_to_l1s,
                args.num_shards,
                args.kmeans_iters,
                args.kmeans_rand_seed,
                args.topology_iters,
                use_multi_assign=not args.disable_multi_assign,
                enable_fission=not args.disable_fission,
            )
            point_to_shards = routing_state.point_to_shards
            effective_num_shards = routing_state.num_shards
            build_row = ensure_collection_from_point_shards(
                args.base_url,
                args.collection,
                train,
                point_to_shards,
                routing_state.upper_indices,
                effective_num_shards,
                args.hnsw_m,
                args.ef_construct,
                args.upload_batch_size,
                args.reuse_existing,
                args.shard_placement,
                peers,
            )
            sample_rows = original_upper_sample_rows(routing_state)
        upper_ids = upper_indices
        use_payload_source_id = True

    source_id_dedup_block_size = args.source_id_dedup_block_size
    if source_id_dedup_block_size is None and use_payload_source_id:
        source_id_dedup_block_size = len(train) + 1

    direct_peer_urls: dict[int, str] | None = None
    shard_key_to_peer: dict[str, int] | None = None
    if args.search_dispatch_mode == "direct_peer":
        direct_peer_urls = parse_peer_http_urls(args.direct_peer_http_urls)
        if not direct_peer_urls:
            raise ValueError("--search-dispatch-mode direct_peer requires --direct-peer-http-urls")
        shard_key_to_peer = collection_shard_key_to_peer(args.base_url, args.collection)
        missing_peer_urls = set(shard_key_to_peer.values()) - set(direct_peer_urls)
        if missing_peer_urls:
            raise ValueError(f"missing direct HTTP URLs for peers: {sorted(missing_peer_urls)}")

    tuning_rows: list[dict[str, Any]] = []
    if args.routing_mode == "naive_hash_all_shards":
        tuning_parameter_rows = [(0, base_ef, 0) for base_ef in args.base_ef_candidates]
    else:
        tuning_parameter_rows = [
            (upper_k, base_ef, factor)
            for upper_k in args.upper_k_candidates
            for base_ef in args.base_ef_candidates
            for factor in args.factor_candidates
        ]

    for upper_k, base_ef, factor in tuning_parameter_rows:
        if args.routing_mode == "naive_hash_all_shards":
            if args.search_dispatch_mode == "direct_peer":
                result = evaluate_all_shards_config_concurrent(
                    args.base_url,
                    args.collection,
                    tuning_queries,
                    tuning_neighbors,
                    args.top_k,
                    effective_num_shards,
                    base_ef,
                    args.batch_size,
                    1,
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    direct_peer_urls=direct_peer_urls,
                    shard_key_to_peer=shard_key_to_peer,
                )
            else:
                result = evaluate_all_shards_config(
                    args.base_url,
                    args.collection,
                    tuning_queries,
                    tuning_neighbors,
                    args.top_k,
                    effective_num_shards,
                    base_ef,
                    args.batch_size,
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                )
        else:
            assert upper_index is not None
            if args.search_dispatch_mode == "direct_peer":
                result = evaluate_config_concurrent(
                    args.base_url,
                    args.collection,
                    tuning_queries,
                    tuning_neighbors,
                    args.top_k,
                    upper_index,
                    upper_k,
                    base_ef,
                    factor,
                    args.batch_size,
                    1,
                    label_to_shard=label_to_shard,
                    point_to_shards=point_to_shards,
                    num_shards=effective_num_shards,
                    search_all_shards=args.search_all_shards,
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    routed_execution_mode=args.routed_execution_mode,
                    compact_ef_mode=args.compact_ef_mode,
                    routed_result_limit_mode=args.routed_result_limit_mode,
                    routed_result_limit_multiplier=args.routed_result_limit_multiplier,
                    direct_peer_urls=direct_peer_urls,
                    shard_key_to_peer=shard_key_to_peer,
                )
            else:
                routed_evaluator = (
                    evaluate_config_materialized_routing
                    if args.routed_planning_mode == "materialized"
                    else evaluate_config
                )
                result = routed_evaluator(
                    args.base_url,
                    args.collection,
                    tuning_queries,
                    tuning_neighbors,
                    args.top_k,
                    upper_index,
                    upper_k,
                    base_ef,
                    factor,
                    args.batch_size,
                    label_to_shard=label_to_shard,
                    point_to_shards=point_to_shards,
                    num_shards=effective_num_shards,
                    search_all_shards=args.search_all_shards,
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    routed_execution_mode=args.routed_execution_mode,
                    compact_ef_mode=args.compact_ef_mode,
                    routed_result_limit_mode=args.routed_result_limit_mode,
                    routed_result_limit_multiplier=args.routed_result_limit_multiplier,
                )
        row = {
            "upper_k": upper_k,
            "base_ef": base_ef,
            "factor": factor,
            "query_count": args.tuning_query_count,
            "top_k": args.top_k,
            "recall_at_k": result["recall_at_k"],
            "qps": result["qps"],
            "wall_s": result["wall_s"],
            "avg_visited_shards": result["avg_visited_shards"],
            "avg_upper_hits": result["avg_upper_hits"],
            "avg_assigned_ef_per_visited_shard": result["avg_assigned_ef_per_visited_shard"],
            "search_batch_calls": result["search_batch_calls"],
            "avg_search_requests_per_query": result["avg_search_requests_per_query"],
        }
        tuning_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    best = choose_best_matched_recall(tuning_rows, args.target_recall)

    if args.routing_mode == "naive_hash_all_shards":
        if args.search_dispatch_mode == "direct_peer":
            final_result = evaluate_all_shards_config_concurrent(
                args.base_url,
                args.collection,
                eval_queries,
                eval_neighbors,
                args.top_k,
                effective_num_shards,
                int(best["base_ef"]),
                args.batch_size,
                1,
                use_payload_source_id=use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
                direct_peer_urls=direct_peer_urls,
                shard_key_to_peer=shard_key_to_peer,
            )
        else:
            final_result = evaluate_all_shards_config(
                args.base_url,
                args.collection,
                eval_queries,
                eval_neighbors,
                args.top_k,
                effective_num_shards,
                int(best["base_ef"]),
                args.batch_size,
                use_payload_source_id=use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
            )
    else:
        assert upper_index is not None
        if args.search_dispatch_mode == "direct_peer":
            final_result = evaluate_config_concurrent(
                args.base_url,
                args.collection,
                eval_queries,
                eval_neighbors,
                args.top_k,
                upper_index,
                int(best["upper_k"]),
                int(best["base_ef"]),
                int(best["factor"]),
                args.batch_size,
                1,
                label_to_shard=label_to_shard,
                point_to_shards=point_to_shards,
                num_shards=effective_num_shards,
                search_all_shards=args.search_all_shards,
                use_payload_source_id=use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
                routed_execution_mode=args.routed_execution_mode,
                compact_ef_mode=args.compact_ef_mode,
                routed_result_limit_mode=args.routed_result_limit_mode,
                routed_result_limit_multiplier=args.routed_result_limit_multiplier,
                direct_peer_urls=direct_peer_urls,
                shard_key_to_peer=shard_key_to_peer,
            )
        else:
            routed_evaluator = (
                evaluate_config_materialized_routing
                if args.routed_planning_mode == "materialized"
                else evaluate_config
            )
            final_result = routed_evaluator(
                args.base_url,
                args.collection,
                eval_queries,
                eval_neighbors,
                args.top_k,
                upper_index,
                int(best["upper_k"]),
                int(best["base_ef"]),
                int(best["factor"]),
                args.batch_size,
                label_to_shard=label_to_shard,
                point_to_shards=point_to_shards,
                num_shards=effective_num_shards,
                search_all_shards=args.search_all_shards,
                use_payload_source_id=use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
                routed_execution_mode=args.routed_execution_mode,
                compact_ef_mode=args.compact_ef_mode,
                routed_result_limit_mode=args.routed_result_limit_mode,
                routed_result_limit_multiplier=args.routed_result_limit_multiplier,
            )
    final_row = {
        "upper_k": int(best["upper_k"]),
        "base_ef": int(best["base_ef"]),
        "factor": int(best["factor"]),
        "query_count": args.eval_query_count,
        "top_k": args.top_k,
        "recall_at_k": final_result["recall_at_k"],
        "qps": final_result["qps"],
        "wall_s": final_result["wall_s"],
        "avg_visited_shards": final_result["avg_visited_shards"],
        "avg_upper_hits": final_result["avg_upper_hits"],
        "avg_assigned_ef_per_visited_shard": final_result["avg_assigned_ef_per_visited_shard"],
        "search_batch_calls": final_result["search_batch_calls"],
        "avg_search_requests_per_query": final_result["avg_search_requests_per_query"],
    }

    stability_rows: list[dict[str, Any]] = []
    for run_idx in range(1, args.stability_repeats + 1):
        if args.routing_mode == "naive_hash_all_shards":
            if args.search_dispatch_mode == "direct_peer":
                result = evaluate_all_shards_config_concurrent(
                    args.base_url,
                    args.collection,
                    eval_queries,
                    eval_neighbors,
                    args.top_k,
                    effective_num_shards,
                    int(best["base_ef"]),
                    args.batch_size,
                    1,
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    direct_peer_urls=direct_peer_urls,
                    shard_key_to_peer=shard_key_to_peer,
                )
            else:
                result = evaluate_all_shards_config(
                    args.base_url,
                    args.collection,
                    eval_queries,
                    eval_neighbors,
                    args.top_k,
                    effective_num_shards,
                    int(best["base_ef"]),
                    args.batch_size,
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                )
        else:
            assert upper_index is not None
            if args.search_dispatch_mode == "direct_peer":
                result = evaluate_config_concurrent(
                    args.base_url,
                    args.collection,
                    eval_queries,
                    eval_neighbors,
                    args.top_k,
                    upper_index,
                    int(best["upper_k"]),
                    int(best["base_ef"]),
                    int(best["factor"]),
                    args.batch_size,
                    1,
                    label_to_shard=label_to_shard,
                    point_to_shards=point_to_shards,
                    num_shards=effective_num_shards,
                    search_all_shards=args.search_all_shards,
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    routed_execution_mode=args.routed_execution_mode,
                    compact_ef_mode=args.compact_ef_mode,
                    routed_result_limit_mode=args.routed_result_limit_mode,
                    routed_result_limit_multiplier=args.routed_result_limit_multiplier,
                    direct_peer_urls=direct_peer_urls,
                    shard_key_to_peer=shard_key_to_peer,
                )
            else:
                routed_evaluator = (
                    evaluate_config_materialized_routing
                    if args.routed_planning_mode == "materialized"
                    else evaluate_config
                )
                result = routed_evaluator(
                    args.base_url,
                    args.collection,
                    eval_queries,
                    eval_neighbors,
                    args.top_k,
                    upper_index,
                    int(best["upper_k"]),
                    int(best["base_ef"]),
                    int(best["factor"]),
                    args.batch_size,
                    label_to_shard=label_to_shard,
                    point_to_shards=point_to_shards,
                    num_shards=effective_num_shards,
                    search_all_shards=args.search_all_shards,
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    routed_execution_mode=args.routed_execution_mode,
                    compact_ef_mode=args.compact_ef_mode,
                    routed_result_limit_mode=args.routed_result_limit_mode,
                    routed_result_limit_multiplier=args.routed_result_limit_multiplier,
                )
        row = {
            "run": run_idx,
            "upper_k": int(best["upper_k"]),
            "base_ef": int(best["base_ef"]),
            "factor": int(best["factor"]),
            "query_count": args.eval_query_count,
            "top_k": args.top_k,
            "recall_at_k": result["recall_at_k"],
            "qps": result["qps"],
            "wall_s": result["wall_s"],
            "avg_visited_shards": result["avg_visited_shards"],
            "avg_upper_hits": result["avg_upper_hits"],
            "avg_assigned_ef_per_visited_shard": result["avg_assigned_ef_per_visited_shard"],
            "search_batch_calls": result["search_batch_calls"],
            "avg_search_requests_per_query": result["avg_search_requests_per_query"],
        }
        stability_rows.append(row)

    concurrency_rows: list[dict[str, Any]] = []
    concurrency_plan_wall_s: float | None = None
    if args.concurrency_candidates:
        query_plans: list[dict[str, Any]] | None = None
        if args.concurrency_evaluation_mode == "preplanned_search":
            plan_start = time.perf_counter()
            if args.routing_mode == "naive_hash_all_shards":
                query_plans = build_all_shard_search_plans(
                    eval_queries,
                    args.top_k,
                    effective_num_shards,
                    int(best["base_ef"]),
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                )
            else:
                assert upper_index is not None
                if point_to_shards is not None:
                    upper_labels = compute_upper_labels(upper_index, eval_queries, int(best["upper_k"]))
                    query_plans = build_routed_search_plans(
                        eval_queries,
                        upper_labels,
                        point_to_shards,
                        effective_num_shards,
                        args.top_k,
                        int(best["base_ef"]),
                        int(best["factor"]),
                        args.search_all_shards,
                        use_payload_source_id=use_payload_source_id,
                        routed_execution_mode=args.routed_execution_mode,
                        compact_ef_mode=args.compact_ef_mode,
                        routed_result_limit_mode=args.routed_result_limit_mode,
                        routed_result_limit_multiplier=args.routed_result_limit_multiplier,
                        source_id_dedup_block_size=source_id_dedup_block_size,
                    )
                else:
                    assert label_to_shard is not None
                    upper_labels = compute_upper_labels(upper_index, eval_queries, int(best["upper_k"]))
                    query_plans = [
                        legacy_routed_search_plan(
                            query.tolist(),
                            labels_row,
                            label_to_shard,
                            args.top_k,
                            int(best["base_ef"]),
                            int(best["factor"]),
                            use_payload_source_id,
                            args.routed_execution_mode,
                            args.compact_ef_mode,
                            args.routed_result_limit_mode,
                            args.routed_result_limit_multiplier,
                            source_id_dedup_block_size,
                        )
                        for query, labels_row in zip(eval_queries, upper_labels)
                    ]
            concurrency_plan_wall_s = time.perf_counter() - plan_start

        for concurrency in args.concurrency_candidates:
            if args.concurrency_evaluation_mode == "preplanned_search":
                assert query_plans is not None
                result = evaluate_preplanned_searches_concurrent(
                    args.base_url,
                    args.collection,
                    query_plans,
                    eval_neighbors,
                    args.top_k,
                    args.batch_size,
                    int(concurrency),
                    direct_peer_urls=direct_peer_urls,
                    shard_key_to_peer=shard_key_to_peer,
                )
            elif args.routing_mode == "naive_hash_all_shards":
                result = evaluate_all_shards_config_concurrent(
                    args.base_url,
                    args.collection,
                    eval_queries,
                    eval_neighbors,
                    args.top_k,
                    effective_num_shards,
                    int(best["base_ef"]),
                    args.batch_size,
                    int(concurrency),
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    direct_peer_urls=direct_peer_urls,
                    shard_key_to_peer=shard_key_to_peer,
                )
            else:
                assert upper_index is not None
                result = evaluate_config_concurrent(
                    args.base_url,
                    args.collection,
                    eval_queries,
                    eval_neighbors,
                    args.top_k,
                    upper_index,
                    int(best["upper_k"]),
                    int(best["base_ef"]),
                    int(best["factor"]),
                    args.batch_size,
                    int(concurrency),
                    label_to_shard=label_to_shard,
                    point_to_shards=point_to_shards,
                    num_shards=effective_num_shards,
                    search_all_shards=args.search_all_shards,
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    routed_execution_mode=args.routed_execution_mode,
                    compact_ef_mode=args.compact_ef_mode,
                    routed_result_limit_mode=args.routed_result_limit_mode,
                    routed_result_limit_multiplier=args.routed_result_limit_multiplier,
                    direct_peer_urls=direct_peer_urls,
                    shard_key_to_peer=shard_key_to_peer,
                )
            row = {
                "concurrency": int(concurrency),
                "concurrency_evaluation_mode": args.concurrency_evaluation_mode,
                "search_dispatch_mode": args.search_dispatch_mode,
                "upper_k": int(best["upper_k"]),
                "base_ef": int(best["base_ef"]),
                "factor": int(best["factor"]),
                "query_count": args.eval_query_count,
                "top_k": args.top_k,
                "recall_at_k": result["recall_at_k"],
                "qps": result["qps"],
                "wall_s": result["wall_s"],
                "avg_visited_shards": result["avg_visited_shards"],
                "avg_upper_hits": result["avg_upper_hits"],
                "avg_assigned_ef_per_visited_shard": result["avg_assigned_ef_per_visited_shard"],
                "search_batch_calls": result["search_batch_calls"],
                "avg_search_requests_per_query": result["avg_search_requests_per_query"],
                "routing_plan_wall_s_excluded_from_qps": concurrency_plan_wall_s,
            }
            concurrency_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    write_csv(output_dir / "builds.csv", [build_row])
    write_csv(output_dir / "upper_sample_stats.csv", sample_rows)
    write_csv(output_dir / "routing_tuning.csv", tuning_rows)
    write_csv(output_dir / "final_metrics.csv", [final_row])
    write_csv(output_dir / "stability_runs.csv", stability_rows)
    write_csv(output_dir / "concurrency_runs.csv", concurrency_rows)

    summary = {
        "base_url": args.base_url,
        "collection": args.collection,
        "hdf5_path": args.hdf5_path,
        "train_limit": args.train_limit,
        "routing_mode": args.routing_mode,
        "initial_num_shards": args.num_shards,
        "num_shards": effective_num_shards,
        "shard_placement": args.shard_placement,
        "placement_peer_uri_contains": args.placement_peer_uri_contains,
        "discovered_peer_ids": peers,
        "collection_cluster": collection_cluster_summary(collection_cluster_info(args.base_url, args.collection)),
        "sample_denominator": args.sample_denominator,
        "k_overlap": args.k_overlap,
        "search_all_shards": args.search_all_shards,
        "concurrency_evaluation_mode": args.concurrency_evaluation_mode,
        "search_dispatch_mode": args.search_dispatch_mode,
        "direct_peer_http_urls": args.direct_peer_http_urls if args.search_dispatch_mode == "direct_peer" else None,
        "routed_execution_mode": args.routed_execution_mode,
        "routed_planning_mode": args.routed_planning_mode,
        "compact_ef_mode": args.compact_ef_mode if args.routed_execution_mode == "compact_query_ef" else None,
        "routed_result_limit_mode": args.routed_result_limit_mode,
        "routed_result_limit_multiplier": (
            args.routed_result_limit_multiplier if args.routed_result_limit_mode == "fixed_multiplier" else None
        ),
        "source_id_dedup_block_size": source_id_dedup_block_size,
        "multi_assign": (not args.disable_multi_assign) if args.routing_mode == "faithful_original_rest" else False,
        "fission_enabled": (not args.disable_fission) if args.routing_mode == "faithful_original_rest" else False,
        "fair_architecture_note": (
            "faithful_original_rest and naive_hash_all_shards both use Qdrant custom shard keys, "
            "the same Docker cluster shape, the same batch search endpoint, and the same client "
            "result merge path. The intended algorithmic difference is shard selection: idea routes "
            "to upper-selected shards, while naive_hash_all_shards searches every shard."
            if args.routing_mode in {"faithful_original_rest", "naive_hash_all_shards"}
            else None
        ),
        "qdrant_rest_equivalence_gap": (
            "This patched Qdrant build accepts per-shard HNSW entry point IDs and "
            "per-shard dynamic EF, and lower-tier custom-entry searches use a "
            "base-layer MultiEP path matching hnswlib searchBaseLayerST_MultiEP. "
            "The remaining non-bit-identical gap is graph construction: bottom "
            "shards are still built by Qdrant's HNSW builder, so level RNG, "
            "parallel insertion scheduling, and neighbor-link tie behavior are "
            "not guaranteed to be byte-for-byte identical to hnswlib."
            if args.routing_mode == "faithful_original_rest"
            else None
        ),
        "rng_equivalence_note": (
            "The port preserves the original seed roles and algorithm order "
            "(upper sample seed 100, KMeans random initialization stream, weighted "
            "balanced assignment), but Python/NumPy RNG streams are not bit-identical "
            "to libstdc++ std::mt19937/std::shuffle or C rand()."
            if args.routing_mode == "faithful_original_rest"
            else None
        ),
        "upper_hnsw": {
            "m": args.upper_m,
            "ef_construction": args.upper_ef_construction,
            "search_ef": args.upper_search_ef,
            "sample_points": int(len(upper_ids)),
        },
        "original_routing": (
            {
                "topology_iterations": routing_state.topology_iterations,
                "total_assigned": routing_state.total_assigned,
                "expansion_ratio": routing_state.expansion_ratio,
                "fission_events": routing_state.fission_events,
                "shard_count_min": int(np.min(routing_state.shard_counts)),
                "shard_count_max": int(np.max(routing_state.shard_counts)),
            }
            if routing_state is not None
            else None
        ),
        "best_tuning_row": best,
        "final_row": final_row,
        "stability_summary": {
            "runs": len(stability_rows),
            "qps_mean": float(np.mean([row["qps"] for row in stability_rows])) if stability_rows else None,
            "qps_stdev": float(np.std([row["qps"] for row in stability_rows], ddof=1)) if len(stability_rows) > 1 else 0.0,
            "recall_mean": float(np.mean([row["recall_at_k"] for row in stability_rows])) if stability_rows else None,
            "recall_stdev": float(np.std([row["recall_at_k"] for row in stability_rows], ddof=1)) if len(stability_rows) > 1 else 0.0,
        },
        "concurrency_rows": concurrency_rows,
        "concurrency_note": (
            None
            if not concurrency_rows
            else (
                "Optional concurrency rows precompute per-query routing/search plans before timing, "
                "then measure concurrent REST batch dispatch and result merge. The timed QPS therefore "
                "isolates Docker/Qdrant search pressure and excludes routing_plan_wall_s_excluded_from_qps."
                if args.concurrency_evaluation_mode == "preplanned_search"
                else (
                    "Optional concurrency rows include per-batch upper routing, search-plan construction, "
                    "REST batch dispatch, and result merge in the timed QPS path."
                )
            )
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote results to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
