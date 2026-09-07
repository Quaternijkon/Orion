#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import queue
import re
import shlex
import socket
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from math import ceil, floor
from typing import Any

import h5py
import numpy as np


ORION_COLLECTION_METADATA_KEY = "orion_harness"
ORION_ROUTING_BUILD_METADATA_SCHEMA_VERSION = 2
RUNTIME_LOG_ERROR_PATTERNS = {
    "too_many_open_files": re.compile(r"too many open files", re.IGNORECASE),
    "peer_transport_failure": re.compile(
        r"failed to (?:send message|connect) to https?://", re.IGNORECASE
    ),
    "peer_unhealthy": re.compile(
        r"(?:unhealthy\s+peer|peer\b.*\bunhealthy)", re.IGNORECASE
    ),
    "shard_transfer_failure": re.compile(
        r"(?:shard transfer.*(?:fail|error)|(?:fail|error).*shard transfer)",
        re.IGNORECASE,
    ),
}


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
        "--vector-distance",
        choices=["cosine", "euclid", "l2"],
        default="cosine",
        help=(
            "Vector distance used for query vectors, the upper hnswlib index, "
            "and Qdrant collection creation. cosine preserves the existing "
            "GloVe/angular path and normalizes vectors; euclid/l2 leaves vectors "
            "un-normalized and uses Euclid/L2 search semantics."
        ),
    )
    parser.add_argument(
        "--routing-mode",
        choices=[
            "faithful_original_rest",
            "cpp_kmeans_baseline",
            "kmeans_simple_nprobe",
            "legacy_centroid",
            "naive_hash_all_shards",
        ],
        default="faithful_original_rest",
    )
    parser.add_argument("--num-shards", type=int, default=31)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--cpp-kmeans-train-size", type=int, default=10000)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--upload-batch-size", type=int, default=1024)
    parser.add_argument("--upload-concurrency", type=int, default=16)
    parser.add_argument("--upload-queue-capacity-batches", type=int, default=4)
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
        "--nprobe-candidates",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Number of KMeans centroid shards to probe for kmeans_simple_nprobe. "
            "Defaults to --upper-k-candidates when omitted so existing matrix "
            "summary columns can still store the probe count as upper_k."
        ),
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
    parser.add_argument(
        "--warmup-query-count",
        type=int,
        default=0,
        help="Run this fixed test-query prefix before each timed parameter point; excluded from QPS.",
    )
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
        "--lower-execution-order",
        choices=["query_major", "shard_major"],
        default="query_major",
        help=(
            "query_major preserves the original client request order. shard_major expands "
            "multi-shard lower searches into single-shard searches and orders each HTTP "
            "batch by shard key, so Qdrant can execute one contiguous shard-local batch "
            "per lower shard."
        ),
    )
    parser.add_argument(
        "--fixed-ef-shard-chunk-size",
        type=int,
        default=0,
        help=(
            "Explicit transport-only chunking for fixed-EF Naive/Simple searches. "
            "0 preserves one search object per query; a positive value partitions "
            "the selected shard keys into disjoint search objects of at most this "
            "size. Use 32 as the initial distributed experiment setting."
        ),
    )
    parser.add_argument(
        "--direct-peer-http-urls",
        nargs="+",
        default=None,
        help="Peer HTTP endpoints for direct_peer mode, as PEER_ID=URL entries.",
    )
    parser.add_argument(
        "--direct-peer-local-premerge",
        action="store_true",
        help=(
            "With direct_peer dispatch, merge each physical peer's local shard results "
            "per query before the final client merge. This simulates the result shape "
            "of a future worker-local pre-merge RPC without changing lower HNSW calls."
        ),
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
        choices=["per_batch", "materialized", "compact_materialized", "pipelined"],
        default="per_batch",
        help=(
            "per_batch preserves the original benchmark implementation and builds route "
            "plans inside each search batch. materialized computes upper routing and "
            "route plans once for the evaluation set inside the timed path, then reuses "
            "the same method4 plans for batched lower-tier searches. compact_materialized "
            "uses a compact CSR routing manifest and pre-encoded entry point IDs before "
            "executing the same compact MultiEP search requests. pipelined preserves the "
            "materialized route and search semantics while overlapping planning for batch "
            "N+1 on the benchmark-client CPU set with distributed lower search for batch N."
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
    parser.add_argument(
        "--orion-multi-assign-min-max-vote",
        type=int,
        default=2,
        help=(
            "For faithful_original_rest, enable multi-assignment only when the "
            "highest shard vote count is at least this value. The default 2 "
            "preserves the original max-vote-is-not-one behavior."
        ),
    )
    parser.add_argument(
        "--orion-multi-assign-vote-delta",
        type=int,
        default=0,
        help=(
            "For faithful_original_rest multi-assignment, include shards whose "
            "vote count is at least max_vote - delta. The default 0 assigns "
            "only top-vote ties; 1 assigns shards within one vote of the winner."
        ),
    )
    parser.add_argument(
        "--orion-multi-assign-max-shards",
        type=int,
        default=0,
        help=(
            "Optional cap on the number of Orion multi-assigned shards per point. "
            "0 means no cap. Shards are ranked by vote count descending, then id."
        ),
    )
    parser.add_argument(
        "--simple-kmeans-multi-assign-alpha",
        type=float,
        default=1.0,
        help=(
            "For kmeans_simple_nprobe collection construction, assign each point "
            "to every centroid whose Euclidean distance is <= alpha times the "
            "nearest-centroid distance. alpha <= 1.0 preserves the existing "
            "single-assignment collection path."
        ),
    )
    parser.add_argument(
        "--simple-kmeans-multi-assign-chunk-size",
        type=int,
        default=50000,
        help="Chunk size for computing simple KMeans alpha multi-assignment.",
    )
    parser.add_argument("--disable-fission", action="store_true")
    parser.add_argument(
        "--orion-balance-mode",
        choices=[
            "none",
            "capacity_constrained",
            "post_layout_capacity_constrained",
        ],
        default="none",
        help=(
            "Optional offline load balancing. capacity_constrained starts from a "
            "fixed P, constrains L1 membership refinement, balances final physical "
            "L0 copies, and disables fission. post_layout_capacity_constrained first "
            "runs the original Orion topology/fission/multi-assignment path, then "
            "freezes its final P and minimally repairs physical-copy placement. Both "
            "modes keep the production upper graph and online route path unchanged."
        ),
    )
    parser.add_argument("--orion-balance-min-load-ratio", type=float, default=0.99)
    parser.add_argument("--orion-balance-max-load-ratio", type=float, default=1.01)
    parser.add_argument("--orion-balance-max-passes", type=int, default=8)
    parser.add_argument(
        "--orion-balance-max-vote-loss",
        type=int,
        default=3,
        help=(
            "Shared default navigation-vote loss for L1 membership and L0 physical-copy "
            "placement. The stage-specific options override it."
        ),
    )
    parser.add_argument(
        "--orion-balance-l1-max-vote-loss",
        type=int,
        default=None,
        help="Optional L1 topology vote-loss override.",
    )
    parser.add_argument(
        "--orion-balance-l0-max-vote-loss",
        type=int,
        default=None,
        help=(
            "Optional L0 physical-copy vote-loss override. Multi-assignment vote delta "
            "remains an implicit floor."
        ),
    )
    parser.add_argument(
        "--claim-a-partition-family",
        choices=[
            "none",
            "random_balanced_46",
            "kmeans_topology_46",
            "kmeans_topology_load_recalibrated_46",
        ],
        default="none",
        help=(
            "Build one of the Claim A offline partition-family route maps as a live "
            "faithful_original_rest collection. These families intentionally disable "
            "fission; load-recalibrated currently matches kmeans_topology without fission."
        ),
    )
    parser.add_argument(
        "--claim-a-random-seed",
        type=int,
        default=12345,
        help="Seed for the Claim A random_balanced_46 L1-to-shard assignment.",
    )
    parser.add_argument("--search-all-shards", action="store_true")
    parser.add_argument(
        "--shard-placement",
        choices=["auto", "round_robin", "none", "map"],
        default="auto",
        help=(
            "Physical placement for custom shard keys. Does not change the "
            "KMeans/routing algorithm; it only selects which Qdrant peer owns "
            "each custom shard in cluster deployments."
        ),
    )
    parser.add_argument(
        "--shard-placement-map",
        default=None,
        help=(
            "JSON placement map used when --shard-placement=map. Accepts either "
            "a direct {shard_key: peer_id_or_ordinal} object or a placement "
            "simulation JSON containing placements[--shard-placement-map-name]."
        ),
    )
    parser.add_argument(
        "--shard-placement-map-name",
        default="method4_aware",
        help="Named placement to load from a placement simulation JSON file.",
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
        "--cluster-topology",
        default=None,
        help=(
            "Optional distributed topology JSON. When supplied, cluster preflight requires exactly "
            "the controller and three worker advertised URIs in that file, and placement uses only "
            "the three exact worker URIs."
        ),
    )
    parser.add_argument(
        "--deployment-manifest",
        default=None,
        help="Optional method4_distributed_cluster.py manifest copied into result provenance.",
    )
    parser.add_argument(
        "--require-clean-runtime",
        action="store_true",
        help=(
            "Reject a distributed benchmark result if its own runtime window contains "
            "file-descriptor, peer-transport, peer-health, or shard-transfer failures, "
            "or if the ending cluster/collection health checks are not clean."
        ),
    )
    parser.add_argument("--image-tag", default=None)
    parser.add_argument("--image-digest", default=None)
    parser.add_argument("--dataset-sha256", default=None)
    parser.add_argument(
        "--placement-simulation",
        action="store_true",
        help=(
            "For faithful routed materialized runs, compute a method4-aware physical "
            "placement simulation from the actual query routing trace and include it "
            "in summary.json. This does not change the deployed collection."
        ),
    )
    parser.add_argument(
        "--placement-simulation-peer-count",
        type=int,
        default=3,
        help="Number of bottom workers to use in the method4-aware placement simulation.",
    )
    parser.add_argument(
        "--physical-execution-trace",
        action="store_true",
        help=(
            "For faithful routed materialized runs, summarize how logical method4 shard "
            "plans map onto the actual physical Qdrant peers. This estimates the "
            "controller fan-in reduction available from worker-local pre-merge and "
            "does not change search behavior."
        ),
    )
    parser.add_argument(
        "--write-per-query-metrics",
        action="store_true",
        help=(
            "Write final_per_query_metrics.csv for the final eval run. This records "
            "per-query hits@k, recall@k, retrieved ids, and ground-truth ids for "
            "recall distribution analysis."
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


def effective_upper_search_ef(args: argparse.Namespace) -> int:
    candidate_ks = [int(value) for value in getattr(args, "upper_k_candidates", []) or []]
    nprobe_candidates = getattr(args, "nprobe_candidates", None)
    if nprobe_candidates:
        candidate_ks.extend(int(value) for value in nprobe_candidates)
    widest_requested_k = max(candidate_ks, default=0)
    return max(int(args.upper_search_ef), widest_requested_k)


def validate_args(args: argparse.Namespace) -> None:
    if int(getattr(args, "upload_concurrency", 16)) <= 0:
        raise ValueError("--upload-concurrency must be positive")
    if int(getattr(args, "upload_queue_capacity_batches", 4)) <= 0:
        raise ValueError("--upload-queue-capacity-batches must be positive")
    if int(args.orion_multi_assign_min_max_vote) <= 0:
        raise ValueError("--orion-multi-assign-min-max-vote must be positive")
    if int(args.orion_multi_assign_vote_delta) < 0:
        raise ValueError("--orion-multi-assign-vote-delta must be non-negative")
    if int(args.orion_multi_assign_max_shards) < 0:
        raise ValueError("--orion-multi-assign-max-shards must be non-negative")
    if float(args.simple_kmeans_multi_assign_alpha) < 1.0:
        raise ValueError("--simple-kmeans-multi-assign-alpha must be >= 1.0")
    if int(args.simple_kmeans_multi_assign_chunk_size) <= 0:
        raise ValueError("--simple-kmeans-multi-assign-chunk-size must be positive")
    if int(getattr(args, "warmup_query_count", 0)) < 0:
        raise ValueError("--warmup-query-count must be non-negative")
    if int(getattr(args, "fixed_ef_shard_chunk_size", 0)) < 0:
        raise ValueError("--fixed-ef-shard-chunk-size must be non-negative")
    validate_capacity_balance_parameters(
        float(getattr(args, "orion_balance_min_load_ratio", 0.99)),
        float(getattr(args, "orion_balance_max_load_ratio", 1.01)),
        int(getattr(args, "orion_balance_max_passes", 8)),
        int(getattr(args, "orion_balance_max_vote_loss", 3)),
    )
    l1_vote_loss, l0_vote_loss = resolve_capacity_balance_vote_losses(
        int(getattr(args, "orion_balance_max_vote_loss", 3)),
        getattr(args, "orion_balance_l1_max_vote_loss", None),
        getattr(args, "orion_balance_l0_max_vote_loss", None),
    )
    for stage_vote_loss in (l1_vote_loss, l0_vote_loss):
        validate_capacity_balance_parameters(
            float(getattr(args, "orion_balance_min_load_ratio", 0.99)),
            float(getattr(args, "orion_balance_max_load_ratio", 1.01)),
            int(getattr(args, "orion_balance_max_passes", 8)),
            stage_vote_loss,
        )
    if (
        str(getattr(args, "routing_mode", "")) == "faithful_original_rest"
        and str(getattr(args, "claim_a_partition_family", "none")) == "none"
    ):
        if bool(getattr(args, "disable_multi_assign", False)):
            raise ValueError("canonical Orion multi-assignment cannot be disabled")
        if int(getattr(args, "orion_multi_assign_min_max_vote", 2)) != 2:
            raise ValueError("canonical Orion fixes the multi-assignment threshold at max_vote > 1")
        if int(getattr(args, "orion_multi_assign_vote_delta", 0)) != 0:
            raise ValueError("canonical Orion assigns only exact maximum-vote shard ties")
        if int(getattr(args, "orion_multi_assign_max_shards", 0)) != 0:
            raise ValueError("canonical Orion cannot cap maximum-vote shard ties")
        if str(getattr(args, "orion_balance_mode", "none")) != "none":
            raise ValueError("canonical Orion currently excludes load-balancing intervention")


def vector_distance_config(vector_distance: str) -> dict[str, Any]:
    normalized = str(vector_distance).lower()
    if normalized == "cosine":
        return {
            "name": "cosine",
            "hnsw_space": "cosine",
            "qdrant_distance": "Cosine",
            "normalize_vectors": True,
            "score_higher_is_better": True,
        }
    if normalized in {"euclid", "l2"}:
        return {
            "name": "euclid",
            "hnsw_space": "l2",
            "qdrant_distance": "Euclid",
            "normalize_vectors": False,
            "score_higher_is_better": False,
        }
    raise ValueError(f"unsupported vector distance: {vector_distance}")


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
        raise RuntimeError(
            f"{method} {url} failed (HTTP {exc.code}): {exc.read().decode()}"
        ) from exc


def request_json_encoded(
    base_url: str,
    method: str,
    path: str,
    data: bytes,
    timeout: float = 300.0,
) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"{method} {url} failed (HTTP {exc.code}): {exc.read().decode()}"
        ) from exc


def encode_search_batch_body(searches: list[dict[str, Any]]) -> bytes:
    return json.dumps({"searches": searches}).encode()


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return arr / norms


def prepare_vectors_for_distance(arr: np.ndarray, vector_distance: str) -> np.ndarray:
    config = vector_distance_config(vector_distance)
    if config["normalize_vectors"]:
        return normalize_rows(arr)
    return arr


def slice_train_rows(rows: Any, train_limit: int | None) -> Any:
    if train_limit is None:
        return rows
    if train_limit <= 0:
        raise ValueError("train_limit must be positive")
    return rows[:train_limit]


def normalize_peer_uri(uri: str) -> str:
    return str(uri).rstrip("/")


def cluster_peer_map(base_url: str) -> tuple[int | None, dict[int, str], dict[str, Any]]:
    result = request_json(base_url, "GET", "/cluster")["result"]
    peer_id = int(result["peer_id"]) if result.get("peer_id") is not None else None
    peers = {
        int(raw_peer_id): normalize_peer_uri(str((peer_info or {}).get("uri") or ""))
        for raw_peer_id, peer_info in (result.get("peers") or {}).items()
    }
    return peer_id, peers, result


def cluster_peer_ids(
    base_url: str,
    uri_filters: list[str] | None = None,
    exact_uris: list[str] | None = None,
) -> list[int]:
    try:
        current_peer_id, peers, _result = cluster_peer_map(base_url)
    except RuntimeError:
        return []

    peer_ids: set[int] = set()
    exact = {normalize_peer_uri(uri) for uri in exact_uris or []}
    if current_peer_id is not None and not uri_filters and not exact:
        peer_ids.add(current_peer_id)
    for peer_id, uri in peers.items():
        if exact and uri not in exact:
            continue
        if uri_filters and not any(uri_filter in uri for uri_filter in uri_filters):
            continue
        peer_ids.add(peer_id)
    return sorted(peer_ids)


def load_cluster_topology(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    controller = data.get("controller")
    workers = data.get("workers")
    ports = data.get("ports") or {"http": 6333, "grpc": 6334, "p2p": 6335}
    if not isinstance(controller, dict) or not isinstance(workers, list) or len(workers) != 3:
        raise ValueError("cluster topology requires one controller and exactly three workers")
    nodes = [controller, *workers]
    ips = [str(node.get("private_ip") or "") for node in nodes]
    if len(set(ips)) != 4 or any(not ip for ip in ips):
        raise ValueError("cluster topology requires four unique private_ip values")
    p2p_port = int(ports.get("p2p", 6335))
    return {
        **data,
        "ports": ports,
        "controller_uri": f"http://{controller['private_ip']}:{p2p_port}",
        "worker_uris": [f"http://{worker['private_ip']}:{p2p_port}" for worker in workers],
    }


def validate_cluster_preflight(
    base_url: str,
    topology: dict[str, Any],
) -> dict[str, Any]:
    current_peer_id, peers, result = cluster_peer_map(base_url)
    expected_controller = normalize_peer_uri(topology["controller_uri"])
    expected_workers = [normalize_peer_uri(uri) for uri in topology["worker_uris"]]
    expected = {expected_controller, *expected_workers}
    actual = set(peers.values())
    errors: list[str] = []
    if len(peers) != 4:
        errors.append(f"expected exactly 4 peers, found {len(peers)}")
    if actual != expected:
        errors.append(f"expected peer URIs {sorted(expected)}, found {sorted(actual)}")
    if current_peer_id is None or peers.get(current_peer_id) != expected_controller:
        errors.append(
            f"benchmark endpoint must be controller {expected_controller}; current peer is "
            f"{current_peer_id}:{peers.get(current_peer_id)}"
        )
    worker_ids = sorted(peer_id for peer_id, uri in peers.items() if uri in expected_workers)
    if len(worker_ids) != 3:
        errors.append(f"expected 3 worker peer IDs, found {worker_ids}")
    raft_info = result.get("raft_info") or {}
    pending_operations = int(raft_info.get("pending_operations") or 0)
    if pending_operations != 0:
        errors.append(f"expected zero pending consensus operations, found {pending_operations}")
    consensus = result.get("consensus_thread_status") or {}
    consensus_status = str(consensus.get("consensus_thread_status") or "")
    if consensus_status != "working":
        errors.append(f"expected consensus thread working, found {consensus_status!r}")
    message_send_failures = result.get("message_send_failures") or {}
    if message_send_failures:
        errors.append(f"cluster reports peer message send failures: {message_send_failures}")
    if errors:
        raise RuntimeError("distributed cluster preflight failed: " + "; ".join(errors))
    return {
        "peer_id": current_peer_id,
        "peer_count": len(peers),
        "peers": {str(peer_id): uri for peer_id, uri in sorted(peers.items())},
        "controller_peer_id": current_peer_id,
        "worker_peer_ids": worker_ids,
        "pending_operations": pending_operations,
        "consensus_thread_status": consensus_status,
        "message_send_failures": message_send_failures,
        "raw": result,
    }


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
    placement_map: dict[str, int] | None = None,
) -> list[int] | None:
    if mode == "none":
        return None
    if mode == "auto" and len(peer_ids) <= 1:
        return None
    if mode == "map":
        if placement_map is None:
            raise ValueError("--shard-placement map requires --shard-placement-map")
        shard_key = shard_key_for_id(shard_index)
        if shard_key not in placement_map:
            raise ValueError(f"missing placement map entry for {shard_key}")
        peer_value = int(placement_map[shard_key])
        if peer_value in peer_ids:
            return [peer_value]
        if 0 <= peer_value < len(peer_ids):
            return [int(peer_ids[peer_value])]
        raise ValueError(
            f"placement map value for {shard_key} must be a peer id or peer ordinal; got {peer_value}"
        )
    if mode not in {"auto", "round_robin"}:
        raise ValueError(f"unsupported shard placement mode: {mode}")
    if not peer_ids:
        return None
    return [int(peer_ids[shard_index % len(peer_ids)])]


def load_shard_placement_map(path: str | Path, map_name: str) -> dict[str, int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "placements" in data:
        placements = data["placements"]
        if map_name not in placements:
            raise ValueError(f"placement map {map_name!r} not found in {path}")
        data = placements[map_name]
    if not isinstance(data, dict):
        raise ValueError(f"placement map in {path} must be a JSON object")
    result: dict[str, int] = {}
    for shard_key, peer_value in data.items():
        result[str(shard_key)] = int(peer_value)
    return result


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
        claim_a_partition_family: str | None = None,
        claim_a_partition_note: str | None = None,
        balance_diagnostics: dict[str, Any] | None = None,
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
        self.claim_a_partition_family = claim_a_partition_family
        self.claim_a_partition_note = claim_a_partition_note
        self.balance_diagnostics = balance_diagnostics


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
    max_load_ratio: float = 1.5,
) -> np.ndarray:
    indices_arr = np.asarray(indices, dtype=np.int64)
    weights = np.asarray(point_weights, dtype=np.int64)
    k = len(centroids)
    if len(indices_arr) != len(weights):
        raise ValueError("indices and point_weights must have the same length")
    if k <= 0:
        raise ValueError("centroids must not be empty")
    if float(max_load_ratio) < 1.0:
        raise ValueError("max_load_ratio must be >= 1")

    distances = squared_l2_distances(data[indices_arr].astype(np.float32, copy=False), centroids)
    assignments = np.full(len(indices_arr), -1, dtype=np.int32)
    cluster_weight = np.zeros(k, dtype=np.int64)
    target_weight = int(np.sum(weights) // k + 1)
    max_cluster_weight = target_weight * float(max_load_ratio)

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
    # Canonical Orion uses one build-time budget: efsearch == number of
    # returned/voting results.  Runtime upper-search tuning is restored by the
    # caller after the immutable partition/layout has been built.
    set_ef = getattr(upper_index, "set_ef", None)
    if callable(set_ef):
        set_ef(k)
    for start in range(0, len(train), batch_size):
        labels, distances = upper_index.knn_query(
            train[start : start + batch_size],
            k=k,
        )
        for label_row, distance_row in zip(
            labels.tolist(),
            distances.tolist(),
            strict=True,
        ):
            ranked = sorted(
                zip(distance_row, label_row, strict=True),
                key=lambda item: (float(item[0]), int(item[1])),
            )
            labels_rows.append([int(label) for _distance, label in ranked])
    return labels_rows


def initial_l1_shards_by_balanced_kmeans(
    train: np.ndarray,
    upper_indices: np.ndarray,
    up_tier_weights: np.ndarray,
    num_shards: int,
    kmeans_iters: int,
    seed: int,
    max_load_ratio: float = 1.5,
) -> list[int]:
    centroids = cpp_style_kmeans_train(
        train,
        upper_indices,
        num_shards,
        max_iter=kmeans_iters,
        seed=seed,
    )
    assignments = cpp_style_predict_balanced(
        train,
        upper_indices,
        up_tier_weights,
        centroids,
        max_load_ratio=max_load_ratio,
    )
    l1_to_shard = [-1] * len(train)
    for point_id, shard_id in zip(upper_indices.tolist(), assignments.tolist()):
        l1_to_shard[int(point_id)] = int(shard_id)
    return l1_to_shard


def initial_l1_shards_by_kmeans(
    train: np.ndarray,
    upper_indices: np.ndarray,
    num_shards: int,
    kmeans_iters: int,
    seed: int,
) -> list[int]:
    """Cluster upper-navigation points without any capacity/load correction."""
    centroids = cpp_style_kmeans_train(
        train,
        upper_indices,
        num_shards,
        max_iter=kmeans_iters,
        seed=seed,
    )
    upper_points = train[np.asarray(upper_indices, dtype=np.int64)].astype(
        np.float32,
        copy=False,
    )
    assignments = np.argmin(
        squared_l2_distances(upper_points, centroids),
        axis=1,
    )
    l1_to_shard = [-1] * len(train)
    for point_id, shard_id in zip(
        upper_indices.tolist(),
        assignments.tolist(),
        strict=True,
    ):
        l1_to_shard[int(point_id)] = int(shard_id)
    return l1_to_shard


def self_vote_target_shard(
    l1s: list[int],
    reference_l1_shard: list[int],
) -> int:
    """Choose one deterministic self-vote winner using nearest-hit tie-break."""
    ordered_shards = [
        int(reference_l1_shard[int(l1_idx)])
        for l1_idx in l1s
        if 0 <= int(l1_idx) < len(reference_l1_shard)
        and int(reference_l1_shard[int(l1_idx)]) >= 0
    ]
    if not ordered_shards:
        raise RuntimeError("upper self-search returned no shard-labelled result")
    votes = Counter(ordered_shards)
    max_vote = max(votes.values())
    winning_shards = {
        int(shard_id) for shard_id, vote in votes.items() if int(vote) == int(max_vote)
    }
    return next(shard_id for shard_id in ordered_shards if shard_id in winning_shards)


def converge_l1_topology_by_self_vote(
    point_to_l1s: list[list[int]],
    upper_indices: np.ndarray,
    l1_to_shard: list[int],
    max_iters: int,
) -> tuple[list[int], int]:
    """Refine clustered upper labels using only immutable navigation votes."""
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    current_shard = list(l1_to_shard)
    iteration_count = 0
    for iteration in range(max_iters):
        next_shard = list(current_shard)
        changed_count = 0
        for raw_l1_idx in upper_indices.tolist():
            l1_idx = int(raw_l1_idx)
            target_shard = self_vote_target_shard(
                point_to_l1s[l1_idx],
                current_shard,
            )
            if target_shard != int(current_shard[l1_idx]):
                next_shard[l1_idx] = target_shard
                changed_count += 1
        current_shard = next_shard
        iteration_count = iteration + 1
        if changed_count == 0:
            return current_shard, iteration_count

    unconverged = [
        int(l1_idx)
        for l1_idx in upper_indices.tolist()
        if self_vote_target_shard(point_to_l1s[int(l1_idx)], current_shard)
        != int(current_shard[int(l1_idx)])
    ]
    if unconverged:
        preview = ", ".join(str(point_id) for point_id in unconverged[:10])
        raise RuntimeError(
            "upper self-vote refinement did not converge within "
            f"{max_iters} iterations; first unstable points: {preview}"
        )
    return current_shard, iteration_count


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


def validate_capacity_balance_parameters(
    min_load_ratio: float,
    max_load_ratio: float,
    max_passes: int,
    max_vote_loss: int,
) -> None:
    if not 0.0 < float(min_load_ratio) <= 1.0:
        raise ValueError("balance min_load_ratio must be in (0, 1]")
    if float(max_load_ratio) < 1.0:
        raise ValueError("balance max_load_ratio must be >= 1")
    if float(min_load_ratio) > float(max_load_ratio):
        raise ValueError("balance min_load_ratio must not exceed max_load_ratio")
    if int(max_passes) <= 0:
        raise ValueError("balance max_passes must be positive")
    if int(max_vote_loss) < 0:
        raise ValueError("balance max_vote_loss must be non-negative")


def resolve_capacity_balance_vote_losses(
    shared_max_vote_loss: int,
    l1_max_vote_loss: int | None,
    l0_max_vote_loss: int | None,
) -> tuple[int, int]:
    shared = int(shared_max_vote_loss)
    return (
        shared if l1_max_vote_loss is None else int(l1_max_vote_loss),
        shared if l0_max_vote_loss is None else int(l0_max_vote_loss),
    )


def require_capacity_balance_bounds(
    stage: str,
    diagnostics: dict[str, Any] | None,
) -> None:
    """Reject incomplete or unsatisfied capacity proofs at build boundaries."""

    if not isinstance(diagnostics, dict):
        raise RuntimeError(f"capacity-constrained Orion is missing {stage} diagnostics")
    over_upper = diagnostics.get("over_upper_shards")
    under_lower = diagnostics.get("under_lower_shards")
    if not isinstance(over_upper, list) or not isinstance(under_lower, list):
        raise RuntimeError(
            f"capacity-constrained Orion {stage} diagnostics lack explicit residual lists"
        )
    if (
        diagnostics.get("bounds_satisfied") is not True
        or over_upper != []
        or under_lower != []
    ):
        raise RuntimeError(
            f"capacity-constrained Orion {stage} bounds are not satisfied: "
            f"bounds_satisfied={diagnostics.get('bounds_satisfied')!r}, "
            f"over_upper_shards={over_upper!r}, "
            f"under_lower_shards={under_lower!r}"
        )


def capacity_load_bounds(
    total_load: int,
    num_shards: int,
    min_load_ratio: float,
    max_load_ratio: float,
) -> tuple[float, int, int]:
    if int(total_load) < 0:
        raise ValueError("total_load must be non-negative")
    if int(num_shards) <= 0:
        raise ValueError("num_shards must be positive")
    ideal_load = float(total_load) / float(num_shards)
    lower_load = int(floor(ideal_load * float(min_load_ratio)))
    upper_load = int(ceil(ideal_load * float(max_load_ratio)))
    lower_load = min(lower_load, int(total_load // num_shards))
    upper_load = max(upper_load, int(ceil(ideal_load)))
    return ideal_load, lower_load, upper_load


def load_balance_summary(loads: np.ndarray | list[int]) -> dict[str, Any]:
    values = np.asarray(loads, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("loads must be a non-empty one-dimensional sequence")
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "loads": [int(value) for value in values.tolist()],
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean > 0.0 else 0.0,
        "max_over_mean": float(np.max(values) / mean) if mean > 0.0 else 0.0,
        "min_over_mean": float(np.min(values) / mean) if mean > 0.0 else 0.0,
    }


def navigation_shard_votes(
    l1s: list[int],
    reference_l1_shard: list[int],
) -> Counter[int]:
    votes: Counter[int] = Counter()
    for ep_l1 in l1s:
        ep_l1 = int(ep_l1)
        shard_id = reference_l1_shard[ep_l1] if ep_l1 < len(reference_l1_shard) else -1
        if shard_id != -1:
            votes[int(shard_id)] += 1
    return votes


def eligible_navigation_shards(
    votes: Counter[int],
    max_vote_loss: int,
) -> list[int]:
    if not votes:
        return []
    max_vote = max(votes.values())
    min_vote = max(1, int(max_vote) - int(max_vote_loss))
    return [
        int(shard_id)
        for shard_id, vote in sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        if int(vote) >= min_vote
    ]


def capacity_band_residual(
    loads: np.ndarray,
    lower_load: int,
    upper_load: int,
) -> dict[str, Any]:
    under_by_shard = [
        max(0, int(lower_load) - int(load)) for load in loads.tolist()
    ]
    over_by_shard = [
        max(0, int(load) - int(upper_load)) for load in loads.tolist()
    ]
    under_total = int(sum(under_by_shard))
    over_total = int(sum(over_by_shard))
    return {
        "under_lower_total": under_total,
        "over_upper_total": over_total,
        "total": int(under_total + over_total),
        "under_lower_shards": [
            int(shard_id)
            for shard_id, residual in enumerate(under_by_shard)
            if residual > 0
        ],
        "over_upper_shards": [
            int(shard_id)
            for shard_id, residual in enumerate(over_by_shard)
            if residual > 0
        ],
    }


def repair_l1_topology_with_weighted_swaps(
    point_to_l1s: list[list[int]],
    upper_rows: list[int],
    weights_by_l1: dict[int, int],
    current_shard: list[int],
    current_loads: np.ndarray,
    lower_load: int,
    upper_load: int,
    max_vote_loss: int,
    max_rounds: int,
) -> dict[str, Any]:
    """Repair indivisible L1 capacity residuals with atomic reciprocal swaps.

    A direct move can be impossible even when a feasible weighted assignment
    exists: the incoming node may overflow the target while moving it out would
    drop the source below its floor. A reciprocal two-node swap transfers only
    the weight difference and can therefore cross that granularity barrier.

    Both legs must independently target a shard supported by the pre-swap
    navigation votes and remain inside ``max_vote_loss``. Every accepted swap
    strictly decreases the total capacity-band residual, so the repair cannot
    oscillate. Candidate and tie ordering is deterministic.
    """

    before_residual = capacity_band_residual(current_loads, lower_load, upper_load)
    attempted = bool(before_residual["total"])
    swap_count = 0
    moved_l1_count = 0
    rounds = 0
    candidate_pair_count = 0
    navigation_vote_delta = 0
    maximum_observed_vote_loss = 0
    maximum_target_vote_loss = 0
    stop_reason = "greedy_satisfied_bounds"

    # (l1 id, source, target, weight, source vote, target vote, best vote)
    MoveOption = tuple[int, int, int, int, int, int, int]

    for round_index in (range(int(max_rounds)) if attempted else []):
        residual_before_round = capacity_band_residual(
            current_loads, lower_load, upper_load
        )
        if int(residual_before_round["total"]) == 0:
            stop_reason = "bounds_satisfied"
            break

        # Only one deterministic representative is needed for each weight on a
        # directed shard edge. If it is consumed, the next round will expose the
        # next-best node of the same weight.
        edge_options: dict[tuple[int, int], dict[int, MoveOption]] = defaultdict(dict)
        for l1_idx in upper_rows:
            source_shard = int(current_shard[l1_idx])
            if source_shard == -1:
                continue
            votes = navigation_shard_votes(point_to_l1s[l1_idx], current_shard)
            if not votes:
                continue
            source_vote = int(votes.get(source_shard, 0))
            best_vote = int(max(votes.values()))
            weight = int(weights_by_l1[l1_idx])
            for target_shard in eligible_navigation_shards(votes, max_vote_loss):
                target_shard = int(target_shard)
                if target_shard == source_shard:
                    continue
                target_vote = int(votes[target_shard])
                option: MoveOption = (
                    int(l1_idx),
                    source_shard,
                    target_shard,
                    weight,
                    source_vote,
                    target_vote,
                    best_vote,
                )
                by_weight = edge_options[(source_shard, target_shard)]
                previous = by_weight.get(weight)
                option_key = (
                    best_vote - target_vote,
                    max(0, source_vote - target_vote),
                    -target_vote,
                    int(l1_idx),
                )
                if previous is None:
                    by_weight[weight] = option
                else:
                    previous_key = (
                        previous[6] - previous[5],
                        max(0, previous[4] - previous[5]),
                        -previous[5],
                        previous[0],
                    )
                    if option_key < previous_key:
                        by_weight[weight] = option

        candidates: list[tuple[tuple[Any, ...], MoveOption, MoveOption]] = []
        global_residual = int(residual_before_round["total"])
        for source_shard in range(len(current_loads)):
            for target_shard in range(source_shard + 1, len(current_loads)):
                forward = edge_options.get((source_shard, target_shard))
                reverse = edge_options.get((target_shard, source_shard))
                if not forward or not reverse:
                    continue
                reverse_weights = sorted(reverse)
                load_source = int(current_loads[source_shard])
                load_target = int(current_loads[target_shard])
                before_pair_residual = int(
                    max(0, int(lower_load) - load_source)
                    + max(0, load_source - int(upper_load))
                    + max(0, int(lower_load) - load_target)
                    + max(0, load_target - int(upper_load))
                )
                # The pair residual is convex in transferred weight d=wa-wb.
                # A discrete minimum lies next to one of these band breakpoints.
                transfer_breakpoints = (
                    load_source - int(upper_load),
                    load_source - int(lower_load),
                    int(lower_load) - load_target,
                    int(upper_load) - load_target,
                )
                best_for_pair: tuple[
                    tuple[Any, ...], MoveOption, MoveOption
                ] | None = None
                for forward_weight, forward_option in sorted(forward.items()):
                    reverse_positions: set[int] = {0, len(reverse_weights) - 1}
                    for breakpoint in transfer_breakpoints:
                        desired_reverse_weight = int(forward_weight) - int(breakpoint)
                        position = bisect_left(reverse_weights, desired_reverse_weight)
                        reverse_positions.update((position - 1, position, position + 1))
                    for position in sorted(reverse_positions):
                        if position < 0 or position >= len(reverse_weights):
                            continue
                        reverse_weight = int(reverse_weights[position])
                        reverse_option = reverse[reverse_weight]
                        transferred = int(forward_weight) - reverse_weight
                        new_source_load = load_source - transferred
                        new_target_load = load_target + transferred
                        after_pair_residual = int(
                            max(0, int(lower_load) - new_source_load)
                            + max(0, new_source_load - int(upper_load))
                            + max(0, int(lower_load) - new_target_load)
                            + max(0, new_target_load - int(upper_load))
                        )
                        improvement = before_pair_residual - after_pair_residual
                        if improvement <= 0:
                            continue
                        forward_target_loss = int(forward_option[6] - forward_option[5])
                        reverse_target_loss = int(reverse_option[6] - reverse_option[5])
                        forward_actual_loss = max(
                            0, int(forward_option[4] - forward_option[5])
                        )
                        reverse_actual_loss = max(
                            0, int(reverse_option[4] - reverse_option[5])
                        )
                        vote_delta = int(
                            forward_option[5]
                            - forward_option[4]
                            + reverse_option[5]
                            - reverse_option[4]
                        )
                        key = (
                            global_residual - improvement,
                            max(forward_target_loss, reverse_target_loss),
                            forward_target_loss + reverse_target_loss,
                            max(forward_actual_loss, reverse_actual_loss),
                            forward_actual_loss + reverse_actual_loss,
                            -vote_delta,
                            forward_option[0],
                            reverse_option[0],
                            source_shard,
                            target_shard,
                        )
                        candidate = (key, forward_option, reverse_option)
                        if best_for_pair is None or key < best_for_pair[0]:
                            best_for_pair = candidate
                if best_for_pair is not None:
                    candidates.append(best_for_pair)

        candidate_pair_count += len(candidates)
        candidates.sort(key=lambda item: item[0])
        moved_this_round: set[int] = set()
        swaps_this_round = 0
        for _key, forward_option, reverse_option in candidates:
            forward_l1, source_shard, target_shard, forward_weight, *_ = forward_option
            reverse_l1, reverse_source, reverse_target, reverse_weight, *_ = reverse_option
            if forward_l1 in moved_this_round or reverse_l1 in moved_this_round:
                continue
            if reverse_source != target_shard or reverse_target != source_shard:
                raise RuntimeError("weighted swap candidate is not reciprocal")
            if int(current_shard[forward_l1]) != source_shard or int(
                current_shard[reverse_l1]
            ) != target_shard:
                continue

            # Earlier swaps in this round may have changed attachment votes or
            # shard loads, so revalidate both legs immediately before the atomic
            # commit.
            forward_votes = navigation_shard_votes(
                point_to_l1s[forward_l1], current_shard
            )
            reverse_votes = navigation_shard_votes(
                point_to_l1s[reverse_l1], current_shard
            )
            if target_shard not in eligible_navigation_shards(
                forward_votes, max_vote_loss
            ) or source_shard not in eligible_navigation_shards(
                reverse_votes, max_vote_loss
            ):
                continue
            load_source = int(current_loads[source_shard])
            load_target = int(current_loads[target_shard])
            before_pair_residual = int(
                max(0, int(lower_load) - load_source)
                + max(0, load_source - int(upper_load))
                + max(0, int(lower_load) - load_target)
                + max(0, load_target - int(upper_load))
            )
            transferred = int(forward_weight) - int(reverse_weight)
            new_source_load = load_source - transferred
            new_target_load = load_target + transferred
            after_pair_residual = int(
                max(0, int(lower_load) - new_source_load)
                + max(0, new_source_load - int(upper_load))
                + max(0, int(lower_load) - new_target_load)
                + max(0, new_target_load - int(upper_load))
            )
            if after_pair_residual >= before_pair_residual:
                continue

            forward_source_vote = int(forward_votes.get(source_shard, 0))
            forward_target_vote = int(forward_votes[target_shard])
            reverse_source_vote = int(reverse_votes.get(target_shard, 0))
            reverse_target_vote = int(reverse_votes[source_shard])
            current_shard[forward_l1] = target_shard
            current_shard[reverse_l1] = source_shard
            current_loads[source_shard] = new_source_load
            current_loads[target_shard] = new_target_load
            moved_this_round.update((forward_l1, reverse_l1))
            swaps_this_round += 1
            swap_count += 1
            moved_l1_count += 2
            vote_delta = int(
                forward_target_vote
                - forward_source_vote
                + reverse_target_vote
                - reverse_source_vote
            )
            navigation_vote_delta += vote_delta
            maximum_observed_vote_loss = max(
                maximum_observed_vote_loss,
                forward_source_vote - forward_target_vote,
                reverse_source_vote - reverse_target_vote,
            )
            maximum_target_vote_loss = max(
                maximum_target_vote_loss,
                max(forward_votes.values()) - forward_target_vote,
                max(reverse_votes.values()) - reverse_target_vote,
            )

        rounds = round_index + 1
        if swaps_this_round == 0:
            stop_reason = "no_improving_reciprocal_swap"
            break
        if int(
            capacity_band_residual(current_loads, lower_load, upper_load)["total"]
        ) == 0:
            stop_reason = "bounds_satisfied"
            break
    else:
        if attempted:
            stop_reason = "round_limit_reached"

    after_residual = capacity_band_residual(current_loads, lower_load, upper_load)
    feasible = int(after_residual["total"]) == 0
    if feasible and attempted:
        stop_reason = "bounds_satisfied"
    return {
        "attempted": attempted,
        "feasible": feasible,
        "reason": stop_reason,
        "rounds": int(rounds),
        "swap_count": int(swap_count),
        "cycle_count": 0,
        "moved_l1_count": int(moved_l1_count),
        "candidate_pair_count": int(candidate_pair_count),
        "navigation_vote_delta": int(navigation_vote_delta),
        "maximum_observed_vote_loss": int(maximum_observed_vote_loss),
        "maximum_target_vote_loss": int(maximum_target_vote_loss),
        "before_residual": before_residual,
        "after_residual": after_residual,
    }


def converge_l1_topology_capacity_constrained(
    point_to_l1s: list[list[int]],
    upper_indices: np.ndarray,
    up_tier_weights: np.ndarray,
    l1_to_shard: list[int],
    num_shards: int,
    max_iters: int,
    min_load_ratio: float,
    max_load_ratio: float,
    max_vote_loss: int,
) -> tuple[list[int], int, dict[str, Any]]:
    """Refine L1 ownership sequentially without allowing majority-vote collapse.

    Every target shard must occur in the node's immutable upper-navigation
    attachments. Topology-improving moves are admitted inside the capacity band;
    an overloaded shard may also make a bounded vote-loss repair when that move
    reduces total absolute load error.
    """

    validate_capacity_balance_parameters(
        min_load_ratio,
        max_load_ratio,
        max_iters,
        max_vote_loss,
    )
    if len(upper_indices) != len(up_tier_weights):
        raise ValueError("upper_indices and up_tier_weights must have the same length")

    current_shard = list(l1_to_shard)
    upper_rows = [int(value) for value in upper_indices.tolist()]
    weights_by_l1 = {
        l1_idx: int(up_tier_weights[local_idx])
        for local_idx, l1_idx in enumerate(upper_rows)
    }
    current_loads = np.zeros(int(num_shards), dtype=np.int64)
    for l1_idx in upper_rows:
        shard_id = int(current_shard[l1_idx])
        if shard_id != -1:
            current_loads[shard_id] += weights_by_l1[l1_idx]

    ideal_load, lower_load, upper_load = capacity_load_bounds(
        int(np.sum(current_loads)),
        int(num_shards),
        min_load_ratio,
        max_load_ratio,
    )
    initial_summary = load_balance_summary(current_loads)
    iteration_count = 0
    move_counts = Counter()
    rejected_counts = Counter()
    navigation_vote_delta = 0
    maximum_observed_vote_loss = 0

    for iteration in range(int(max_iters)):
        proposals: list[tuple[tuple[Any, ...], int, int, int, int, int, str]] = []
        for l1_idx in upper_rows:
            source_shard = int(current_shard[l1_idx])
            if source_shard == -1:
                continue
            weight = int(weights_by_l1[l1_idx])
            votes = navigation_shard_votes(point_to_l1s[l1_idx], current_shard)
            if not votes:
                rejected_counts["no_navigation_evidence"] += 1
                continue
            source_vote = int(votes.get(source_shard, 0))
            best_vote = int(max(votes.values()))
            best: tuple[tuple[Any, ...], int, int, int, int, int, str] | None = None
            for target_shard, raw_target_vote in votes.items():
                target_shard = int(target_shard)
                target_vote = int(raw_target_vote)
                if target_shard == source_shard:
                    continue
                if best_vote - target_vote > int(max_vote_loss):
                    rejected_counts["vote_loss_bound"] += 1
                    continue
                if int(current_loads[target_shard]) + weight > upper_load:
                    rejected_counts["target_capacity"] += 1
                    continue
                if int(current_loads[source_shard]) - weight < lower_load:
                    rejected_counts["source_floor"] += 1
                    continue

                before_error = abs(float(current_loads[source_shard]) - ideal_load) + abs(
                    float(current_loads[target_shard]) - ideal_load
                )
                after_error = abs(
                    float(current_loads[source_shard] - weight) - ideal_load
                ) + abs(float(current_loads[target_shard] + weight) - ideal_load)
                imbalance_improvement = before_error - after_error
                vote_gain = target_vote - source_vote

                reason: str | None = None
                needs_balance_repair = (
                    int(current_loads[source_shard]) > upper_load
                    or int(current_loads[target_shard]) < lower_load
                )
                if needs_balance_repair and imbalance_improvement > 0.0:
                    reason = "balance_repair"
                    key = (
                        0,
                        max(0, -vote_gain) / float(max(1, weight)),
                        -imbalance_improvement,
                        -target_vote,
                        target_shard,
                    )
                elif vote_gain > 0:
                    reason = "topology_gain"
                    key = (
                        1,
                        -vote_gain / float(max(1, weight)),
                        -vote_gain,
                        int(current_loads[target_shard]),
                        target_shard,
                    )
                else:
                    rejected_counts["no_objective_improvement"] += 1
                    continue

                proposal = (
                    key,
                    l1_idx,
                    source_shard,
                    target_shard,
                    source_vote,
                    target_vote,
                    reason,
                )
                if best is None or proposal[0] < best[0]:
                    best = proposal
            if best is not None:
                proposals.append(best)

        proposals.sort(key=lambda item: (item[0], item[1]))
        changed_count = 0
        moved_l1s: set[int] = set()
        for (
            _key,
            l1_idx,
            proposed_source,
            target_shard,
            source_vote,
            target_vote,
            reason,
        ) in proposals:
            if l1_idx in moved_l1s:
                continue
            source_shard = int(current_shard[l1_idx])
            if source_shard != int(proposed_source):
                rejected_counts["stale_source"] += 1
                continue
            weight = int(weights_by_l1[l1_idx])
            if int(current_loads[target_shard]) + weight > upper_load:
                rejected_counts["target_capacity_after_prior_move"] += 1
                continue
            if int(current_loads[source_shard]) - weight < lower_load:
                rejected_counts["source_floor_after_prior_move"] += 1
                continue
            if reason == "balance_repair" and int(current_loads[source_shard]) <= upper_load:
                if int(target_vote) <= int(source_vote):
                    rejected_counts["repair_no_longer_needed"] += 1
                    continue
                reason = "topology_gain"

            current_shard[l1_idx] = int(target_shard)
            current_loads[source_shard] -= weight
            current_loads[target_shard] += weight
            moved_l1s.add(l1_idx)
            changed_count += 1
            move_counts[reason] += 1
            vote_delta = int(target_vote) - int(source_vote)
            navigation_vote_delta += vote_delta
            maximum_observed_vote_loss = max(maximum_observed_vote_loss, -vote_delta)

        iteration_count = iteration + 1
        if changed_count == 0:
            break

    weighted_repair = repair_l1_topology_with_weighted_swaps(
        point_to_l1s,
        upper_rows,
        weights_by_l1,
        current_shard,
        current_loads,
        int(lower_load),
        int(upper_load),
        int(max_vote_loss),
        int(max_iters),
    )
    if int(weighted_repair["moved_l1_count"]) > 0:
        move_counts["weighted_swap_repair"] += int(
            weighted_repair["moved_l1_count"]
        )
    navigation_vote_delta += int(weighted_repair["navigation_vote_delta"])
    maximum_observed_vote_loss = max(
        maximum_observed_vote_loss,
        int(weighted_repair["maximum_observed_vote_loss"]),
    )

    final_summary = load_balance_summary(current_loads)
    diagnostics = {
        "mode": "capacity_constrained",
        "ideal_load": ideal_load,
        "lower_load": int(lower_load),
        "upper_load": int(upper_load),
        "min_load_ratio": float(min_load_ratio),
        "max_load_ratio": float(max_load_ratio),
        "max_vote_loss": int(max_vote_loss),
        "iterations": int(iteration_count),
        "move_counts": {key: int(value) for key, value in sorted(move_counts.items())},
        "rejected_counts": {
            key: int(value) for key, value in sorted(rejected_counts.items())
        },
        "navigation_vote_delta": int(navigation_vote_delta),
        "maximum_observed_vote_loss": int(maximum_observed_vote_loss),
        "moves_without_navigation_evidence": 0,
        "weighted_repair": weighted_repair,
        "initial": initial_summary,
        "final": final_summary,
        "over_upper_shards": [
            int(shard_id)
            for shard_id, load in enumerate(current_loads.tolist())
            if int(load) > upper_load
        ],
        "under_lower_shards": [
            int(shard_id)
            for shard_id, load in enumerate(current_loads.tolist())
            if int(load) < lower_load
        ],
    }
    diagnostics["bounds_satisfied"] = not diagnostics["over_upper_shards"] and not diagnostics[
        "under_lower_shards"
    ]
    return current_shard, iteration_count, diagnostics


def weighted_random_l1_shards(
    num_points: int,
    upper_indices: np.ndarray,
    up_tier_weights: np.ndarray,
    num_shards: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    l1_to_shard = [-1] * int(num_points)
    shard_loads = np.zeros(int(num_shards), dtype=np.int64)
    order = rng.permutation(len(upper_indices))
    for local_idx in order.tolist():
        l1_idx = int(upper_indices[int(local_idx)])
        shard_id = int(np.argmin(shard_loads))
        l1_to_shard[l1_idx] = shard_id
        shard_loads[shard_id] += int(up_tier_weights[int(local_idx)])
    return l1_to_shard


def target_shards_from_votes(
    l1s: list[int],
    reference_l1_shard: list[int],
    point_index: int,
    num_shards: int,
    use_multi_assign: bool,
    multi_assign_min_max_vote: int = 2,
    multi_assign_vote_delta: int = 0,
    multi_assign_max_shards: int = 0,
) -> list[int]:
    if multi_assign_min_max_vote <= 0:
        raise ValueError("multi_assign_min_max_vote must be positive")
    if multi_assign_vote_delta < 0:
        raise ValueError("multi_assign_vote_delta must be non-negative")
    if multi_assign_max_shards < 0:
        raise ValueError("multi_assign_max_shards must be non-negative")

    votes: Counter[int] = Counter()
    for ep_l1 in l1s:
        shard_id = reference_l1_shard[int(ep_l1)] if int(ep_l1) < len(reference_l1_shard) else -1
        if shard_id != -1:
            votes[int(shard_id)] += 1

    target_shards: list[int] = []
    if votes:
        max_vote = max(votes.values())
        if use_multi_assign and max_vote >= multi_assign_min_max_vote:
            min_vote = max(1, max_vote - multi_assign_vote_delta)
            ranked_shards = [
                shard_id
                for shard_id, _vote in sorted(
                    votes.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                if votes[shard_id] >= min_vote
            ]
            if multi_assign_max_shards > 0:
                ranked_shards = ranked_shards[:multi_assign_max_shards]
            target_shards = [int(shard_id) for shard_id in ranked_shards]

    if not target_shards:
        for ep_l1 in l1s:
            shard_id = reference_l1_shard[int(ep_l1)] if int(ep_l1) < len(reference_l1_shard) else -1
            if shard_id != -1:
                target_shards.append(int(shard_id))
                break
        if not target_shards:
            target_shards.append(int(point_index % num_shards))
    return target_shards


def strict_target_shards_from_votes(
    l1s: list[int],
    reference_l1_shard: list[int],
) -> list[int]:
    """Apply the canonical all-max-vote rule to one ordered upper search."""
    ordered_shards = [
        int(reference_l1_shard[int(l1_idx)])
        for l1_idx in l1s
        if 0 <= int(l1_idx) < len(reference_l1_shard)
        and int(reference_l1_shard[int(l1_idx)]) >= 0
    ]
    if not ordered_shards:
        raise RuntimeError("point assignment has no shard-labelled navigation result")

    votes = Counter(ordered_shards)
    max_vote = int(max(votes.values()))
    if max_vote == 1:
        # Search results are stored nearest first, so this is exactly the
        # specified single-assignment exception for an all-one vote.
        return [int(ordered_shards[0])]

    winning_shards = {
        int(shard_id) for shard_id, vote in votes.items() if int(vote) == max_vote
    }
    # Preserve first navigation appearance for deterministic primary selection,
    # while returning every and only maximum-vote shard.
    return list(dict.fromkeys(
        shard_id for shard_id in ordered_shards if shard_id in winning_shards
    ))


def recalibrate_l1_weights_by_voting(
    point_to_l1s: list[list[int]],
    upper_indices: np.ndarray,
    l1_to_shard: list[int],
    num_shards: int,
    use_multi_assign: bool,
    multi_assign_min_max_vote: int = 2,
    multi_assign_vote_delta: int = 0,
    multi_assign_max_shards: int = 0,
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
            multi_assign_min_max_vote,
            multi_assign_vote_delta,
            multi_assign_max_shards,
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
    multi_assign_min_max_vote: int = 2,
    multi_assign_vote_delta: int = 0,
    multi_assign_max_shards: int = 0,
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
            multi_assign_min_max_vote,
            multi_assign_vote_delta,
            multi_assign_max_shards,
        )
        primary_shards[point_index] = int(target_shards[0])
        point_to_shards.append([int(shard_id) for shard_id in target_shards])
    return primary_shards, point_to_shards


def assign_points_by_strict_l1_vote(
    point_to_l1s: list[list[int]],
    reference_l1_shard: list[int],
) -> tuple[np.ndarray, list[list[int]]]:
    primary_shards = np.full(len(point_to_l1s), -1, dtype=np.int32)
    point_to_shards: list[list[int]] = []
    for point_index, l1s in enumerate(point_to_l1s):
        target_shards = strict_target_shards_from_votes(l1s, reference_l1_shard)
        primary_shards[point_index] = int(target_shards[0])
        point_to_shards.append(target_shards)
    return primary_shards, point_to_shards


def requested_copy_count_from_votes(
    votes: Counter[int],
    use_multi_assign: bool,
    multi_assign_min_max_vote: int,
    multi_assign_vote_delta: int,
    multi_assign_max_shards: int,
) -> int:
    if not votes:
        return 1
    max_vote = int(max(votes.values()))
    if not use_multi_assign or max_vote < int(multi_assign_min_max_vote):
        return 1
    min_vote = max(1, max_vote - int(multi_assign_vote_delta))
    count = sum(1 for vote in votes.values() if int(vote) >= min_vote)
    if int(multi_assign_max_shards) > 0:
        count = min(count, int(multi_assign_max_shards))
    return max(1, int(count))


class _DinicCapacityFlow:
    def __init__(self, node_count: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(int(node_count))]

    def add_edge(self, source: int, target: int, capacity: int) -> tuple[int, int]:
        if int(capacity) < 0:
            raise ValueError("flow capacity must be non-negative")
        forward = [int(target), len(self.graph[int(target)]), int(capacity), int(capacity)]
        reverse = [int(source), len(self.graph[int(source)]), 0, 0]
        self.graph[int(source)].append(forward)
        self.graph[int(target)].append(reverse)
        return int(source), len(self.graph[int(source)]) - 1

    def edge_flow(self, reference: tuple[int, int]) -> int:
        source, edge_index = reference
        edge = self.graph[int(source)][int(edge_index)]
        return int(edge[3] - edge[2])

    def max_flow(self, source: int, sink: int) -> int:
        source = int(source)
        sink = int(sink)
        total_flow = 0
        node_count = len(self.graph)
        while True:
            level = [-1] * node_count
            level[source] = 0
            queue = [source]
            for node in queue:
                for target, _reverse, capacity, _original in self.graph[node]:
                    if capacity > 0 and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink] < 0:
                break
            edge_cursor = [0] * node_count

            def send(node: int, available: int) -> int:
                if node == sink:
                    return int(available)
                while edge_cursor[node] < len(self.graph[node]):
                    edge_index = edge_cursor[node]
                    edge = self.graph[node][edge_index]
                    target, reverse_index, capacity, _original = edge
                    if capacity > 0 and level[target] == level[node] + 1:
                        pushed = send(target, min(int(available), int(capacity)))
                        if pushed > 0:
                            edge[2] -= pushed
                            self.graph[target][reverse_index][2] += pushed
                            return int(pushed)
                    edge_cursor[node] += 1
                return 0

            while True:
                pushed = send(source, 1 << 60)
                if pushed == 0:
                    break
                total_flow += int(pushed)
        return int(total_flow)


def capacity_flow_rebalance_point_assignments(
    point_to_l1s: list[list[int]],
    reference_l1_shard: list[int],
    point_to_shards: list[list[int]],
    required_copies: np.ndarray,
    num_shards: int,
    lower_load: int,
    upper_load: int,
    effective_vote_loss: int,
) -> tuple[list[list[int]], np.ndarray, dict[str, Any]]:
    """Solve the residual balance problem as a grouped capacitated circulation.

    Points with the same eligible shard set and copy count share one flow node.
    Group-to-shard capacity is capped by group size, which preserves the invariant
    that a point cannot place two copies in the same shard. The resulting integral
    flow is decomposed deterministically back into per-point memberships.
    """

    started = time.perf_counter()
    grouped_points: dict[tuple[tuple[int, ...], int], list[int]] = defaultdict(list)
    for point_id, l1s in enumerate(point_to_l1s):
        votes = navigation_shard_votes(l1s, reference_l1_shard)
        if votes:
            candidates = tuple(
                sorted(eligible_navigation_shards(votes, int(effective_vote_loss)))
            )
        else:
            candidates = tuple(range(int(num_shards)))
        copies = int(required_copies[point_id])
        if len(candidates) < copies:
            return point_to_shards, shard_counts_from_point_to_shards(
                point_to_shards, num_shards
            ), {
                "attempted": True,
                "feasible": False,
                "reason": "candidate_count_below_requested_copy_count",
                "point_id": int(point_id),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        grouped_points[(candidates, copies)].append(int(point_id))

    groups = sorted(
        grouped_points.items(),
        key=lambda item: (len(item[0][0]), item[0][0], item[0][1]),
    )
    group_count = len(groups)
    source_node = group_count + int(num_shards)
    sink_node = source_node + 1
    super_source = sink_node + 1
    super_sink = super_source + 1
    flow = _DinicCapacityFlow(super_sink + 1)
    demands = [0] * (sink_node + 1)
    group_shard_edges: list[dict[int, tuple[int, int]]] = []

    def add_bounded_edge(source: int, target: int, lower: int, upper: int) -> None:
        if int(lower) < 0 or int(upper) < int(lower):
            raise ValueError("invalid bounded flow edge")
        flow.add_edge(source, target, int(upper) - int(lower))
        demands[source] -= int(lower)
        demands[target] += int(lower)

    for group_id, ((candidates, copies), point_ids_in_group) in enumerate(groups):
        group_size = len(point_ids_in_group)
        add_bounded_edge(
            source_node,
            group_id,
            group_size * int(copies),
            group_size * int(copies),
        )
        references: dict[int, tuple[int, int]] = {}
        for shard_id in candidates:
            shard_node = group_count + int(shard_id)
            references[int(shard_id)] = flow.add_edge(
                group_id,
                shard_node,
                group_size,
            )
        group_shard_edges.append(references)

    for shard_id in range(int(num_shards)):
        add_bounded_edge(
            group_count + shard_id,
            sink_node,
            int(lower_load),
            int(upper_load),
        )
    flow.add_edge(sink_node, source_node, sum(len(shards) for shards in point_to_shards))

    required_super_flow = 0
    for node, demand in enumerate(demands):
        if demand > 0:
            flow.add_edge(super_source, node, int(demand))
            required_super_flow += int(demand)
        elif demand < 0:
            flow.add_edge(node, super_sink, -int(demand))
    achieved_super_flow = flow.max_flow(super_source, super_sink)
    if achieved_super_flow != required_super_flow:
        return point_to_shards, shard_counts_from_point_to_shards(
            point_to_shards, num_shards
        ), {
            "attempted": True,
            "feasible": False,
            "reason": "capacity_circulation_infeasible",
            "group_count": int(group_count),
            "required_super_flow": int(required_super_flow),
            "achieved_super_flow": int(achieved_super_flow),
            "elapsed_seconds": float(time.perf_counter() - started),
        }

    rebalanced: list[list[int]] = [[] for _ in point_to_shards]
    for group_id, ((candidates, copies), point_ids_in_group) in enumerate(groups):
        remaining = {
            int(shard_id): int(flow.edge_flow(group_shard_edges[group_id][int(shard_id)]))
            for shard_id in candidates
        }
        remaining_points = len(point_ids_in_group)
        for point_id in sorted(point_ids_in_group):
            votes = navigation_shard_votes(
                point_to_l1s[point_id], reference_l1_shard
            )
            selectable = [shard_id for shard_id in candidates if remaining[shard_id] > 0]
            selectable.sort(
                key=lambda shard_id: (
                    -remaining[shard_id],
                    -int(votes.get(shard_id, 0)),
                    int(shard_id),
                )
            )
            selected = [int(shard_id) for shard_id in selectable[: int(copies)]]
            if len(selected) != int(copies):
                raise RuntimeError(
                    "failed to decompose grouped capacity flow into unique point copies"
                )
            rebalanced[point_id] = selected
            for shard_id in selected:
                remaining[shard_id] -= 1
            remaining_points -= 1
            if any(value > remaining_points for value in remaining.values()):
                raise RuntimeError("grouped capacity flow decomposition became non-graphical")
        if any(value != 0 for value in remaining.values()):
            raise RuntimeError("grouped capacity flow decomposition left residual copies")

    shard_loads = shard_counts_from_point_to_shards(rebalanced, int(num_shards))
    return rebalanced, shard_loads, {
        "attempted": True,
        "feasible": True,
        "reason": None,
        "group_count": int(group_count),
        "required_super_flow": int(required_super_flow),
        "achieved_super_flow": int(achieved_super_flow),
        "final": load_balance_summary(shard_loads),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def assign_points_by_l1_vote_capacity_constrained(
    point_to_l1s: list[list[int]],
    reference_l1_shard: list[int],
    num_shards: int,
    use_multi_assign: bool,
    multi_assign_min_max_vote: int = 2,
    multi_assign_vote_delta: int = 0,
    multi_assign_max_shards: int = 0,
    min_load_ratio: float = 0.99,
    max_load_ratio: float = 1.01,
    max_passes: int = 8,
    max_vote_loss: int = 3,
    initial_point_to_shards: list[list[int]] | None = None,
    required_copies_override: np.ndarray | None = None,
) -> tuple[np.ndarray, list[list[int]], dict[str, Any]]:
    """Place every physical L0 copy under deterministic capacity constraints.

    The number of copies requested by the existing multi-assignment policy is
    preserved. Their shard locations may change, but only among shards supported
    by this point's immutable L1 navigation attachments and within the configured
    vote-loss bound. No non-evidence shard is used unless the input itself has no
    usable L1 membership, which is reported explicitly.
    """

    if multi_assign_min_max_vote <= 0:
        raise ValueError("multi_assign_min_max_vote must be positive")
    if multi_assign_vote_delta < 0:
        raise ValueError("multi_assign_vote_delta must be non-negative")
    if multi_assign_max_shards < 0:
        raise ValueError("multi_assign_max_shards must be non-negative")
    validate_capacity_balance_parameters(
        min_load_ratio,
        max_load_ratio,
        max_passes,
        max_vote_loss,
    )
    if int(num_shards) <= 0:
        raise ValueError("num_shards must be positive")

    started = time.perf_counter()
    point_count = len(point_to_l1s)
    override = None
    if required_copies_override is not None:
        override = np.asarray(required_copies_override)
        if override.shape != (point_count,):
            raise ValueError(
                "required_copies_override must have one value per point"
            )
        if not np.issubdtype(override.dtype, np.integer):
            raise TypeError("required_copies_override must contain integers")
        if np.any(override <= 0) or np.any(override > int(num_shards)):
            raise ValueError(
                "required_copies_override values must lie in [1, num_shards]"
            )
        override = np.asarray(override, dtype=np.int16)
    effective_vote_loss = max(int(max_vote_loss), int(multi_assign_vote_delta))
    required_copies = np.ones(point_count, dtype=np.int16)
    candidate_slack = np.zeros(point_count, dtype=np.int16)
    max_votes = np.zeros(point_count, dtype=np.int16)
    vote_margins = np.zeros(point_count, dtype=np.int16)
    navigation_point_coverage = np.zeros(int(num_shards), dtype=np.int64)
    eligible_point_coverage = np.zeros(int(num_shards), dtype=np.int64)
    no_evidence_points = 0

    for point_id, l1s in enumerate(point_to_l1s):
        votes = navigation_shard_votes(l1s, reference_l1_shard)
        if not votes:
            no_evidence_points += 1
            continue
        for shard_id in votes:
            navigation_point_coverage[int(shard_id)] += 1
        requested = (
            int(override[point_id])
            if override is not None
            else requested_copy_count_from_votes(
                votes,
                use_multi_assign,
                multi_assign_min_max_vote,
                multi_assign_vote_delta,
                multi_assign_max_shards,
            )
        )
        eligible = eligible_navigation_shards(votes, effective_vote_loss)
        for shard_id in eligible:
            eligible_point_coverage[int(shard_id)] += 1
        if len(eligible) < requested:
            raise RuntimeError(
                "capacity-constrained assignment has fewer eligible evidence shards "
                f"than requested copies for point {point_id}: {len(eligible)} < {requested}"
            )
        ranked_votes = sorted((int(value) for value in votes.values()), reverse=True)
        max_votes[point_id] = ranked_votes[0]
        vote_margins[point_id] = ranked_votes[0] - (
            ranked_votes[1] if len(ranked_votes) > 1 else 0
        )
        required_copies[point_id] = int(requested)
        candidate_slack[point_id] = int(len(eligible) - requested)

    requested_total_copies = int(np.sum(required_copies, dtype=np.int64))
    ideal_load, lower_load, upper_load = capacity_load_bounds(
        requested_total_copies,
        int(num_shards),
        min_load_ratio,
        max_load_ratio,
    )
    point_ids = np.arange(point_count, dtype=np.int64)
    assignment_order = np.lexsort(
        (
            point_ids,
            -max_votes.astype(np.int32),
            -vote_margins.astype(np.int32),
            candidate_slack.astype(np.int32),
        )
    )

    shard_loads = np.zeros(int(num_shards), dtype=np.int64)
    point_to_shards: list[list[int]] = [[] for _ in range(point_count)]
    initial_capacity_overflow_copies = 0
    non_evidence_assignment_count = 0

    if initial_point_to_shards is not None:
        if len(initial_point_to_shards) != point_count:
            raise ValueError(
                "initial_point_to_shards length must match point_to_l1s length"
            )
        for point_id, initial_shards in enumerate(initial_point_to_shards):
            votes = navigation_shard_votes(
                point_to_l1s[point_id], reference_l1_shard
            )
            eligible = set(eligible_navigation_shards(votes, effective_vote_loss))
            selected = [int(shard_id) for shard_id in initial_shards]
            if len(selected) != int(required_copies[point_id]):
                raise ValueError(
                    "initial assignment does not preserve requested copy count for "
                    f"point {point_id}: {len(selected)} != {required_copies[point_id]}"
                )
            if len(set(selected)) != len(selected):
                raise ValueError(
                    f"initial assignment contains duplicate shards for point {point_id}"
                )
            if any(shard_id < 0 or shard_id >= int(num_shards) for shard_id in selected):
                raise ValueError(
                    f"initial assignment contains an invalid shard for point {point_id}"
                )
            if votes and any(shard_id not in eligible for shard_id in selected):
                raise ValueError(
                    "initial assignment is outside the navigation vote-loss bound for "
                    f"point {point_id}: {selected}"
                )
            if not votes:
                non_evidence_assignment_count += len(selected)
            point_to_shards[point_id] = selected
            for shard_id in selected:
                shard_loads[shard_id] += 1
        initial_capacity_overflow_copies = int(
            np.sum(np.maximum(shard_loads - int(upper_load), 0), dtype=np.int64)
        )
    else:
        for raw_point_id in assignment_order.tolist():
            point_id = int(raw_point_id)
            votes = navigation_shard_votes(
                point_to_l1s[point_id], reference_l1_shard
            )
            requested = int(required_copies[point_id])
            if not votes:
                for _copy_index in range(requested):
                    target_shard = min(
                        range(int(num_shards)),
                        key=lambda shard_id: (int(shard_loads[shard_id]), shard_id),
                    )
                    point_to_shards[point_id].append(int(target_shard))
                    shard_loads[target_shard] += 1
                    non_evidence_assignment_count += 1
                continue

            eligible = eligible_navigation_shards(votes, effective_vote_loss)
            available = [
                shard_id
                for shard_id in eligible
                if int(shard_loads[shard_id]) < upper_load
            ]
            available.sort(
                key=lambda shard_id: (
                    -int(votes[shard_id]),
                    int(shard_loads[shard_id]),
                    int(shard_id),
                )
            )
            selected = [int(shard_id) for shard_id in available[:requested]]
            if len(selected) < requested:
                overflow_candidates = [
                    shard_id
                    for shard_id in eligible
                    if int(shard_id) not in set(selected)
                ]
                overflow_candidates.sort(
                    key=lambda shard_id: (
                        int(shard_loads[shard_id]),
                        -int(votes[shard_id]),
                        int(shard_id),
                    )
                )
                needed = requested - len(selected)
                selected.extend(
                    int(shard_id) for shard_id in overflow_candidates[:needed]
                )
                initial_capacity_overflow_copies += needed
            if len(selected) != requested or len(set(selected)) != len(selected):
                raise RuntimeError(
                    "invalid capacity-constrained assignment for point "
                    f"{point_id}: {selected}"
                )
            point_to_shards[point_id] = selected
            for shard_id in selected:
                shard_loads[shard_id] += 1

    initial_summary = load_balance_summary(shard_loads)
    repair_counts = Counter()
    repair_vote_delta = 0
    maximum_observed_vote_loss = 0
    repair_passes = 0

    def repair_phase(phase: str) -> int:
        nonlocal repair_vote_delta, maximum_observed_vote_loss
        moved = 0
        for point_id, l1s in enumerate(point_to_l1s):
            votes = navigation_shard_votes(l1s, reference_l1_shard)
            if not votes:
                continue
            eligible = eligible_navigation_shards(votes, effective_vote_loss)
            assigned = point_to_shards[point_id]
            assigned_set = set(assigned)
            if phase == "overload":
                sources = [
                    shard_id for shard_id in assigned if int(shard_loads[shard_id]) > upper_load
                ]
                targets = [
                    shard_id
                    for shard_id in eligible
                    if shard_id not in assigned_set and int(shard_loads[shard_id]) < upper_load
                ]
            else:
                sources = [
                    shard_id for shard_id in assigned if int(shard_loads[shard_id]) > lower_load
                ]
                targets = [
                    shard_id
                    for shard_id in eligible
                    if shard_id not in assigned_set and int(shard_loads[shard_id]) < lower_load
                ]
            if not sources or not targets:
                continue

            source_shard, target_shard = min(
                (
                    (int(source), int(target))
                    for source in sources
                    for target in targets
                ),
                key=lambda pair: (
                    int(votes.get(pair[0], 0)) - int(votes.get(pair[1], 0)),
                    int(shard_loads[pair[1]]),
                    -int(shard_loads[pair[0]]),
                    pair[1],
                    pair[0],
                ),
            )
            source_position = assigned.index(source_shard)
            assigned[source_position] = target_shard
            shard_loads[source_shard] -= 1
            shard_loads[target_shard] += 1
            moved += 1
            vote_delta = int(votes.get(target_shard, 0)) - int(
                votes.get(source_shard, 0)
            )
            repair_vote_delta += vote_delta
            maximum_observed_vote_loss = max(maximum_observed_vote_loss, -vote_delta)
        return moved

    for _pass_index in range(int(max_passes)):
        overload_moves = repair_phase("overload")
        underfill_moves = repair_phase("underfill")
        repair_counts["overload"] += int(overload_moves)
        repair_counts["underfill"] += int(underfill_moves)
        repair_passes += 1
        over_upper = bool(np.any(shard_loads > upper_load))
        under_lower = bool(np.any(shard_loads < lower_load))
        if (not over_upper and not under_lower) or (
            overload_moves == 0 and underfill_moves == 0
        ):
            break

    capacity_flow_diagnostics: dict[str, Any] = {
        "attempted": False,
        "feasible": None,
        "reason": "greedy_and_direct_repair_satisfied_bounds",
    }
    if bool(np.any(shard_loads > upper_load)) or bool(np.any(shard_loads < lower_load)):
        point_to_shards, shard_loads, capacity_flow_diagnostics = (
            capacity_flow_rebalance_point_assignments(
                point_to_l1s,
                reference_l1_shard,
                point_to_shards,
                required_copies,
                int(num_shards),
                int(lower_load),
                int(upper_load),
                int(effective_vote_loss),
            )
        )

    primary_shards = np.full(point_count, -1, dtype=np.int32)
    assignments_changed_from_unconstrained = 0
    all_assignments_have_navigation_evidence = True
    per_point_copy_count_preserved = True
    assignment_target_vote_loss_histogram = Counter()
    assignment_target_vote_loss_total = 0
    for point_id, l1s in enumerate(point_to_l1s):
        votes = navigation_shard_votes(l1s, reference_l1_shard)
        assigned = point_to_shards[point_id]
        assigned.sort(key=lambda shard_id: (-int(votes.get(shard_id, 0)), int(shard_id)))
        if len(assigned) != int(required_copies[point_id]):
            per_point_copy_count_preserved = False
        if votes and any(shard_id not in votes for shard_id in assigned):
            all_assignments_have_navigation_evidence = False
        if not votes:
            all_assignments_have_navigation_evidence = False
        else:
            best_vote = int(max(votes.values()))
            for shard_id in assigned:
                if shard_id in votes:
                    target_vote_loss = best_vote - int(votes[shard_id])
                    assignment_target_vote_loss_histogram[target_vote_loss] += 1
                    assignment_target_vote_loss_total += target_vote_loss
        primary_shards[point_id] = int(assigned[0])
        unconstrained = target_shards_from_votes(
            l1s,
            reference_l1_shard,
            point_id,
            int(num_shards),
            use_multi_assign,
            multi_assign_min_max_vote,
            multi_assign_vote_delta,
            multi_assign_max_shards,
        )
        if assigned != unconstrained:
            assignments_changed_from_unconstrained += 1

    final_summary = load_balance_summary(shard_loads)
    over_upper_shards = [
        int(shard_id)
        for shard_id, load in enumerate(shard_loads.tolist())
        if int(load) > upper_load
    ]
    under_lower_shards = [
        int(shard_id)
        for shard_id, load in enumerate(shard_loads.tolist())
        if int(load) < lower_load
    ]
    diagnostics = {
        "mode": "capacity_constrained",
        "initialization": (
            "existing_orion_layout"
            if initial_point_to_shards is not None
            else "capacity_aware_greedy"
        ),
        "ideal_load": ideal_load,
        "lower_load": int(lower_load),
        "upper_load": int(upper_load),
        "min_load_ratio": float(min_load_ratio),
        "max_load_ratio": float(max_load_ratio),
        "configured_max_vote_loss": int(max_vote_loss),
        "effective_max_vote_loss": int(effective_vote_loss),
        "requested_total_copies": int(requested_total_copies),
        "required_copy_count_source": (
            "explicit_override"
            if override is not None
            else "multi_assignment_vote_rule"
        ),
        "navigation_point_coverage_by_shard": [
            int(value) for value in navigation_point_coverage.tolist()
        ],
        "eligible_point_coverage_by_shard": [
            int(value) for value in eligible_point_coverage.tolist()
        ],
        "eligible_coverage_below_lower_bound_shards": [
            int(shard_id)
            for shard_id, coverage in enumerate(eligible_point_coverage.tolist())
            if int(coverage) < lower_load
        ],
        "copy_count_preserved": bool(per_point_copy_count_preserved),
        "initial_capacity_overflow_copies": int(initial_capacity_overflow_copies),
        "no_evidence_points": int(no_evidence_points),
        "non_evidence_assignment_count": int(non_evidence_assignment_count),
        "all_assignments_have_navigation_evidence": bool(
            all_assignments_have_navigation_evidence
        ),
        "assignments_changed_from_unconstrained": int(
            assignments_changed_from_unconstrained
        ),
        "repair_passes": int(repair_passes),
        "repair_counts": {
            key: int(value) for key, value in sorted(repair_counts.items())
        },
        "repair_navigation_vote_delta": int(repair_vote_delta),
        "maximum_observed_vote_loss": int(maximum_observed_vote_loss),
        "assignment_target_vote_loss_histogram": {
            str(loss): int(count)
            for loss, count in sorted(assignment_target_vote_loss_histogram.items())
        },
        "maximum_assignment_target_vote_loss": int(
            max(assignment_target_vote_loss_histogram, default=0)
        ),
        "mean_assignment_target_vote_loss": float(
            assignment_target_vote_loss_total / max(1, requested_total_copies)
        ),
        "capacity_flow": capacity_flow_diagnostics,
        "initial": initial_summary,
        "final": final_summary,
        "over_upper_shards": over_upper_shards,
        "under_lower_shards": under_lower_shards,
        "bounds_satisfied": not over_upper_shards and not under_lower_shards,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return primary_shards, point_to_shards, diagnostics


def total_assigned_points(point_to_shards: list[list[int]]) -> int:
    return sum(len(shards) for shards in point_to_shards)


def canonical_orion_assignment_line(point_id: int, shard_ids: list[int]) -> bytes:
    return (
        json.dumps(
            {"id": int(point_id), "shards": [int(shard_id) for shard_id in shard_ids]},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_orion_point_to_shards(
    point_to_shards: list[list[int]],
    num_points: int,
    num_shards: int,
) -> tuple[list[list[int]], int]:
    if len(point_to_shards) != int(num_points):
        raise ValueError(
            "point_to_shards length must equal the number of training vectors: "
            f"{len(point_to_shards)} != {num_points}"
        )
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    normalized_assignments: list[list[int]] = []
    total_copies = 0
    for point_id, shard_ids in enumerate(point_to_shards):
        if not shard_ids:
            raise ValueError(f"point {point_id} has no target numeric shards")
        normalized: list[int] = []
        seen: set[int] = set()
        for raw_shard_id in shard_ids:
            if isinstance(raw_shard_id, (bool, np.bool_)) or not isinstance(
                raw_shard_id, (int, np.integer)
            ):
                raise TypeError(
                    f"point {point_id} shard IDs must be integers, got {raw_shard_id!r}"
                )
            shard_id = int(raw_shard_id)
            if not 0 <= shard_id < num_shards:
                raise ValueError(
                    f"point {point_id} targets shard {shard_id}, outside [0, {num_shards})"
                )
            if shard_id in seen:
                raise ValueError(f"point {point_id} repeats shard {shard_id}")
            seen.add(shard_id)
            normalized.append(shard_id)
        normalized_assignments.append(normalized)
        total_copies += len(normalized)
    return normalized_assignments, total_copies


def orion_layout_sha256(point_to_shards: list[list[int]], num_shards: int) -> str:
    normalized, _total_copies = validate_orion_point_to_shards(
        point_to_shards,
        len(point_to_shards),
        num_shards,
    )
    digest = hashlib.sha256()
    for point_id, shard_ids in enumerate(normalized):
        digest.update(canonical_orion_assignment_line(point_id, shard_ids))
    return digest.hexdigest()


def write_orion_graphless_artifact(
    train: np.ndarray,
    upper_indices: np.ndarray,
    point_to_shards: list[list[int]],
    upper_owner_by_point: list[int],
    num_shards: int,
    output_path: str | Path,
    *,
    generation: int,
    vector_distance: str,
    upper_k: int,
    upper_ef_search: int,
    dynamic_ef_base: int,
    dynamic_ef_factor: int,
    vector_name: str = "",
) -> Path:
    vectors = np.asarray(train, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] <= 0 or vectors.shape[1] <= 0:
        raise ValueError("train must be a non-empty two-dimensional array")
    normalized_assignments, total_copies = validate_orion_point_to_shards(
        point_to_shards,
        int(vectors.shape[0]),
        num_shards,
    )
    upper_indices = np.asarray(upper_indices, dtype=np.int64)
    if upper_indices.ndim != 1 or len(upper_indices) == 0:
        raise ValueError("upper_indices must be a non-empty one-dimensional array")
    if len(set(map(int, upper_indices.tolist()))) != len(upper_indices):
        raise ValueError("upper_indices must not contain duplicate point IDs")
    if np.any(upper_indices < 0) or np.any(upper_indices >= len(vectors)):
        raise ValueError("upper_indices contains a point outside the training set")
    if len(upper_owner_by_point) != len(vectors):
        raise ValueError("upper_owner_by_point must contain one entry per training vector")
    normalized_upper_owners: dict[int, int] = {}
    for point_id in map(int, upper_indices.tolist()):
        owner_shard = int(upper_owner_by_point[point_id])
        if owner_shard < 0 or owner_shard >= int(num_shards):
            raise ValueError(
                f"upper point {point_id} owner shard {owner_shard} is outside [0, {num_shards})"
            )
        if owner_shard not in normalized_assignments[point_id]:
            raise ValueError(
                f"upper point {point_id} owner shard {owner_shard} has no lower-tier copy"
            )
        normalized_upper_owners[point_id] = owner_shard
    if generation <= 0:
        raise ValueError("generation must be positive")
    if upper_k <= 0 or upper_k > len(upper_indices):
        raise ValueError("upper_k must be positive and no larger than the upper tier")
    if upper_ef_search < upper_k:
        raise ValueError("upper_ef_search must be at least upper_k")
    if dynamic_ef_base <= 0 or dynamic_ef_factor < 0:
        raise ValueError("Dynamic EF base must be positive and factor non-negative")
    if not isinstance(vector_name, str):
        raise TypeError("vector_name must be a string")

    distance_name = str(vector_distance).strip().lower()
    distance = {
        "cosine": "Cosine",
        "dot": "Dot",
        "euclid": "Euclid",
        "l2": "Euclid",
        "manhattan": "Manhattan",
    }.get(distance_name)
    if distance is None:
        raise ValueError(f"unsupported Orion artifact distance: {vector_distance!r}")

    layout_digest = hashlib.sha256()
    for point_id, shard_ids in enumerate(normalized_assignments):
        layout_digest.update(canonical_orion_assignment_line(point_id, shard_ids))

    artifact = {
        "format_version": 2,
        "generation": int(generation),
        "vector_schema": {
            "vector_name": vector_name,
            "dimension": int(vectors.shape[1]),
            "distance": distance,
            "datatype": "float32",
        },
        "shard_count": int(num_shards),
        "layout_sha256": layout_digest.hexdigest(),
        "logical_point_count": int(len(vectors)),
        "physical_point_count": int(total_copies),
        "upper_k": int(upper_k),
        "upper_ef_search": int(upper_ef_search),
        "dynamic_ef_base": int(dynamic_ef_base),
        "dynamic_ef_factor": int(dynamic_ef_factor),
        "upper_nodes": [
            {
                "label": int(point_id),
                "vector": vectors[int(point_id)].tolist(),
                "owner_shard": normalized_upper_owners[int(point_id)],
            }
            for point_id in upper_indices.tolist()
        ],
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Orion artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def write_orion_upper_seed_graphless_artifact(
    train: np.ndarray,
    upper_indices: np.ndarray,
    output_path: str | Path,
    *,
    generation: int,
    vector_distance: str,
    upper_k: int,
    upper_ef_search: int,
    dynamic_ef_base: int,
    dynamic_ef_factor: int,
    vector_name: str = "",
) -> Path:
    """Write the neutral, build-only input for the one canonical upper graph.

    The upper HNSW is independent of shard memberships, but the typed production
    artifact requires every upper node to have a valid membership and requires a
    full-layout checksum.  This seed therefore binds every logical point to the
    single neutral shard 0.  It must never be imported or served as the final
    Orion layout; ``orion_rebind_memberships --finalize-build`` replaces only the
    layout metadata and upper-node memberships after attachments have been
    generated from the frozen graph.

    The neutral checksum is streamed instead of materializing ``N`` singleton
    Python lists, keeping the pre-attachment stage bounded by the upper vectors.
    """

    vectors = np.asarray(train, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] <= 0 or vectors.shape[1] <= 0:
        raise ValueError("train must be a non-empty two-dimensional array")
    upper_indices = np.asarray(upper_indices, dtype=np.int64)
    if upper_indices.ndim != 1 or len(upper_indices) == 0:
        raise ValueError("upper_indices must be a non-empty one-dimensional array")
    if len(set(map(int, upper_indices.tolist()))) != len(upper_indices):
        raise ValueError("upper_indices must not contain duplicate point IDs")
    if np.any(upper_indices < 0) or np.any(upper_indices >= len(vectors)):
        raise ValueError("upper_indices contains a point outside the training set")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if upper_k <= 0 or upper_k > len(upper_indices):
        raise ValueError("upper_k must be positive and no larger than the upper tier")
    if upper_ef_search < upper_k:
        raise ValueError("upper_ef_search must be at least upper_k")
    if dynamic_ef_base <= 0 or dynamic_ef_factor < 0:
        raise ValueError("Dynamic EF base must be positive and factor non-negative")
    if not isinstance(vector_name, str):
        raise TypeError("vector_name must be a string")

    distance_name = str(vector_distance).strip().lower()
    distance = {
        "cosine": "Cosine",
        "dot": "Dot",
        "euclid": "Euclid",
        "l2": "Euclid",
        "manhattan": "Manhattan",
    }.get(distance_name)
    if distance is None:
        raise ValueError(f"unsupported Orion artifact distance: {vector_distance!r}")

    layout_digest = hashlib.sha256()
    for point_id in range(len(vectors)):
        layout_digest.update(canonical_orion_assignment_line(point_id, [0]))

    artifact = {
        "format_version": 2,
        "generation": int(generation),
        "vector_schema": {
            "vector_name": vector_name,
            "dimension": int(vectors.shape[1]),
            "distance": distance,
            "datatype": "float32",
        },
        "shard_count": 1,
        "layout_sha256": layout_digest.hexdigest(),
        "logical_point_count": int(len(vectors)),
        "physical_point_count": int(len(vectors)),
        "upper_k": int(upper_k),
        "upper_ef_search": int(upper_ef_search),
        "dynamic_ef_base": int(dynamic_ef_base),
        "dynamic_ef_factor": int(dynamic_ef_factor),
        "upper_nodes": [
            {
                "label": int(point_id),
                "vector": vectors[int(point_id)].tolist(),
                "owner_shard": 0,
            }
            for point_id in upper_indices.tolist()
        ],
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Orion artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def write_simple_kmeans_graphless_artifact(
    train: np.ndarray,
    centroids: np.ndarray,
    point_to_shards: list[list[int]],
    num_shards: int,
    output_path: str | Path,
    *,
    generation: int,
    vector_distance: str,
    nprobe: int,
    lower_hnsw_ef: int,
    vector_name: str = "",
) -> Path:
    """Write the typed input consumed by ``simple_kmeans_build_artifact``.

    Partitioning is intentionally outside this serializer. Callers must pass the
    single-assignment output of ``build_cpp_kmeans_baseline_assignments``.
    """

    vectors = np.asarray(train, dtype=np.float32)
    centroid_vectors = np.asarray(centroids, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] <= 0 or vectors.shape[1] <= 0:
        raise ValueError("train must be a non-empty two-dimensional array")
    if centroid_vectors.shape != (int(num_shards), int(vectors.shape[1])):
        raise ValueError(
            "centroids must have shape (num_shards, dimension): "
            f"expected={(int(num_shards), int(vectors.shape[1]))}, "
            f"actual={centroid_vectors.shape}"
        )
    if not np.isfinite(centroid_vectors).all():
        raise ValueError("centroids contain a non-finite value")
    normalized_assignments, total_copies = validate_orion_point_to_shards(
        point_to_shards,
        int(vectors.shape[0]),
        num_shards,
    )
    if total_copies != len(vectors) or any(
        len(shard_ids) != 1 for shard_ids in normalized_assignments
    ):
        raise ValueError("Simple KMeans native layout requires exactly one shard per point")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if nprobe <= 0 or nprobe > num_shards:
        raise ValueError("nprobe must be in [1, num_shards]")
    if lower_hnsw_ef <= 0:
        raise ValueError("lower_hnsw_ef must be positive")
    if not isinstance(vector_name, str):
        raise TypeError("vector_name must be a string")

    distance_name = str(vector_distance).strip().lower()
    distance = {
        "cosine": "Cosine",
        "dot": "Dot",
        "euclid": "Euclid",
        "l2": "Euclid",
        "manhattan": "Manhattan",
    }.get(distance_name)
    if distance is None:
        raise ValueError(f"unsupported Simple KMeans artifact distance: {vector_distance!r}")

    layout_digest = hashlib.sha256()
    for point_id, shard_ids in enumerate(normalized_assignments):
        layout_digest.update(canonical_orion_assignment_line(point_id, shard_ids))

    artifact = {
        "format_version": 1,
        "generation": int(generation),
        "vector_schema": {
            "vector_name": vector_name,
            "dimension": int(vectors.shape[1]),
            "distance": distance,
            "datatype": "float32",
        },
        "shard_count": int(num_shards),
        "layout_sha256": layout_digest.hexdigest(),
        "logical_point_count": int(len(vectors)),
        "physical_point_count": int(total_copies),
        "routing_distance": "squared_l2",
        "nprobe": int(nprobe),
        "lower_hnsw_ef": int(lower_hnsw_ef),
        "centroids": [
            {
                "shard_id": shard_id,
                "vector": centroid_vectors[shard_id].tolist(),
            }
            for shard_id in range(num_shards)
        ],
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite Simple KMeans artifact: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def write_orion_numeric_vector_file(
    train: np.ndarray,
    output_path: str | Path,
    *,
    row_chunk_size: int = 16384,
) -> str:
    """Write canonical row-major little-endian f32 vectors exactly once.

    The canonical Orion build uses this file first as the query input for
    ``orion_export_upper_hits`` and later reuses the same bytes in the numeric
    shard import bundle.  Existing files are never overwritten.
    """

    vectors = np.asarray(train)
    if vectors.ndim != 2 or vectors.shape[0] <= 0 or vectors.shape[1] <= 0:
        raise ValueError("train must be a non-empty two-dimensional array")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite vector file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    try:
        little_endian_f32 = np.dtype("<f4")
        with temporary.open("wb") as handle:
            for start in range(0, int(vectors.shape[0]), row_chunk_size):
                chunk = np.ascontiguousarray(
                    vectors[start : start + row_chunk_size],
                    dtype=little_endian_f32,
                )
                if not np.isfinite(chunk).all():
                    raise ValueError(f"train contains a non-finite value near row {start}")
                raw = memoryview(chunk).cast("B")
                handle.write(raw)
                digest.update(raw)
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def write_orion_numeric_shard_import_bundle(
    train: np.ndarray,
    point_to_shards: list[list[int]],
    num_shards: int,
    output_dir: str | Path,
    *,
    orion_artifact_path: str | Path,
    vector_name: str = "",
    prefix: str = "orion_numeric_import",
    row_chunk_size: int = 16384,
    prewritten_vectors_path: str | Path | None = None,
) -> Path:
    """Write the bounded-memory input bundle for ``orion_numeric_shard_import``.

    External point IDs are the original zero-based training row IDs. A single JSONL
    assignment record contains the complete numeric-shard membership for that ID, so
    multi-assigned points keep one external ID and never need synthetic copy IDs or a
    ``source_id`` payload.
    """

    vectors = np.asarray(train)
    if vectors.ndim != 2 or vectors.shape[0] <= 0 or vectors.shape[1] <= 0:
        raise ValueError("train must be a non-empty two-dimensional array")
    normalized_assignments, total_copies = validate_orion_point_to_shards(
        point_to_shards,
        int(vectors.shape[0]),
        num_shards,
    )
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")
    if not isinstance(vector_name, str):
        raise TypeError("vector_name must be a string")
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty file-name component")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_resolved = output_dir.resolve()
    orion_artifact_path = Path(orion_artifact_path).resolve()
    if not orion_artifact_path.is_file():
        raise FileNotFoundError(f"Orion production artifact not found: {orion_artifact_path}")
    if orion_artifact_path.parent != output_dir_resolved:
        raise ValueError("Orion production artifact must be in the import bundle output directory")
    artifact = json.loads(orion_artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or artifact.get("upper_graph") is None:
        raise ValueError("numeric import bundle requires a production artifact with upper_graph")
    artifact_sha256 = sha256_path(orion_artifact_path)
    vectors_path = output_dir / f"{prefix}.f32le"
    assignments_path = output_dir / f"{prefix}.assignments.jsonl"
    manifest_path = output_dir / f"{prefix}.manifest.json"
    reuse_vectors = prewritten_vectors_path is not None
    if reuse_vectors:
        supplied_vectors_path = Path(prewritten_vectors_path).expanduser().resolve()
        if supplied_vectors_path != vectors_path.resolve():
            raise ValueError(
                "prewritten_vectors_path must be the bundle's canonical vectors path"
            )
        if not vectors_path.is_file():
            raise FileNotFoundError(f"prewritten vector file not found: {vectors_path}")
    final_paths = [assignments_path, manifest_path]
    if not reuse_vectors:
        final_paths.insert(0, vectors_path)
    existing = [path for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite an existing Orion import bundle: "
            + ", ".join(str(path) for path in existing)
        )

    vectors_tmp = vectors_path.with_name(vectors_path.name + ".tmp")
    assignments_tmp = assignments_path.with_name(assignments_path.name + ".tmp")
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    temp_paths = [assignments_tmp, manifest_tmp]
    if not reuse_vectors:
        temp_paths.insert(0, vectors_tmp)
    promoted_paths: list[Path] = []
    for path in temp_paths:
        path.unlink(missing_ok=True)

    vectors_digest = hashlib.sha256()
    assignments_digest = hashlib.sha256()
    try:
        little_endian_f32 = np.dtype("<f4")
        if reuse_vectors:
            expected_size = int(vectors.shape[0]) * int(vectors.shape[1]) * 4
            if vectors_path.stat().st_size != expected_size:
                raise ValueError(
                    "prewritten vector file size does not match the training matrix"
                )
            with vectors_path.open("rb") as handle:
                for start in range(0, int(vectors.shape[0]), row_chunk_size):
                    chunk = np.ascontiguousarray(
                        vectors[start : start + row_chunk_size],
                        dtype=little_endian_f32,
                    )
                    if not np.isfinite(chunk).all():
                        raise ValueError(
                            f"train contains a non-finite value near row {start}"
                        )
                    expected = memoryview(chunk).cast("B").tobytes()
                    observed = handle.read(len(expected))
                    if observed != expected:
                        raise ValueError(
                            "prewritten vector file differs from the training matrix "
                            f"near row {start}"
                        )
                    vectors_digest.update(observed)
                if handle.read(1):
                    raise ValueError("prewritten vector file contains trailing bytes")
        else:
            with vectors_tmp.open("wb") as handle:
                for start in range(0, int(vectors.shape[0]), row_chunk_size):
                    chunk = np.ascontiguousarray(
                        vectors[start : start + row_chunk_size],
                        dtype=little_endian_f32,
                    )
                    if not np.isfinite(chunk).all():
                        raise ValueError(f"train contains a non-finite value near row {start}")
                    raw = memoryview(chunk).cast("B")
                    handle.write(raw)
                    vectors_digest.update(raw)

        with assignments_tmp.open("wb") as handle:
            for point_id, shard_ids in enumerate(normalized_assignments):
                encoded = canonical_orion_assignment_line(point_id, shard_ids)
                handle.write(encoded)
                assignments_digest.update(encoded)

        if str(artifact.get("layout_sha256") or "").lower() != assignments_digest.hexdigest():
            raise ValueError("Orion artifact layout_sha256 does not match the import assignments")
        if int(artifact.get("generation") or 0) <= 0:
            raise ValueError("Orion artifact generation must be positive")
        if int(artifact.get("shard_count") or 0) != int(num_shards):
            raise ValueError("Orion artifact shard_count does not match the import bundle")
        if int(artifact.get("logical_point_count") or 0) != int(len(vectors)):
            raise ValueError("Orion artifact logical_point_count does not match the training set")
        if int(artifact.get("physical_point_count") or 0) != int(total_copies):
            raise ValueError("Orion artifact physical_point_count does not match assignments")
        artifact_schema = artifact.get("vector_schema") or {}
        if int(artifact_schema.get("dimension") or 0) != int(vectors.shape[1]):
            raise ValueError("Orion artifact vector dimension does not match the import vectors")
        if str(artifact_schema.get("vector_name") or "") != vector_name:
            raise ValueError("Orion artifact vector name does not match the import bundle")

        manifest = {
            "format_version": 1,
            "dimension": int(vectors.shape[1]),
            "point_count": int(vectors.shape[0]),
            "shard_count": int(num_shards),
            "vector_name": vector_name,
            "orion_generation": int(artifact["generation"]),
            "orion_artifact_file": orion_artifact_path.name,
            "orion_artifact_sha256": artifact_sha256,
            "vectors_file": vectors_path.name,
            "vectors_sha256": vectors_digest.hexdigest(),
            "assignments_file": assignments_path.name,
            "assignments_sha256": assignments_digest.hexdigest(),
            "total_point_copies": int(total_copies),
        }
        with manifest_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")

        if not reuse_vectors:
            os.replace(vectors_tmp, vectors_path)
            promoted_paths.append(vectors_path)
        os.replace(assignments_tmp, assignments_path)
        promoted_paths.append(assignments_path)
        os.replace(manifest_tmp, manifest_path)
        promoted_paths.append(manifest_path)
    except BaseException:
        for path in temp_paths + promoted_paths:
            path.unlink(missing_ok=True)
        raise

    return manifest_path


def write_numeric_shard_import_bundle_v2(
    train: np.ndarray,
    point_to_shards: list[list[int]],
    num_shards: int,
    output_dir: str | Path,
    *,
    routing_policy: str,
    routing_generation: int,
    routing_artifact_path: str | Path,
    vector_name: str = "",
    prefix: str = "numeric_shard_import",
    row_chunk_size: int = 16384,
) -> Path:
    """Write the generic version-2 numeric-shard import bundle.

    Version 2 replaces the Orion-specific manifest binding with a routing policy,
    generation, artifact filename, and artifact checksum. Vector and assignment
    files retain the existing bounded-memory canonical format.
    """

    vectors = np.asarray(train)
    if vectors.ndim != 2 or vectors.shape[0] <= 0 or vectors.shape[1] <= 0:
        raise ValueError("train must be a non-empty two-dimensional array")
    normalized_assignments, total_copies = validate_orion_point_to_shards(
        point_to_shards,
        int(vectors.shape[0]),
        num_shards,
    )
    if routing_policy not in {"orion", "simple_kmeans"}:
        raise ValueError("routing_policy must be one of: orion, simple_kmeans")
    if routing_generation <= 0:
        raise ValueError("routing_generation must be positive")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")
    if not isinstance(vector_name, str):
        raise TypeError("vector_name must be a string")
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("prefix must be a non-empty file-name component")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_resolved = output_dir.resolve()
    routing_artifact_path = Path(routing_artifact_path).resolve()
    if not routing_artifact_path.is_file():
        raise FileNotFoundError(
            f"routing production artifact not found: {routing_artifact_path}"
        )
    if routing_artifact_path.parent != output_dir_resolved:
        raise ValueError(
            "routing production artifact must be in the import bundle output directory"
        )
    artifact = json.loads(routing_artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("routing production artifact root must be a JSON object")
    if routing_policy == "orion":
        if not isinstance(artifact.get("upper_graph"), dict):
            raise ValueError("Orion routing artifact must contain upper_graph")
    else:
        if "upper_graph" in artifact:
            raise ValueError("Simple KMeans routing artifact must not contain upper_graph")
        if artifact.get("routing_distance") != "squared_l2":
            raise ValueError(
                "Simple KMeans routing artifact must use routing_distance=squared_l2"
            )
        if total_copies != len(vectors):
            raise ValueError(
                "Simple KMeans numeric import requires exactly one point copy per logical point"
            )
    artifact_sha256 = sha256_path(routing_artifact_path)

    vectors_path = output_dir / f"{prefix}.f32le"
    assignments_path = output_dir / f"{prefix}.assignments.jsonl"
    manifest_path = output_dir / f"{prefix}.manifest.json"
    final_paths = [vectors_path, assignments_path, manifest_path]
    existing = [path for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite an existing numeric import bundle: "
            + ", ".join(str(path) for path in existing)
        )

    vectors_tmp = vectors_path.with_name(vectors_path.name + ".tmp")
    assignments_tmp = assignments_path.with_name(assignments_path.name + ".tmp")
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    temp_paths = [vectors_tmp, assignments_tmp, manifest_tmp]
    promoted_paths: list[Path] = []
    for path in temp_paths:
        path.unlink(missing_ok=True)

    vectors_digest = hashlib.sha256()
    assignments_digest = hashlib.sha256()
    try:
        little_endian_f32 = np.dtype("<f4")
        with vectors_tmp.open("wb") as handle:
            for start in range(0, int(vectors.shape[0]), row_chunk_size):
                chunk = np.ascontiguousarray(
                    vectors[start : start + row_chunk_size],
                    dtype=little_endian_f32,
                )
                if not np.isfinite(chunk).all():
                    raise ValueError(f"train contains a non-finite value near row {start}")
                raw = memoryview(chunk).cast("B")
                handle.write(raw)
                vectors_digest.update(raw)

        with assignments_tmp.open("wb") as handle:
            for point_id, shard_ids in enumerate(normalized_assignments):
                encoded = canonical_orion_assignment_line(point_id, shard_ids)
                handle.write(encoded)
                assignments_digest.update(encoded)

        if str(artifact.get("layout_sha256") or "").lower() != assignments_digest.hexdigest():
            raise ValueError(
                "routing artifact layout_sha256 does not match the import assignments"
            )
        if int(artifact.get("generation") or 0) != int(routing_generation):
            raise ValueError("routing artifact generation does not match the import bundle")
        if int(artifact.get("shard_count") or 0) != int(num_shards):
            raise ValueError("routing artifact shard_count does not match the import bundle")
        if int(artifact.get("logical_point_count") or 0) != int(len(vectors)):
            raise ValueError(
                "routing artifact logical_point_count does not match the training set"
            )
        if int(artifact.get("physical_point_count") or 0) != int(total_copies):
            raise ValueError(
                "routing artifact physical_point_count does not match assignments"
            )
        artifact_schema = artifact.get("vector_schema") or {}
        if int(artifact_schema.get("dimension") or 0) != int(vectors.shape[1]):
            raise ValueError(
                "routing artifact vector dimension does not match the import vectors"
            )
        if str(artifact_schema.get("vector_name") or "") != vector_name:
            raise ValueError(
                "routing artifact vector name does not match the import bundle"
            )
        if str(artifact_schema.get("datatype") or "").lower() != "float32":
            raise ValueError("numeric import bundle requires a float32 routing artifact")

        manifest = {
            "format_version": 2,
            "routing_policy": routing_policy,
            "routing_generation": int(routing_generation),
            "routing_artifact_file": routing_artifact_path.name,
            "routing_artifact_sha256": artifact_sha256,
            "dimension": int(vectors.shape[1]),
            "point_count": int(vectors.shape[0]),
            "shard_count": int(num_shards),
            "vector_name": vector_name,
            "vectors_file": vectors_path.name,
            "vectors_sha256": vectors_digest.hexdigest(),
            "assignments_file": assignments_path.name,
            "assignments_sha256": assignments_digest.hexdigest(),
            "total_point_copies": int(total_copies),
        }
        with manifest_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")

        os.replace(vectors_tmp, vectors_path)
        promoted_paths.append(vectors_path)
        os.replace(assignments_tmp, assignments_path)
        promoted_paths.append(assignments_path)
        os.replace(manifest_tmp, manifest_path)
        promoted_paths.append(manifest_path)
    except BaseException:
        for path in temp_paths + promoted_paths:
            path.unlink(missing_ok=True)
        raise

    return manifest_path


def expansion_ratio_from_assigned_points(assigned_points: int, logical_points: int) -> float:
    if logical_points <= 0:
        raise ValueError("logical_points must be positive")
    return float(int(assigned_points) / int(logical_points))


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


def iter_shard_upload_batches(
    point_to_shards: list[list[int]],
    num_shards: int,
    batch_size: int,
    upper_indices: np.ndarray | None = None,
) -> Any:
    """Yield bounded point-major batches so shard indexing can overlap ingestion."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if upper_indices is None:
        ordered_points: Any = range(len(point_to_shards))
    else:
        upper_ids = np.asarray(upper_indices, dtype=np.int64)
        if np.any(upper_ids < 0) or np.any(upper_ids >= len(point_to_shards)):
            raise ValueError("upper_indices contains an out-of-range point ID")
        if len(np.unique(upper_ids)) != len(upper_ids):
            raise ValueError("upper_indices must be unique")
        is_upper = np.zeros(len(point_to_shards), dtype=bool)
        is_upper[upper_ids] = True
        ordered_points = [int(point_id) for point_id in upper_ids.tolist()]
        ordered_points.extend(
            int(point_id) for point_id in np.flatnonzero(~is_upper).tolist()
        )

    pending: list[list[int]] = [[] for _ in range(num_shards)]
    for raw_point_id in ordered_points:
        point_id = int(raw_point_id)
        shard_ids = [int(shard_id) for shard_id in point_to_shards[point_id]]
        if not shard_ids:
            raise ValueError(f"point {point_id} has no shard assignment")
        if len(set(shard_ids)) != len(shard_ids):
            raise ValueError(f"point {point_id} repeats a shard assignment")
        for shard_id in shard_ids:
            if not 0 <= shard_id < num_shards:
                raise ValueError(
                    f"point {point_id} references out-of-range shard {shard_id}"
                )
            pending[shard_id].append(point_id)
            if len(pending[shard_id]) == batch_size:
                yield shard_id, np.asarray(pending[shard_id], dtype=np.int64)
                pending[shard_id] = []

    for shard_id, point_ids in enumerate(pending):
        if point_ids:
            yield shard_id, np.asarray(point_ids, dtype=np.int64)


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


def fixed_ef_shard_key_chunks(
    shard_keys: list[str],
    chunk_size: int = 0,
) -> list[list[str]]:
    if chunk_size < 0:
        raise ValueError("fixed EF shard chunk size must be non-negative")
    normalized = [str(shard_key) for shard_key in shard_keys]
    if len(set(normalized)) != len(normalized):
        raise ValueError("fixed EF shard chunking requires unique shard keys")
    if not normalized:
        return []
    if chunk_size == 0 or chunk_size >= len(normalized):
        return [normalized]
    chunks = [
        normalized[start_idx : start_idx + chunk_size]
        for start_idx in range(0, len(normalized), chunk_size)
    ]
    flattened = [shard_key for chunk in chunks for shard_key in chunk]
    if flattened != normalized or sum(map(len, chunks)) != len(set(flattened)):
        raise RuntimeError("fixed EF shard chunks are not a disjoint full partition")
    return chunks


def build_canonical_routing_state(
    train: np.ndarray,
    upper_indices: np.ndarray,
    point_to_l1s: list[list[int]],
    num_shards: int,
    kmeans_iters: int,
    kmeans_seed: int,
    topology_iters: int,
) -> OriginalRoutingState:
    """Build the current Orion layout without balancing, fission, or vote tuning."""
    if len(point_to_l1s) != len(train):
        raise ValueError("point_to_l1s must contain one ordered search result per point")
    l1_to_shard = initial_l1_shards_by_kmeans(
        train,
        upper_indices,
        num_shards,
        kmeans_iters,
        kmeans_seed,
    )
    l1_to_shard, topology_iteration_count = converge_l1_topology_by_self_vote(
        point_to_l1s,
        upper_indices,
        l1_to_shard,
        topology_iters,
    )
    primary_shards, point_to_shards = assign_points_by_strict_l1_vote(
        point_to_l1s,
        l1_to_shard,
    )

    missing_entry_copies = [
        int(point_id)
        for point_id in upper_indices.tolist()
        if int(l1_to_shard[int(point_id)]) not in point_to_shards[int(point_id)]
    ]
    if missing_entry_copies:
        preview = ", ".join(str(point_id) for point_id in missing_entry_copies[:10])
        raise RuntimeError(
            "frozen upper ownership is inconsistent with strict navigation-vote "
            f"assignment; first missing entry copies: {preview}"
        )

    shard_counts = shard_counts_from_point_to_shards(point_to_shards, num_shards)
    total_assigned = int(np.sum(shard_counts))
    return OriginalRoutingState(
        initial_num_shards=num_shards,
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
        fission_events=[],
        balance_diagnostics=None,
    )


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
    multi_assign_min_max_vote: int = 2,
    multi_assign_vote_delta: int = 0,
    multi_assign_max_shards: int = 0,
    enable_topology_refinement: bool = True,
    balance_mode: str = "none",
    balance_min_load_ratio: float = 0.99,
    balance_max_load_ratio: float = 1.01,
    balance_max_passes: int = 8,
    balance_max_vote_loss: int = 3,
    balance_l1_max_vote_loss: int | None = None,
    balance_l0_max_vote_loss: int | None = None,
) -> OriginalRoutingState:
    normalized_balance_mode = str(balance_mode).strip().lower()
    if normalized_balance_mode not in {
        "none",
        "capacity_constrained",
        "post_layout_capacity_constrained",
    }:
        raise ValueError(f"unsupported Orion balance mode: {balance_mode}")
    balance_enabled = normalized_balance_mode != "none"
    fixed_p_balance = normalized_balance_mode == "capacity_constrained"
    post_layout_balance = (
        normalized_balance_mode == "post_layout_capacity_constrained"
    )
    effective_l1_vote_loss, effective_l0_vote_loss = (
        resolve_capacity_balance_vote_losses(
            balance_max_vote_loss,
            balance_l1_max_vote_loss,
            balance_l0_max_vote_loss,
        )
    )
    if balance_enabled:
        for stage_vote_loss in (effective_l1_vote_loss, effective_l0_vote_loss):
            validate_capacity_balance_parameters(
                balance_min_load_ratio,
                balance_max_load_ratio,
                balance_max_passes,
                stage_vote_loss,
            )
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
        max_load_ratio=(
            float(balance_max_load_ratio)
            if fixed_p_balance
            else 1.5
        ),
    )
    topology_iteration_count = 0
    l1_balance_diagnostics: dict[str, Any] | None = None
    if enable_topology_refinement:
        if fixed_p_balance:
            l1_to_shard, topology_iteration_count, l1_balance_diagnostics = (
                converge_l1_topology_capacity_constrained(
                    point_to_l1s,
                    upper_indices,
                    up_tier_weights,
                    l1_to_shard,
                    initial_num_shards,
                    min(int(topology_iters), int(balance_max_passes)),
                    balance_min_load_ratio,
                    balance_max_load_ratio,
                    effective_l1_vote_loss,
                )
            )
            require_capacity_balance_bounds("L1 topology", l1_balance_diagnostics)
        else:
            l1_to_shard, topology_iteration_count = converge_l1_topology(
                point_to_l1s,
                upper_indices,
                up_tier_weights,
                l1_to_shard,
                len(train),
                initial_num_shards,
                topology_iters,
            )
    elif fixed_p_balance:
        require_capacity_balance_bounds("L1 topology", l1_balance_diagnostics)

    fission_events: list[dict[str, Any]] = []
    num_shards = initial_num_shards
    if enable_fission and not fixed_p_balance:
        recalibrated_weights = recalibrate_l1_weights_by_voting(
            point_to_l1s,
            upper_indices,
            l1_to_shard,
            initial_num_shards,
            use_multi_assign,
            multi_assign_min_max_vote,
            multi_assign_vote_delta,
            multi_assign_max_shards,
        )
        l1_to_shard, num_shards, fission_events = apply_fission_simulator(
            train,
            upper_indices,
            recalibrated_weights,
            l1_to_shard,
            initial_num_shards,
            kmeans_iters,
            kmeans_seed,
        )

    point_balance_diagnostics: dict[str, Any] | None = None
    if fixed_p_balance:
        primary_shards, point_to_shards, point_balance_diagnostics = (
            assign_points_by_l1_vote_capacity_constrained(
                point_to_l1s,
                l1_to_shard,
                num_shards,
                use_multi_assign,
                multi_assign_min_max_vote,
                multi_assign_vote_delta,
                multi_assign_max_shards,
                min_load_ratio=balance_min_load_ratio,
                max_load_ratio=balance_max_load_ratio,
                max_passes=balance_max_passes,
                max_vote_loss=effective_l0_vote_loss,
            )
        )
        require_capacity_balance_bounds("L0 physical-copy", point_balance_diagnostics)
    else:
        primary_shards, point_to_shards = assign_points_by_l1_vote(
            point_to_l1s,
            l1_to_shard,
            num_shards,
            use_multi_assign,
            multi_assign_min_max_vote,
            multi_assign_vote_delta,
            multi_assign_max_shards,
        )
        if post_layout_balance:
            primary_shards, point_to_shards, point_balance_diagnostics = (
                assign_points_by_l1_vote_capacity_constrained(
                    point_to_l1s,
                    l1_to_shard,
                    num_shards,
                    use_multi_assign,
                    multi_assign_min_max_vote,
                    multi_assign_vote_delta,
                    multi_assign_max_shards,
                    min_load_ratio=balance_min_load_ratio,
                    max_load_ratio=balance_max_load_ratio,
                    max_passes=balance_max_passes,
                    max_vote_loss=effective_l0_vote_loss,
                    initial_point_to_shards=point_to_shards,
                )
            )
            require_capacity_balance_bounds("L0 physical-copy", point_balance_diagnostics)
    shard_counts = np.zeros(num_shards, dtype=np.int64)
    for shards in point_to_shards:
        for shard_id in shards:
            shard_counts[int(shard_id)] += 1

    total_assigned = int(np.sum(shard_counts))
    balance_diagnostics = None
    if balance_enabled:
        balance_diagnostics = {
            "mode": normalized_balance_mode,
            "fixed_num_shards": True,
            "l1_max_vote_loss": int(effective_l1_vote_loss),
            "l0_max_vote_loss": int(effective_l0_vote_loss),
            "fission_requested": bool(enable_fission),
            "fission_applied": bool(fission_events),
            "fission_disabled_reason": (
                "capacity_constrained_balance_preserves_fixed_P"
                if enable_fission and fixed_p_balance
                else None
            ),
            "l1_topology": l1_balance_diagnostics,
            "l0_physical_copies": point_balance_diagnostics,
        }
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
        balance_diagnostics=balance_diagnostics,
    )


def build_claim_a_partition_routing_state(
    family: str,
    train: np.ndarray,
    upper_indices: np.ndarray,
    point_to_l1s: list[list[int]],
    num_shards: int,
    kmeans_iters: int,
    kmeans_seed: int,
    topology_iters: int,
    use_multi_assign: bool,
    multi_assign_min_max_vote: int = 2,
    multi_assign_vote_delta: int = 0,
    multi_assign_max_shards: int = 0,
    random_seed: int = 12345,
) -> OriginalRoutingState:
    nearest_l1 = np.asarray([l1s[0] for l1s in point_to_l1s], dtype=np.int64)
    l1_weights_map = np.bincount(nearest_l1, minlength=len(train))
    up_tier_weights = l1_weights_map[upper_indices].astype(np.int64, copy=False)

    topology_iteration_count = 0
    note = "claim_a_partition_family_no_fission"
    if family == "random_balanced_46":
        l1_to_shard = weighted_random_l1_shards(
            len(train),
            upper_indices,
            up_tier_weights,
            num_shards,
            random_seed,
        )
        note = "random_balanced_l1_weight_assignment"
    elif family in {"kmeans_topology_46", "kmeans_topology_load_recalibrated_46"}:
        l1_to_shard = initial_l1_shards_by_balanced_kmeans(
            train,
            upper_indices,
            up_tier_weights,
            num_shards,
            kmeans_iters,
            kmeans_seed,
        )
        l1_to_shard, topology_iteration_count = converge_l1_topology(
            point_to_l1s,
            upper_indices,
            up_tier_weights,
            l1_to_shard,
            len(train),
            num_shards,
            topology_iters,
        )
        if family == "kmeans_topology_load_recalibrated_46":
            _recalibrated_weights = recalibrate_l1_weights_by_voting(
                point_to_l1s,
                upper_indices,
                l1_to_shard,
                num_shards,
                use_multi_assign,
                multi_assign_min_max_vote,
                multi_assign_vote_delta,
                multi_assign_max_shards,
            )
            note = "load_recalibration_matches_kmeans_topology_without_fission"
        else:
            note = "balanced_kmeans_plus_topology_without_fission"
    else:
        raise ValueError(f"unsupported Claim A partition family: {family}")

    primary_shards, point_to_shards = assign_points_by_l1_vote(
        point_to_l1s,
        l1_to_shard,
        num_shards,
        use_multi_assign,
        multi_assign_min_max_vote,
        multi_assign_vote_delta,
        multi_assign_max_shards,
    )
    shard_counts = shard_counts_from_point_to_shards(point_to_shards, num_shards)
    total_assigned = int(np.sum(shard_counts))
    return OriginalRoutingState(
        initial_num_shards=num_shards,
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
        fission_events=[],
        claim_a_partition_family=family,
        claim_a_partition_note=note,
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


def cpp_baseline_kmeans_train(
    data: np.ndarray,
    indices: np.ndarray | list[int],
    k: int,
    max_iter: int = 10,
) -> np.ndarray:
    indices_arr = np.asarray(indices, dtype=np.int64)
    if k <= 0:
        raise ValueError("k must be positive")
    if len(indices_arr) == 0:
        raise ValueError("indices must not be empty")
    if len(indices_arr) < k:
        raise ValueError("k must not exceed number of indexed points")

    points = data[indices_arr].astype(np.float32, copy=False)
    centroids = np.empty((k, data.shape[1]), dtype=np.float32)
    for centroid_id in range(k):
        centroids[centroid_id] = points[centroid_id % len(points)]

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


def build_cpp_kmeans_baseline_assignments(
    train: np.ndarray,
    num_shards: int,
    train_size: int,
    kmeans_iters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if len(train) < num_shards:
        raise ValueError("num_shards must not exceed number of training points")

    rng = np.random.default_rng(seed)
    sample_indices = rng.permutation(len(train))
    kmeans_train = sample_indices[: min(len(train), train_size)]
    centroids = cpp_baseline_kmeans_train(
        train,
        kmeans_train,
        num_shards,
        max_iter=kmeans_iters,
    )
    assignments = np.argmin(squared_l2_distances(train.astype(np.float32, copy=False), centroids), axis=1)
    return assignments.astype(np.int32, copy=False), centroids


def simple_kmeans_point_to_shards_by_distance_alpha(
    train: np.ndarray,
    centroids: np.ndarray,
    alpha: float,
    chunk_size: int = 50000,
) -> list[list[int]]:
    if alpha < 1.0:
        raise ValueError("alpha must be >= 1.0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if centroids.ndim != 2 or len(centroids) == 0:
        raise ValueError("centroids must be a non-empty 2D array")

    point_to_shards: list[list[int]] = []
    alpha_squared = float(alpha) * float(alpha)
    centroids_f32 = centroids.astype(np.float32, copy=False)
    for start_idx in range(0, len(train), chunk_size):
        chunk = train[start_idx : start_idx + chunk_size].astype(np.float32, copy=False)
        distances = squared_l2_distances(chunk, centroids_f32)
        nearest = np.min(distances, axis=1)
        thresholds = nearest * alpha_squared
        for row_distances, threshold in zip(distances, thresholds):
            shard_ids = np.flatnonzero(row_distances <= threshold).astype(np.int64, copy=False).tolist()
            if not shard_ids:
                shard_ids = [int(np.argmin(row_distances))]
            point_to_shards.append([int(shard_id) for shard_id in shard_ids])
    return point_to_shards


def shard_counts_from_point_to_shards(point_to_shards: list[list[int]], num_shards: int) -> np.ndarray:
    counts = np.zeros(num_shards, dtype=np.int64)
    for shard_ids in point_to_shards:
        for shard_id in shard_ids:
            if 0 <= int(shard_id) < num_shards:
                counts[int(shard_id)] += 1
    return counts


def sample_cpp_kmeans_upper_points(
    train: np.ndarray,
    assignments: np.ndarray,
    num_shards: int,
    denominator: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], list[dict[str, Any]]]:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    rng = np.random.default_rng(seed)
    sampled_vectors: list[np.ndarray] = []
    sampled_ids: list[int] = []
    label_to_shard: dict[int, str] = {}
    rows: list[dict[str, Any]] = []

    for shard_id in range(num_shards):
        shard_key = shard_key_for_id(shard_id)
        indices = np.flatnonzero(assignments == shard_id).astype(np.int64, copy=False)
        if len(indices) > 1:
            indices = rng.permutation(indices)
        sample_size = int(len(indices) // denominator)
        if sample_size > 0:
            sampled_local = indices[:sample_size]
            sampled_vectors.append(train[sampled_local])
            point_ids = (sampled_local + 1).astype(np.int64)
            sampled_ids.extend(point_ids.tolist())
            for point_id in point_ids.tolist():
                label_to_shard[int(point_id)] = shard_key
        rows.append(
            {
                "shard_id": shard_id,
                "shard_key": shard_key,
                "points_count": int(len(indices)),
                "sample_count": int(sample_size),
            }
        )

    if sampled_vectors:
        merged_vectors = np.vstack(sampled_vectors).astype(np.float32, copy=False)
    else:
        merged_vectors = np.empty((0, train.shape[1]), dtype=np.float32)
    merged_ids = np.asarray(sampled_ids, dtype=np.int64)
    return merged_vectors, merged_ids, label_to_shard, rows


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
    hnsw_space: str = "cosine",
    *,
    random_seed: int = 100,
    construction_threads: int = -1,
) -> Any:
    import hnswlib

    if int(random_seed) <= 0:
        raise ValueError("upper-index random_seed must be positive")
    if int(construction_threads) == 0 or int(construction_threads) < -1:
        raise ValueError("upper-index construction_threads must be -1 or positive")
    index = hnswlib.Index(space=hnsw_space, dim=dim)
    index.init_index(
        max_elements=len(labels),
        ef_construction=ef_construction,
        M=m,
        random_seed=int(random_seed),
    )
    index.add_items(vectors, labels, num_threads=int(construction_threads))
    index.set_ef(ef_search)
    return index


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def routing_metadata_field_mismatches(
    actual: Any,
    expected: Any,
    prefix: str = "",
) -> list[str]:
    if isinstance(actual, dict) and isinstance(expected, dict):
        mismatches: list[str] = []
        for key in sorted(set(actual) | set(expected)):
            field = f"{prefix}.{key}" if prefix else str(key)
            if key not in actual:
                mismatches.append(f"{field} is missing; expected={expected[key]!r}")
            elif key not in expected:
                mismatches.append(f"{field} is unexpected; actual={actual[key]!r}")
            else:
                mismatches.extend(
                    routing_metadata_field_mismatches(actual[key], expected[key], field)
                )
        return mismatches
    if type(actual) is not type(expected) or actual != expected:
        return [f"{prefix}={actual!r} expected={expected!r}"]
    return []


def routing_family_for_args(args: argparse.Namespace) -> str:
    routing_mode = str(args.routing_mode)
    if routing_mode == "faithful_original_rest":
        claim_a_family = str(getattr(args, "claim_a_partition_family", "none"))
        return claim_a_family if claim_a_family != "none" else "orion_method4"
    return {
        "cpp_kmeans_baseline": "cpp_kmeans_baseline",
        "kmeans_simple_nprobe": "simple_kmeans",
        "legacy_centroid": "legacy_centroid",
        "naive_hash_all_shards": "naive_hash",
    }.get(routing_mode, routing_mode)


def build_routing_build_metadata(
    args: argparse.Namespace,
    *,
    train_count: int,
    effective_num_shards: int,
    vector_distance: str,
) -> dict[str, Any]:
    """Return the canonical, collection-build-sensitive routing description.

    This intentionally excludes query-time tuning controls such as upper-k, nprobe,
    Dynamic EF, concurrency, and batching. Those settings can safely change while
    reusing the same lower collection. The fields below describe the partition and
    lower HNSW that are expensive to build and unsafe to infer from Qdrant's schema.
    """

    routing_mode = str(args.routing_mode)
    is_orion = routing_mode == "faithful_original_rest"
    is_simple = routing_mode == "kmeans_simple_nprobe"
    claim_a_family = str(getattr(args, "claim_a_partition_family", "none"))
    canonical_orion = is_orion and claim_a_family == "none"
    orion_multi_assign = canonical_orion or (
        is_orion and not bool(getattr(args, "disable_multi_assign", False))
    )
    simple_alpha = float(getattr(args, "simple_kmeans_multi_assign_alpha", 1.0))
    simple_multi_assign = is_simple and simple_alpha > 1.0
    upper_k_candidates = [
        int(value)
        for value in (getattr(args, "upper_k_candidates", None) or [])
    ]
    upper_search_ef_during_build = (
        int(getattr(args, "k_overlap", 10))
        if canonical_orion
        else max([int(getattr(args, "upper_search_ef", 100)), *upper_k_candidates])
    )
    fission_enabled = (
        is_orion
        and claim_a_family == "none"
        and not canonical_orion
        and not bool(getattr(args, "disable_fission", False))
        and str(getattr(args, "orion_balance_mode", "none")) == "none"
    )

    return {
        "schema_version": ORION_ROUTING_BUILD_METADATA_SCHEMA_VERSION,
        "routing_mode": routing_mode,
        "routing_family": routing_family_for_args(args),
        "num_shards": int(getattr(args, "num_shards", effective_num_shards)),
        "effective_num_shards": int(effective_num_shards),
        "train_count": int(train_count),
        "distance": str(vector_distance),
        "hnsw_m": int(getattr(args, "hnsw_m", 32)),
        "hnsw_ef_construct": int(getattr(args, "ef_construct", 100)),
        "upper_hnsw_m": int(getattr(args, "upper_m", 32)),
        "upper_hnsw_ef_construction": int(
            getattr(args, "upper_ef_construction", 100)
        ),
        "upper_hnsw_search_ef_during_routing_build": upper_search_ef_during_build,
        "upper_sample_seed": int(getattr(args, "upper_sample_seed", 100)),
        "kmeans_rand_seed": int(getattr(args, "kmeans_rand_seed", 1)),
        "kmeans_iters": int(getattr(args, "kmeans_iters", 10)),
        "cpp_kmeans_train_size": int(getattr(args, "cpp_kmeans_train_size", 10000)),
        "sample_denominator": int(getattr(args, "sample_denominator", 32)),
        "k_overlap": int(getattr(args, "k_overlap", 10)),
        "topology_iters": int(getattr(args, "topology_iters", 50)),
        "upper_routing": {
            "ownership": (
                "single_frozen_owner"
                if is_orion
                else "method_specific"
            ),
            "lower_multi_assignment_can_widen_routing": False,
        },
        "multi_assign": {
            "enabled": bool(orion_multi_assign or simple_multi_assign),
            "orion_enabled": bool(orion_multi_assign),
            "orion_min_max_vote": (
                2
                if canonical_orion
                else int(getattr(args, "orion_multi_assign_min_max_vote", 2))
            ),
            "orion_vote_delta": (
                0
                if canonical_orion
                else int(getattr(args, "orion_multi_assign_vote_delta", 0))
            ),
            "orion_max_shards": (
                0
                if canonical_orion
                else int(getattr(args, "orion_multi_assign_max_shards", 0))
            ),
            "policy": (
                "all_max_vote_ties_else_nearest_when_max_vote_is_one"
                if canonical_orion
                else "experimental"
            ),
            "simple_kmeans_enabled": bool(simple_multi_assign),
            "simple_kmeans_alpha": simple_alpha,
            "simple_kmeans_chunk_size": int(
                getattr(args, "simple_kmeans_multi_assign_chunk_size", 50000)
            ),
        },
        "fission": {
            "enabled": bool(fission_enabled),
            "claim_a_partition_family": claim_a_family,
            "claim_a_random_seed": int(getattr(args, "claim_a_random_seed", 12345)),
        },
        "orion_balance": (
            {"mode": "none", "intervention": False}
            if canonical_orion
            else {
                "mode": str(getattr(args, "orion_balance_mode", "none")),
                "min_load_ratio": float(
                    getattr(args, "orion_balance_min_load_ratio", 0.99)
                ),
                "max_load_ratio": float(
                    getattr(args, "orion_balance_max_load_ratio", 1.01)
                ),
                "max_passes": int(getattr(args, "orion_balance_max_passes", 8)),
                "max_vote_loss": int(
                    getattr(args, "orion_balance_max_vote_loss", 3)
                ),
                "l1_max_vote_loss": resolve_capacity_balance_vote_losses(
                    int(getattr(args, "orion_balance_max_vote_loss", 3)),
                    getattr(args, "orion_balance_l1_max_vote_loss", None),
                    getattr(args, "orion_balance_l0_max_vote_loss", None),
                )[0],
                "l0_max_vote_loss": resolve_capacity_balance_vote_losses(
                    int(getattr(args, "orion_balance_max_vote_loss", 3)),
                    getattr(args, "orion_balance_l1_max_vote_loss", None),
                    getattr(args, "orion_balance_l0_max_vote_loss", None),
                )[1],
            }
        ),
        "lower_hnsw": {
            "construction": "independent_full_multilayer_per_shard",
            "retain_non_base_layers": True,
            "routed_search_start_level": 0,
            "routed_entry_points_required": bool(canonical_orion),
        },
        "legacy_centroid": {
            "sample_size": int(getattr(args, "sample_size", 50000)),
            "seed": int(getattr(args, "seed", 42)),
        },
    }


def collection_metadata_for_routing_build(
    routing_build_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        ORION_COLLECTION_METADATA_KEY: {
            "schema_version": ORION_ROUTING_BUILD_METADATA_SCHEMA_VERSION,
            "routing_build": routing_build_metadata,
            "routing_build_sha256": canonical_json_sha256(routing_build_metadata),
        }
    }


def collection_routing_build_metadata_validation(
    info: dict[str, Any],
    expected_routing_build_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_fingerprint = (
        canonical_json_sha256(expected_routing_build_metadata)
        if expected_routing_build_metadata is not None
        else None
    )
    if expected_routing_build_metadata is None:
        return {
            "status": "not_requested",
            "verified": False,
            "expected_fingerprint": None,
            "actual_fingerprint": None,
            "actual_metadata": None,
            "mismatches": [],
        }

    config = info.get("config") or {}
    collection_metadata = config.get("metadata")
    if collection_metadata is not None and not isinstance(collection_metadata, dict):
        return {
            "status": "mismatch",
            "verified": False,
            "expected_fingerprint": expected_fingerprint,
            "actual_fingerprint": None,
            "actual_metadata": None,
            "mismatches": ["collection metadata is not an object"],
        }
    collection_metadata = collection_metadata or {}
    if ORION_COLLECTION_METADATA_KEY not in collection_metadata:
        return {
            "status": "missing_unverified",
            "verified": False,
            "expected_fingerprint": expected_fingerprint,
            "actual_fingerprint": None,
            "actual_metadata": None,
            "mismatches": [],
        }
    envelope = collection_metadata[ORION_COLLECTION_METADATA_KEY]
    if not isinstance(envelope, dict):
        return {
            "status": "mismatch",
            "verified": False,
            "expected_fingerprint": expected_fingerprint,
            "actual_fingerprint": None,
            "actual_metadata": None,
            "mismatches": [f"{ORION_COLLECTION_METADATA_KEY} metadata is not an object"],
        }

    actual_metadata = envelope.get("routing_build")
    stored_fingerprint = envelope.get("routing_build_sha256")
    mismatches: list[str] = []
    if envelope.get("schema_version") != ORION_ROUTING_BUILD_METADATA_SCHEMA_VERSION:
        mismatches.append(
            f"schema_version={envelope.get('schema_version')} "
            f"expected={ORION_ROUTING_BUILD_METADATA_SCHEMA_VERSION}"
        )
    if not isinstance(actual_metadata, dict):
        mismatches.append("routing_build metadata is missing or not an object")
        computed_actual_fingerprint = None
    else:
        computed_actual_fingerprint = canonical_json_sha256(actual_metadata)
        if actual_metadata != expected_routing_build_metadata:
            mismatches.extend(
                routing_metadata_field_mismatches(
                    actual_metadata,
                    expected_routing_build_metadata,
                )[:20]
            )
            mismatches.append(
                f"routing_build_sha256={computed_actual_fingerprint} "
                f"expected={expected_fingerprint}"
            )
    if stored_fingerprint != computed_actual_fingerprint:
        mismatches.append(
            f"stored routing_build_sha256={stored_fingerprint} "
            f"computed={computed_actual_fingerprint}"
        )
    if stored_fingerprint != expected_fingerprint:
        mismatches.append(
            f"stored routing_build_sha256={stored_fingerprint} "
            f"expected={expected_fingerprint}"
        )
    return {
        "status": "mismatch" if mismatches else "verified",
        "verified": not mismatches,
        "expected_fingerprint": expected_fingerprint,
        "actual_fingerprint": stored_fingerprint,
        "actual_metadata": actual_metadata if isinstance(actual_metadata, dict) else None,
        "mismatches": mismatches,
    }


def routing_build_metadata_validation_fields(
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "routing_build_metadata_status": validation["status"],
        "routing_build_metadata_verified": bool(validation["verified"]),
        "routing_build_metadata_expected_fingerprint": validation[
            "expected_fingerprint"
        ],
        "routing_build_metadata_actual_fingerprint": validation["actual_fingerprint"],
    }


def create_collection(
    base_url: str,
    name: str,
    dim: int,
    m: int,
    ef_construct: int,
    vector_distance: str = "Cosine",
    routing_build_metadata: dict[str, Any] | None = None,
    *,
    indexing_threshold: int = 10,
) -> None:
    if isinstance(indexing_threshold, bool) or not isinstance(indexing_threshold, int):
        raise TypeError("indexing_threshold must be an integer")
    if indexing_threshold <= 0:
        raise ValueError("indexing_threshold must be positive")
    body: dict[str, Any] = {
        "vectors": {"size": dim, "distance": vector_distance},
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
        "optimizers_config": {
            "default_segment_number": 1,
            "indexing_threshold": indexing_threshold,
        },
    }
    if routing_build_metadata is not None:
        body["metadata"] = collection_metadata_for_routing_build(routing_build_metadata)
    request_json(
        base_url,
        "PUT",
        f"/collections/{urllib.parse.quote(name, safe='')}",
        body=body,
    )


def normalized_orion_auto_shard_policy(
    auto_shard_policy: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate and canonicalize an optional REST Orion auto-shard policy."""
    if auto_shard_policy is None:
        return None
    if not isinstance(auto_shard_policy, dict):
        raise TypeError("auto_shard_policy must be a dictionary")
    if str(auto_shard_policy.get("type") or "").lower() != "orion":
        raise ValueError("auto_shard_policy.type must be 'orion'")

    generation = auto_shard_policy.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
        raise ValueError("Orion auto_shard_policy generation must be a positive integer")
    artifact_sha256 = str(auto_shard_policy.get("artifact_sha256") or "").lower()
    if len(artifact_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in artifact_sha256
    ):
        raise ValueError("Orion auto_shard_policy artifact_sha256 must be 64 hexadecimal digits")
    return {
        "type": "orion",
        "generation": generation,
        "artifact_sha256": artifact_sha256,
    }


def normalized_auto_shard_policy(
    auto_shard_policy: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate a REST native auto-shard routing policy."""
    if auto_shard_policy is None:
        return None
    if not isinstance(auto_shard_policy, dict):
        raise TypeError("auto_shard_policy must be a dictionary")
    policy_type = str(auto_shard_policy.get("type") or "").lower()
    if policy_type == "orion":
        return normalized_orion_auto_shard_policy(auto_shard_policy)
    if policy_type != "simple_kmeans":
        raise ValueError("auto_shard_policy.type must be 'orion' or 'simple_kmeans'")
    generation = auto_shard_policy.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
        raise ValueError(
            "Simple KMeans auto_shard_policy generation must be a positive integer"
        )
    artifact_sha256 = str(auto_shard_policy.get("artifact_sha256") or "").lower()
    if len(artifact_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in artifact_sha256
    ):
        raise ValueError(
            "Simple KMeans auto_shard_policy artifact_sha256 must be 64 hexadecimal digits"
        )
    return {
        "type": "simple_kmeans",
        "generation": generation,
        "artifact_sha256": artifact_sha256,
    }


def create_numeric_auto_shard_collection(
    base_url: str,
    name: str,
    dim: int,
    num_shards: int,
    m: int,
    ef_construct: int,
    *,
    vector_distance: str = "Cosine",
    auto_shard_policy: dict[str, Any] | None = None,
    replication_factor: int = 1,
    write_consistency_factor: int = 1,
    full_scan_threshold: int = 10,
    indexing_threshold: int = 10,
    max_indexing_threads: int = 0,
    max_optimization_threads: int | None = None,
    max_segment_size_kb: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a numeric, automatically sharded collection for native routing.

    Omitting ``auto_shard_policy`` creates Qdrant's default HashAll collection.
    Supplying an Orion policy changes only server-side automatic routing; it does
    not switch the collection to custom shard keys.
    """
    positive_fields = {
        "dim": dim,
        "num_shards": num_shards,
        "m": m,
        "ef_construct": ef_construct,
        "replication_factor": replication_factor,
        "write_consistency_factor": write_consistency_factor,
        "full_scan_threshold": full_scan_threshold,
        "indexing_threshold": indexing_threshold,
    }
    for field_name, value in positive_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
    if write_consistency_factor > replication_factor:
        raise ValueError("write_consistency_factor cannot exceed replication_factor")
    if (
        isinstance(max_indexing_threads, bool)
        or not isinstance(max_indexing_threads, int)
        or max_indexing_threads < 0
    ):
        raise ValueError("max_indexing_threads must be a non-negative integer")
    for field_name, value in (
        ("max_optimization_threads", max_optimization_threads),
        ("max_segment_size_kb", max_segment_size_kb),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{field_name} must be a positive integer when set")

    policy = normalized_auto_shard_policy(auto_shard_policy)
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    body: dict[str, Any] = {
        "vectors": {"size": dim, "distance": vector_distance},
        "shard_number": num_shards,
        "sharding_method": "auto",
        "replication_factor": replication_factor,
        "write_consistency_factor": write_consistency_factor,
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
    if max_optimization_threads is not None:
        body["optimizers_config"]["max_optimization_threads"] = (
            max_optimization_threads
        )
    if max_segment_size_kb is not None:
        body["optimizers_config"]["max_segment_size_kb"] = max_segment_size_kb
    if policy is not None:
        body["auto_shard_policy"] = policy
    if metadata is not None:
        body["metadata"] = metadata
    return request_json(
        base_url,
        "PUT",
        f"/collections/{urllib.parse.quote(name, safe='')}",
        body=body,
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


def upsert_numeric_auto_points(
    base_url: str,
    collection: str,
    vectors: np.ndarray,
    *,
    vector_name: str = "",
    batch_size: int = 512,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Publicly upsert IDs ``0..N-1`` without a shard key into an auto collection."""
    rows = np.asarray(vectors, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] <= 0 or rows.shape[1] <= 0:
        raise ValueError("vectors must be a non-empty two-dimensional array")
    if not np.isfinite(rows).all():
        raise ValueError("vectors contain a non-finite value")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not isinstance(vector_name, str):
        raise TypeError("vector_name must be a string")

    path = f"/collections/{urllib.parse.quote(collection, safe='')}/points?wait=true"
    batch_count = 0
    for start in range(0, len(rows), batch_size):
        points: list[dict[str, Any]] = []
        for point_id in range(start, min(start + batch_size, len(rows))):
            vector = rows[point_id].tolist()
            points.append(
                {
                    "id": point_id,
                    "vector": {vector_name: vector} if vector_name else vector,
                }
            )
        request_json(
            base_url,
            "PUT",
            path,
            body={"points": points},
            timeout=timeout,
        )
        batch_count += 1
    return {
        "point_count": int(len(rows)),
        "batch_count": batch_count,
        "first_id": 0,
        "last_id": int(len(rows) - 1),
        "vector_name": vector_name,
        "uses_shard_key": False,
    }


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


def numeric_shard_replicas(cluster_info: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Return numeric shard replicas from a collection cluster response.

    Numeric auto-shards must not carry a custom shard key. Local rows sometimes
    omit ``peer_id``, so the coordinator peer ID is used for those rows only.
    """
    if not isinstance(cluster_info, dict):
        raise TypeError("cluster_info must be a dictionary")
    controller_peer_id = cluster_info.get("peer_id")
    if isinstance(controller_peer_id, bool) or not isinstance(controller_peer_id, int):
        raise ValueError("collection cluster response is missing a numeric peer_id")

    replicas: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for local, rows in (
        (True, cluster_info.get("local_shards") or []),
        (False, cluster_info.get("remote_shards") or []),
    ):
        if not isinstance(rows, list):
            raise ValueError("collection cluster shard rows must be lists")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"invalid collection cluster shard row: {row!r}")
            if row.get("shard_key") is not None:
                raise ValueError(
                    "expected numeric auto-shards, found custom shard key "
                    f"{row.get('shard_key')!r}"
                )
            shard_id = row.get("shard_id")
            if isinstance(shard_id, bool) or not isinstance(shard_id, int) or shard_id < 0:
                raise ValueError(f"invalid numeric shard_id in cluster response: {shard_id!r}")
            peer_id = row.get("peer_id", controller_peer_id if local else None)
            if isinstance(peer_id, bool) or not isinstance(peer_id, int):
                raise ValueError(
                    f"numeric shard {shard_id} is missing a numeric replica peer_id"
                )
            replicas[shard_id].append(
                {
                    "peer_id": peer_id,
                    "state": str(row.get("state") or ""),
                    "local": local,
                }
            )
    return {shard_id: replicas[shard_id] for shard_id in sorted(replicas)}


def numeric_shard_placement_from_cluster(
    cluster_info: dict[str, Any],
    *,
    expected_shard_count: int | None = None,
    require_active: bool = True,
) -> dict[int, int]:
    """Extract a strict RF=1 numeric shard-to-peer mapping."""
    if cluster_info.get("shard_transfers"):
        raise RuntimeError("numeric shard placement is not stable while transfers are active")
    replicas = numeric_shard_replicas(cluster_info)
    if expected_shard_count is not None:
        if expected_shard_count <= 0:
            raise ValueError("expected_shard_count must be positive")
        advertised_count = cluster_info.get("shard_count")
        if advertised_count != expected_shard_count:
            raise RuntimeError(
                f"collection reports {advertised_count!r} shards, expected {expected_shard_count}"
            )
        expected_ids = set(range(expected_shard_count))
        if set(replicas) != expected_ids:
            raise RuntimeError(
                "numeric shard IDs do not match the expected contiguous range: "
                f"expected={sorted(expected_ids)}, actual={sorted(replicas)}"
            )

    placement: dict[int, int] = {}
    for shard_id, shard_replicas in replicas.items():
        if len(shard_replicas) != 1:
            raise RuntimeError(
                f"numeric shard {shard_id} has {len(shard_replicas)} replicas; RF=1 is required"
            )
        replica = shard_replicas[0]
        if require_active and replica["state"] != "Active":
            raise RuntimeError(
                f"numeric shard {shard_id} replica is {replica['state']!r}, expected 'Active'"
            )
        placement[shard_id] = int(replica["peer_id"])
    return placement


def discover_numeric_shard_placement(
    base_url: str,
    collection: str,
    *,
    expected_shard_count: int | None = None,
) -> dict[int, int]:
    cluster_info = collection_cluster_info(base_url, collection)
    if cluster_info is None:
        raise RuntimeError(f"could not read cluster placement for collection {collection!r}")
    return numeric_shard_placement_from_cluster(
        cluster_info,
        expected_shard_count=expected_shard_count,
    )


def round_robin_numeric_shard_targets(
    shard_ids: list[int] | set[int] | tuple[int, ...],
    worker_peer_ids: list[int] | tuple[int, ...],
) -> dict[int, int]:
    normalized_workers: list[int] = []
    for peer_id in worker_peer_ids:
        if isinstance(peer_id, bool) or not isinstance(peer_id, int) or peer_id < 0:
            raise ValueError(f"worker peer IDs must be non-negative integers, got {peer_id!r}")
        normalized_workers.append(peer_id)
    if not normalized_workers:
        raise ValueError("at least one worker peer ID is required")
    if len(set(normalized_workers)) != len(normalized_workers):
        raise ValueError("worker peer IDs must be unique")

    normalized_shards = sorted(shard_ids)
    if any(
        isinstance(shard_id, bool) or not isinstance(shard_id, int) or shard_id < 0
        for shard_id in normalized_shards
    ):
        raise ValueError("numeric shard IDs must be non-negative integers")
    if len(set(normalized_shards)) != len(normalized_shards):
        raise ValueError("numeric shard IDs must be unique")
    return {
        shard_id: normalized_workers[index % len(normalized_workers)]
        for index, shard_id in enumerate(normalized_shards)
    }


def size_balanced_numeric_shard_targets(
    shard_weights: list[int] | tuple[int, ...],
    worker_peer_ids: list[int] | tuple[int, ...],
    *,
    current_placement: dict[int, int] | None = None,
) -> dict[int, int]:
    """Place unequal numeric shards with deterministic count-balanced LPT bin packing.

    The weights are layout facts such as the number of physical vectors in each Orion shard.
    They do not depend on benchmark queries.  The optional current placement only chooses which
    physical worker receives each already-computed bin, minimizing live shard moves without
    changing the bins or their loads.
    """
    workers = list(worker_peer_ids)
    round_robin_numeric_shard_targets([], workers)
    if not shard_weights:
        raise ValueError("at least one numeric shard weight is required")

    weights: list[int] = []
    for shard_id, weight in enumerate(shard_weights):
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise ValueError(
                f"numeric shard {shard_id} weight must be a positive integer, got {weight!r}"
            )
        weights.append(weight)

    shard_count = len(weights)
    worker_count = len(workers)
    max_shards_per_worker = ceil(shard_count / worker_count)
    min_shards_per_worker = shard_count // worker_count
    bins: list[list[int]] = [[] for _ in workers]
    loads = [0 for _ in workers]
    for shard_id in sorted(range(shard_count), key=lambda index: (-weights[index], index)):
        candidates = [
            index
            for index, assigned in enumerate(bins)
            if len(assigned) < max_shards_per_worker
        ]
        target = min(candidates, key=lambda index: (loads[index], len(bins[index]), index))
        bins[target].append(shard_id)
        loads[target] += weights[shard_id]

    # A ceiling cap can still produce counts such as 6/6/4.  Rebalance only the shard count,
    # selecting the move that minimizes the resulting maximum load and then the load range.
    while min(map(len, bins)) < min_shards_per_worker:
        destination = min(
            (index for index in range(worker_count) if len(bins[index]) < min_shards_per_worker),
            key=lambda index: (loads[index], index),
        )
        candidates: list[tuple[tuple[int, int, int, int], int, int]] = []
        for source in range(worker_count):
            if len(bins[source]) <= min_shards_per_worker:
                continue
            for shard_id in bins[source]:
                candidate_loads = list(loads)
                candidate_loads[source] -= weights[shard_id]
                candidate_loads[destination] += weights[shard_id]
                score = (
                    max(candidate_loads),
                    max(candidate_loads) - min(candidate_loads),
                    weights[shard_id],
                    shard_id,
                )
                candidates.append((score, source, shard_id))
        if not candidates:
            raise RuntimeError("could not count-balance size-aware numeric shard bins")
        _score, source, shard_id = min(candidates)
        bins[source].remove(shard_id)
        bins[destination].append(shard_id)
        loads[source] -= weights[shard_id]
        loads[destination] += weights[shard_id]

    bin_to_worker = list(range(worker_count))
    if current_placement is not None:
        expected_ids = set(range(shard_count))
        if set(current_placement) != expected_ids:
            raise ValueError(
                "current numeric shard placement must cover the contiguous shard range: "
                f"expected={sorted(expected_ids)}, actual={sorted(current_placement)}"
            )
        if any(peer_id not in workers for peer_id in current_placement.values()):
            raise ValueError("current numeric shard placement contains a non-worker peer")
        if worker_count <= 8:
            def assignment_score(permutation: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
                moved = 0
                moved_weight = 0
                for bin_index, shard_ids in enumerate(bins):
                    peer_id = workers[permutation[bin_index]]
                    for shard_id in shard_ids:
                        if current_placement[shard_id] != peer_id:
                            moved += 1
                            moved_weight += weights[shard_id]
                return moved, moved_weight, permutation

            bin_to_worker = list(
                min(itertools.permutations(range(worker_count)), key=assignment_score)
            )

    return {
        shard_id: workers[bin_to_worker[bin_index]]
        for bin_index, shard_ids in enumerate(bins)
        for shard_id in shard_ids
    }


def _validate_numeric_auto_rf1_config(
    info: dict[str, Any],
    expected_shard_count: int,
) -> None:
    config = info.get("config") or {}
    params = config.get("params") or {}
    errors: list[str] = []
    if str(params.get("sharding_method") or "auto").lower() != "auto":
        errors.append("collection is not using sharding_method=auto")
    if params.get("shard_number") != expected_shard_count:
        errors.append(
            f"collection shard_number={params.get('shard_number')!r}, "
            f"expected {expected_shard_count}"
        )
    if params.get("replication_factor") != 1:
        errors.append(
            f"collection replication_factor={params.get('replication_factor')!r}, expected 1"
        )
    if errors:
        raise RuntimeError("; ".join(errors))


def validate_numeric_shard_round_robin_placement(
    info: dict[str, Any],
    cluster_info: dict[str, Any],
    worker_peer_ids: list[int] | tuple[int, ...],
    expected_shard_count: int,
    *,
    include_controller: bool = False,
) -> dict[str, Any]:
    """Strictly validate the final native RF=1 round-robin placement."""
    workers = list(worker_peer_ids)
    expected = round_robin_numeric_shard_targets(
        list(range(expected_shard_count)), workers
    )
    return _validate_numeric_shard_expected_placement(
        info,
        cluster_info,
        workers,
        expected_shard_count,
        expected,
        placement_mode="round_robin",
        mismatch_label="round-robin",
        include_controller=include_controller,
    )


def validate_numeric_shard_explicit_placement(
    info: dict[str, Any],
    cluster_info: dict[str, Any],
    worker_peer_ids: list[int] | tuple[int, ...],
    expected_shard_count: int,
    expected_placement: dict[int, int],
    *,
    include_controller: bool = False,
) -> dict[str, Any]:
    """Strictly validate one explicit native RF=1 placement."""
    return _validate_numeric_shard_expected_placement(
        info,
        cluster_info,
        list(worker_peer_ids),
        expected_shard_count,
        expected_placement,
        placement_mode="explicit",
        mismatch_label="explicit",
        include_controller=include_controller,
    )


def _validate_numeric_shard_expected_placement(
    info: dict[str, Any],
    cluster_info: dict[str, Any],
    workers: list[int],
    expected_shard_count: int,
    expected_placement: dict[int, int],
    *,
    placement_mode: str,
    mismatch_label: str,
    include_controller: bool = False,
) -> dict[str, Any]:
    _validate_numeric_auto_rf1_config(info, expected_shard_count)
    controller_peer_id = cluster_info.get("peer_id")
    if not workers:
        raise ValueError("at least one worker peer ID is required")
    if any(
        isinstance(peer_id, bool) or not isinstance(peer_id, int) or peer_id < 0
        for peer_id in workers
    ):
        raise ValueError("worker peer IDs must be non-negative integers")
    if len(set(workers)) != len(workers):
        raise ValueError("worker peer IDs must be unique")
    if controller_peer_id in workers and not include_controller:
        raise RuntimeError("worker peer IDs must not include the coordinator/controller peer")
    if include_controller and controller_peer_id not in workers:
        raise RuntimeError("all-peer placement must include the coordinator/controller peer")

    if not isinstance(expected_placement, dict):
        raise TypeError("expected numeric shard placement must be a dictionary")
    expected_ids = set(range(expected_shard_count))
    if set(expected_placement) != expected_ids:
        raise ValueError(
            "expected numeric shard placement must cover the contiguous shard range: "
            f"expected={sorted(expected_ids)}, actual={sorted(expected_placement)}"
        )
    for shard_id, peer_id in expected_placement.items():
        if isinstance(shard_id, bool) or not isinstance(shard_id, int) or shard_id < 0:
            raise ValueError(f"invalid expected numeric shard ID: {shard_id!r}")
        if isinstance(peer_id, bool) or not isinstance(peer_id, int) or peer_id < 0:
            raise ValueError(
                f"expected owner for numeric shard {shard_id} must be a non-negative integer"
            )
        if peer_id not in workers:
            raise ValueError(
                f"expected owner for numeric shard {shard_id} is not a worker peer: {peer_id}"
            )

    placement = numeric_shard_placement_from_cluster(
        cluster_info,
        expected_shard_count=expected_shard_count,
    )
    mismatches = {
        shard_id: {
            "actual": placement[shard_id],
            "expected": expected_placement[shard_id],
        }
        for shard_id in sorted(placement)
        if placement[shard_id] != expected_placement[shard_id]
    }
    if mismatches:
        raise RuntimeError(
            f"numeric shard {mismatch_label} placement mismatch: {mismatches}"
        )

    counts = Counter(placement.values())
    unexpected_peers = sorted(set(counts) - set(workers))
    if unexpected_peers:
        raise RuntimeError(f"numeric shards are placed on unexpected peers: {unexpected_peers}")
    counts_by_worker = {peer_id: int(counts.get(peer_id, 0)) for peer_id in workers}
    if max(counts_by_worker.values()) - min(counts_by_worker.values()) > 1:
        raise RuntimeError(f"numeric shard placement is imbalanced: {counts_by_worker}")
    if not include_controller and controller_peer_id in counts:
        raise RuntimeError("controller still owns one or more lower numeric shards")
    return {
        "valid": True,
        "placement_mode": placement_mode,
        "controller_peer_id": controller_peer_id,
        "shard_count": expected_shard_count,
        "replication_factor": 1,
        "placement": placement,
        "expected_placement": dict(expected_placement),
        "shards_per_worker": counts_by_worker,
        "includes_controller": include_controller,
        "shard_transfers": [],
    }


def wait_for_numeric_shard_cluster_idle(
    base_url: str,
    collection: str,
    *,
    expected_shard_count: int,
    timeout_sec: float = 3600.0,
    poll_interval_sec: float = 1.0,
) -> dict[str, Any]:
    """Wait until an RF=1 numeric collection has no transfers and all replicas are Active."""
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")
    if poll_interval_sec < 0:
        raise ValueError("poll_interval_sec must be non-negative")
    deadline = time.perf_counter() + timeout_sec
    last_cluster: dict[str, Any] | None = None
    while time.perf_counter() < deadline:
        last_cluster = collection_cluster_info(base_url, collection)
        if last_cluster is None:
            raise RuntimeError(f"could not read cluster placement for collection {collection!r}")
        if not (last_cluster.get("shard_transfers") or []):
            placement = numeric_shard_placement_from_cluster(
                last_cluster,
                expected_shard_count=expected_shard_count,
                require_active=False,
            )
            replicas = numeric_shard_replicas(last_cluster)
            if len(placement) == expected_shard_count and all(
                shard_replicas[0]["state"] == "Active"
                for shard_replicas in replicas.values()
            ):
                return last_cluster
        if poll_interval_sec:
            time.sleep(poll_interval_sec)
    raise TimeoutError(
        f"timed out waiting for numeric shard transfers to finish for {collection!r}; "
        f"last_cluster={last_cluster}"
    )


def _wait_for_numeric_shard_owner(
    base_url: str,
    collection: str,
    shard_id: int,
    from_peer_id: int,
    to_peer_id: int,
    *,
    expected_shard_count: int,
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_sec
    last_cluster: dict[str, Any] | None = None
    while time.perf_counter() < deadline:
        last_cluster = collection_cluster_info(base_url, collection)
        if last_cluster is None:
            raise RuntimeError(f"could not read cluster placement for collection {collection!r}")
        transfers = last_cluster.get("shard_transfers") or []
        if not transfers:
            placement = numeric_shard_placement_from_cluster(
                last_cluster,
                expected_shard_count=expected_shard_count,
                require_active=False,
            )
            replicas = numeric_shard_replicas(last_cluster)
            current_peer = placement.get(shard_id)
            current_state = replicas[shard_id][0]["state"]
            if current_peer == to_peer_id:
                if current_state == "Active":
                    return last_cluster
            elif current_peer not in {None, from_peer_id}:
                raise RuntimeError(
                    f"numeric shard {shard_id} moved to unexpected peer {current_peer}; "
                    f"expected {to_peer_id}"
                )
        if poll_interval_sec:
            time.sleep(poll_interval_sec)
    raise TimeoutError(
        f"timed out moving numeric shard {shard_id} from peer {from_peer_id} "
        f"to peer {to_peer_id}; last_cluster={last_cluster}"
    )


def _move_numeric_shards_to_targets(
    base_url: str,
    collection: str,
    worker_peer_ids: list[int] | tuple[int, ...],
    targets: dict[int, int],
    *,
    expected_shard_count: int,
    placement_mode: str,
    transfer_method: str = "stream_records",
    include_controller: bool = False,
    timeout_sec: float = 3600.0,
    poll_interval_sec: float = 1.0,
) -> dict[str, Any]:
    """Idempotently move native numeric shards to one exact peer layout."""
    if expected_shard_count <= 0:
        raise ValueError("expected_shard_count must be positive")
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")
    if poll_interval_sec < 0:
        raise ValueError("poll_interval_sec must be non-negative")
    if not isinstance(transfer_method, str) or not transfer_method:
        raise ValueError("transfer_method must be a non-empty string")

    info = collection_info(base_url, collection)
    _validate_numeric_auto_rf1_config(info, expected_shard_count)
    current_peer_id, peer_uris, _cluster = cluster_peer_map(base_url)
    if current_peer_id is None:
        raise RuntimeError("controller endpoint did not report its peer ID")
    workers = list(worker_peer_ids)
    round_robin_numeric_shard_targets([], workers)
    if current_peer_id in workers and not include_controller:
        raise RuntimeError("worker peer IDs must not include the coordinator/controller peer")
    if include_controller and current_peer_id not in workers:
        raise RuntimeError("all-peer placement must include the coordinator/controller peer")
    known_peer_ids = set(peer_uris)
    known_peer_ids.add(current_peer_id)
    unknown_workers = sorted(set(workers) - known_peer_ids)
    if unknown_workers:
        raise RuntimeError(f"worker peer IDs are not cluster members: {unknown_workers}")
    expected_ids = set(range(expected_shard_count))
    if set(targets) != expected_ids:
        raise ValueError(
            "target numeric shard placement must cover the contiguous shard range: "
            f"expected={sorted(expected_ids)}, actual={sorted(targets)}"
        )
    invalid_targets = {
        shard_id: peer_id
        for shard_id, peer_id in targets.items()
        if (
            isinstance(shard_id, bool)
            or not isinstance(shard_id, int)
            or shard_id < 0
            or isinstance(peer_id, bool)
            or not isinstance(peer_id, int)
            or peer_id not in workers
        )
    }
    if invalid_targets:
        raise ValueError(f"invalid worker-only numeric shard targets: {invalid_targets}")

    deadline = time.perf_counter() + timeout_sec

    def remaining_timeout() -> float:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(
                f"timed out placing numeric shards for collection {collection!r}"
            )
        return remaining

    idle_cluster = wait_for_numeric_shard_cluster_idle(
        base_url,
        collection,
        expected_shard_count=expected_shard_count,
        timeout_sec=remaining_timeout(),
        poll_interval_sec=poll_interval_sec,
    )
    placement = numeric_shard_placement_from_cluster(
        idle_cluster,
        expected_shard_count=expected_shard_count,
    )
    moves: list[dict[str, Any]] = []
    cluster_path = f"/collections/{urllib.parse.quote(collection, safe='')}/cluster"
    for shard_id in sorted(targets):
        idle_cluster = wait_for_numeric_shard_cluster_idle(
            base_url,
            collection,
            expected_shard_count=expected_shard_count,
            timeout_sec=remaining_timeout(),
            poll_interval_sec=poll_interval_sec,
        )
        placement = numeric_shard_placement_from_cluster(
            idle_cluster,
            expected_shard_count=expected_shard_count,
        )
        from_peer_id = placement[shard_id]
        to_peer_id = targets[shard_id]
        if from_peer_id == to_peer_id:
            continue
        body = {
            "move_shard": {
                "shard_id": shard_id,
                "from_peer_id": from_peer_id,
                "to_peer_id": to_peer_id,
                "method": transfer_method,
            }
        }
        request_json(base_url, "POST", cluster_path, body=body, timeout=remaining_timeout())
        _wait_for_numeric_shard_owner(
            base_url,
            collection,
            shard_id,
            from_peer_id,
            to_peer_id,
            expected_shard_count=expected_shard_count,
            timeout_sec=remaining_timeout(),
            poll_interval_sec=poll_interval_sec,
        )
        moves.append(body["move_shard"])

    final_info = collection_info(base_url, collection)
    final_cluster = wait_for_numeric_shard_cluster_idle(
        base_url,
        collection,
        expected_shard_count=expected_shard_count,
        timeout_sec=remaining_timeout(),
        poll_interval_sec=poll_interval_sec,
    )
    if placement_mode == "round_robin":
        audit = validate_numeric_shard_round_robin_placement(
            final_info,
            final_cluster,
            workers,
            expected_shard_count,
            include_controller=include_controller,
        )
    elif placement_mode == "explicit":
        audit = validate_numeric_shard_explicit_placement(
            final_info,
            final_cluster,
            workers,
            expected_shard_count,
            targets,
            include_controller=include_controller,
        )
    else:
        raise ValueError(f"unsupported numeric shard placement mode: {placement_mode!r}")
    return {**audit, "moves": moves, "transfer_method": transfer_method}


def move_numeric_shards_explicit(
    base_url: str,
    collection: str,
    worker_peer_ids: list[int] | tuple[int, ...],
    target_placement: dict[int, int],
    *,
    expected_shard_count: int,
    transfer_method: str = "snapshot",
    include_controller: bool = False,
    timeout_sec: float = 3600.0,
    poll_interval_sec: float = 1.0,
) -> dict[str, Any]:
    """Idempotently move native numeric shards to an explicit worker-only layout."""
    return _move_numeric_shards_to_targets(
        base_url,
        collection,
        worker_peer_ids,
        dict(target_placement),
        expected_shard_count=expected_shard_count,
        placement_mode="explicit",
        transfer_method=transfer_method,
        include_controller=include_controller,
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
    )


def move_numeric_shards_round_robin(
    base_url: str,
    collection: str,
    worker_peer_ids: list[int] | tuple[int, ...],
    *,
    expected_shard_count: int,
    transfer_method: str = "stream_records",
    include_controller: bool = False,
    timeout_sec: float = 3600.0,
    poll_interval_sec: float = 1.0,
) -> dict[str, Any]:
    """Idempotently move native numeric shards to an exact worker round-robin layout."""
    targets = round_robin_numeric_shard_targets(
        list(range(expected_shard_count)), list(worker_peer_ids)
    )
    return _move_numeric_shards_to_targets(
        base_url,
        collection,
        worker_peer_ids,
        targets,
        expected_shard_count=expected_shard_count,
        placement_mode="round_robin",
        transfer_method=transfer_method,
        include_controller=include_controller,
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
    )


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
            "cluster_shards_per_peer": {},
            "cluster_placement_valid": False,
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
    shards_per_peer: Counter[int] = Counter()
    controller_peer_id = cluster_info.get("peer_id")
    for shard in local_shards:
        peer_id = shard.get("peer_id", controller_peer_id)
        if peer_id is not None:
            shards_per_peer[int(peer_id)] += 1
    for shard in remote_shards:
        if shard.get("peer_id") is not None:
            shards_per_peer[int(shard["peer_id"])] += 1
    return {
        "cluster_shard_count": int(cluster_info.get("shard_count") or 0),
        "cluster_peer_count": len(peer_ids),
        "cluster_local_shards": len(local_shards),
        "cluster_remote_shards": len(remote_shards),
        "cluster_active_shards": active,
        "cluster_shards_per_peer": {
            str(peer_id): count for peer_id, count in sorted(shards_per_peer.items())
        },
        "cluster_placement_valid": bool(
            not local_shards
            and len(shards_per_peer) == 3
            and (max(shards_per_peer.values()) - min(shards_per_peer.values()) <= 1)
        ),
    }


def vector_config_from_collection(info: dict[str, Any]) -> dict[str, Any]:
    config = info.get("config") or {}
    params = config.get("params") or {}
    vectors = params.get("vectors") or {}
    if "size" in vectors:
        return vectors
    if isinstance(vectors, dict) and len(vectors) == 1:
        only = next(iter(vectors.values()))
        return only if isinstance(only, dict) else {}
    return {}


def collection_reuse_mismatches(
    info: dict[str, Any],
    cluster_info: dict[str, Any] | None,
    *,
    expected_dimension: int,
    expected_distance: str,
    expected_hnsw_m: int,
    expected_ef_construct: int,
    expected_points_count: int,
    expected_shard_count: int,
    expected_replication_factor: int,
    allowed_peer_ids: list[int],
    expected_routing_build_metadata: dict[str, Any] | None = None,
) -> list[str]:
    mismatches: list[str] = []
    config = info.get("config") or {}
    params = config.get("params") or {}
    vectors = vector_config_from_collection(info)
    hnsw = config.get("hnsw_config") or {}

    if int(vectors.get("size") or 0) != int(expected_dimension):
        mismatches.append(f"dimension={vectors.get('size')} expected={expected_dimension}")
    if str(vectors.get("distance") or "").lower() != str(expected_distance).lower():
        mismatches.append(f"distance={vectors.get('distance')} expected={expected_distance}")
    if int(hnsw.get("m") or 0) != int(expected_hnsw_m):
        mismatches.append(f"hnsw.m={hnsw.get('m')} expected={expected_hnsw_m}")
    if int(hnsw.get("ef_construct") or 0) != int(expected_ef_construct):
        mismatches.append(
            f"hnsw.ef_construct={hnsw.get('ef_construct')} expected={expected_ef_construct}"
        )
    if int(info.get("points_count") or 0) != int(expected_points_count):
        mismatches.append(
            f"points_count={info.get('points_count')} expected={expected_points_count}"
        )
    if int(params.get("replication_factor") or 0) != int(expected_replication_factor):
        mismatches.append(
            f"replication_factor={params.get('replication_factor')} "
            f"expected={expected_replication_factor}"
        )

    routing_metadata_validation = collection_routing_build_metadata_validation(
        info,
        expected_routing_build_metadata,
    )
    if routing_metadata_validation["status"] == "mismatch":
        mismatches.extend(
            f"routing build metadata {item}"
            for item in routing_metadata_validation["mismatches"]
        )

    if cluster_info is None:
        mismatches.append("collection cluster placement is unavailable")
        return mismatches

    local_shards = cluster_info.get("local_shards") or []
    remote_shards = cluster_info.get("remote_shards") or []
    all_shards = [*local_shards, *remote_shards]
    shard_keys = [str(shard.get("shard_key")) for shard in all_shards if shard.get("shard_key") is not None]
    unique_shard_keys = set(shard_keys)
    cluster_shard_count = int(cluster_info.get("shard_count") or len(unique_shard_keys))
    if cluster_shard_count != int(expected_shard_count) or len(unique_shard_keys) != int(expected_shard_count):
        mismatches.append(
            f"shard_count={cluster_shard_count}/{len(unique_shard_keys)} expected={expected_shard_count}"
        )
    duplicate_keys = sorted(key for key, count in Counter(shard_keys).items() if count != expected_replication_factor)
    if duplicate_keys:
        mismatches.append(f"replica count mismatch for shard keys {duplicate_keys[:5]}")
    inactive = [str(shard.get("shard_key")) for shard in all_shards if shard.get("state") != "Active"]
    if inactive:
        mismatches.append(f"inactive shards={inactive[:5]}")
    if local_shards:
        mismatches.append(f"controller owns {len(local_shards)} lower shards")
    allowed = set(map(int, allowed_peer_ids))
    actual_peers = {
        int(shard["peer_id"])
        for shard in remote_shards
        if shard.get("peer_id") is not None
    }
    if allowed and (not actual_peers or not actual_peers.issubset(allowed)):
        mismatches.append(
            f"placement peers={sorted(actual_peers)} expected subset of {sorted(allowed)}"
        )
    peer_counts = Counter(
        int(shard["peer_id"])
        for shard in remote_shards
        if shard.get("peer_id") is not None
    )
    if allowed and set(peer_counts) != allowed:
        mismatches.append(f"placement does not use all workers: {dict(peer_counts)}")
    if peer_counts and max(peer_counts.values()) - min(peer_counts.values()) > 1:
        mismatches.append(f"worker shard counts are imbalanced: {dict(peer_counts)}")
    return mismatches


def validate_existing_collection(
    base_url: str,
    collection: str,
    *,
    expected_dimension: int,
    expected_distance: str,
    expected_hnsw_m: int,
    expected_ef_construct: int,
    expected_points_count: int,
    expected_shard_count: int,
    allowed_peer_ids: list[int],
    expected_routing_build_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = collection_info(base_url, collection)
    routing_metadata_validation = collection_routing_build_metadata_validation(
        info,
        expected_routing_build_metadata,
    )
    mismatches = collection_reuse_mismatches(
        info,
        collection_cluster_info(base_url, collection),
        expected_dimension=expected_dimension,
        expected_distance=expected_distance,
        expected_hnsw_m=expected_hnsw_m,
        expected_ef_construct=expected_ef_construct,
        expected_points_count=expected_points_count,
        expected_shard_count=expected_shard_count,
        expected_replication_factor=1,
        allowed_peer_ids=allowed_peer_ids,
        expected_routing_build_metadata=expected_routing_build_metadata,
    )
    if mismatches:
        raise RuntimeError(
            f"refusing to reuse collection {collection!r}: " + "; ".join(mismatches)
        )
    return routing_metadata_validation


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
    require_fully_indexed: bool = False,
) -> dict:
    start = time.perf_counter()
    last_indexed: int | None = None
    last_change = start
    stable_since: float | None = None
    while True:
        info = collection_info(base_url, collection)
        cluster_info = collection_cluster_info(base_url, collection)
        indexed = int(info.get("indexed_vectors_count") or 0)
        points = int(info.get("points_count") or 0)
        if indexed != last_indexed:
            last_indexed = indexed
            last_change = time.perf_counter()
        optimizer_status = info.get("optimizer_status")
        optimizer_ok = optimizer_status is None or optimizer_status == "ok" or (
            isinstance(optimizer_status, dict) and optimizer_status.get("ok") is True
        )
        collection_ok = str(info.get("status") or "green").lower() == "green"
        cluster_shards = (
            (cluster_info.get("local_shards") or []) + (cluster_info.get("remote_shards") or [])
            if cluster_info
            else []
        )
        shards_active = not cluster_shards or all(
            shard.get("state") == "Active" for shard in cluster_shards
        )
        no_transfers = not cluster_info or not (cluster_info.get("shard_transfers") or [])
        indexing_complete = indexed >= expected_points or (
            not require_fully_indexed
            and points == expected_points
            and time.perf_counter() - last_change >= 30.0
        )
        stable = (
            points == expected_points
            and indexing_complete
            and optimizer_ok
            and collection_ok
            and shards_active
            and no_transfers
        )
        if stable:
            stable_since = stable_since or time.perf_counter()
            if time.perf_counter() - stable_since >= 5.0:
                return info
        else:
            stable_since = None
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
    shard_placement_map: dict[str, int] | None = None,
    vector_distance: str = "Cosine",
    routing_build_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shard_keys = [shard_key_for_id(shard_id) for shard_id in range(num_shards)]
    exists = collection_exists(base_url, collection)
    if reuse_existing and exists:
        validate_existing_collection(
            base_url,
            collection,
            expected_dimension=train.shape[1],
            expected_distance=vector_distance,
            expected_hnsw_m=hnsw_m,
            expected_ef_construct=ef_construct,
            expected_points_count=len(train),
            expected_shard_count=num_shards,
            allowed_peer_ids=peer_ids,
            expected_routing_build_metadata=routing_build_metadata,
        )
    created = not reuse_existing or not exists
    if created:
        delete_collection_if_exists(base_url, collection)
        create_collection(
            base_url,
            collection,
            train.shape[1],
            hnsw_m,
            ef_construct,
            vector_distance,
            routing_build_metadata,
        )
        for shard_id, shard_key in enumerate(shard_keys):
            placement = placement_for_shard_key(shard_id, peer_ids, shard_placement, shard_placement_map)
            create_shard_key(base_url, collection, shard_key, placement=placement)
        for shard_id, shard_key in enumerate(shard_keys):
            point_indices = np.where(assignments == shard_id)[0]
            for start_idx in range(0, len(point_indices), upload_batch_size):
                idx_chunk = point_indices[start_idx : start_idx + upload_batch_size]
                ids = (idx_chunk + 1).tolist()
                vectors = train[idx_chunk].tolist()
                upsert_points(base_url, collection, shard_key, ids, vectors)

    info = wait_collection_indexed(base_url, collection, len(train))
    routing_metadata_validation = collection_routing_build_metadata_validation(
        info,
        routing_build_metadata,
    )
    if created and routing_build_metadata is not None and not routing_metadata_validation["verified"]:
        raise RuntimeError(
            f"new collection {collection!r} did not preserve Orion routing build metadata: "
            f"{routing_metadata_validation['status']}"
        )
    cluster_summary = collection_cluster_summary(collection_cluster_info(base_url, collection))
    return {
        "collection": collection,
        "points_count": int(info.get("points_count") or 0),
        "indexed_vectors_count": int(info.get("indexed_vectors_count") or 0),
        "segments_count": int(info.get("segments_count") or 0),
        "shard_placement": shard_placement,
        "discovered_peer_count": len(peer_ids),
        **routing_build_metadata_validation_fields(routing_metadata_validation),
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
    shard_placement_map: dict[str, int] | None = None,
    vector_distance: str = "Cosine",
    routing_build_metadata: dict[str, Any] | None = None,
    force_hnsw_for_nonempty_shards: bool = False,
    upload_concurrency: int = 16,
    upload_queue_capacity_batches: int = 4,
) -> dict[str, Any]:
    shard_keys = [shard_key_for_id(shard_id) for shard_id in range(num_shards)]
    expected_points = total_assigned_points(point_to_shards)
    if (
        isinstance(upload_concurrency, bool)
        or not isinstance(upload_concurrency, int)
        or upload_concurrency <= 0
    ):
        raise ValueError("upload_concurrency must be a positive integer")
    if (
        isinstance(upload_queue_capacity_batches, bool)
        or not isinstance(upload_queue_capacity_batches, int)
        or upload_queue_capacity_batches <= 0
    ):
        raise ValueError("upload_queue_capacity_batches must be a positive integer")
    exists = collection_exists(base_url, collection)
    if reuse_existing and exists:
        validate_existing_collection(
            base_url,
            collection,
            expected_dimension=train.shape[1],
            expected_distance=vector_distance,
            expected_hnsw_m=hnsw_m,
            expected_ef_construct=ef_construct,
            expected_points_count=expected_points,
            expected_shard_count=num_shards,
            allowed_peer_ids=peer_ids,
            expected_routing_build_metadata=routing_build_metadata,
        )
    created = not reuse_existing or not exists
    upload_elapsed_seconds = 0.0
    upload_backpressure_events = 0
    max_observed_queue_batches = 0
    if created:
        delete_collection_if_exists(base_url, collection)
        create_collection(
            base_url,
            collection,
            train.shape[1],
            hnsw_m,
            ef_construct,
            vector_distance,
            routing_build_metadata,
            indexing_threshold=(1 if force_hnsw_for_nonempty_shards else 10),
        )
        for shard_id, shard_key in enumerate(shard_keys):
            placement = placement_for_shard_key(shard_id, peer_ids, shard_placement, shard_placement_map)
            create_shard_key(base_url, collection, shard_key, placement=placement)

        shard_queues = [
            queue.Queue(maxsize=upload_queue_capacity_batches)
            for _ in range(num_shards)
        ]
        request_budget = threading.BoundedSemaphore(upload_concurrency)
        stop_marker = object()
        stop_event = threading.Event()
        upload_started = time.perf_counter()

        def upload_one_shard(shard_id: int) -> tuple[int, int, int]:
            request_count = 0
            uploaded_copies = 0
            try:
                while not stop_event.is_set():
                    try:
                        item = shard_queues[shard_id].get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is stop_marker:
                        shard_queues[shard_id].task_done()
                        break
                    idx_chunk = item
                    source_ids = [int(point_idx) for point_idx in idx_chunk.tolist()]
                    ids = [
                        encode_copy_id(point_idx, shard_id, len(train))
                        for point_idx in source_ids
                    ]
                    # Each shard has one ordered producer task. Different shards upload
                    # concurrently while Qdrant independently builds their HNSW indexes.
                    with request_budget:
                        upsert_points(
                            base_url,
                            collection,
                            shard_keys[shard_id],
                            ids,
                            train[idx_chunk].tolist(),
                            source_ids=source_ids,
                        )
                    request_count += 1
                    uploaded_copies += len(idx_chunk)
                    shard_queues[shard_id].task_done()
            except BaseException:
                stop_event.set()
                raise
            return shard_id, uploaded_copies, request_count

        worker_count = num_shards
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(upload_one_shard, shard_id) for shard_id in range(num_shards)]

            def enqueue(shard_id: int, item: Any) -> None:
                nonlocal upload_backpressure_events, max_observed_queue_batches
                while True:
                    if stop_event.is_set():
                        for future in futures:
                            if future.done():
                                future.result()
                        raise RuntimeError("parallel shard upload stopped")
                    try:
                        shard_queues[shard_id].put(item, timeout=0.1)
                        max_observed_queue_batches = max(
                            max_observed_queue_batches,
                            shard_queues[shard_id].qsize(),
                        )
                        return
                    except queue.Full:
                        upload_backpressure_events += 1
                        for future in futures:
                            if future.done():
                                future.result()

            try:
                for shard_id, idx_chunk in iter_shard_upload_batches(
                    point_to_shards,
                    num_shards,
                    upload_batch_size,
                    upper_indices,
                ):
                    enqueue(shard_id, idx_chunk)
                for shard_id in range(num_shards):
                    enqueue(shard_id, stop_marker)
            except BaseException:
                stop_event.set()
                raise
            upload_results = [future.result() for future in futures]
        upload_elapsed_seconds = time.perf_counter() - upload_started
        uploaded_copies = sum(result[1] for result in upload_results)
        if uploaded_copies != expected_points:
            raise RuntimeError(
                f"parallel shard upload acknowledged {uploaded_copies} copies, expected {expected_points}"
            )
        upload_request_count = sum(result[2] for result in upload_results)
    else:
        worker_count = num_shards
        upload_request_count = 0

    info = wait_collection_indexed(
        base_url,
        collection,
        expected_points,
        require_fully_indexed=force_hnsw_for_nonempty_shards,
    )
    routing_metadata_validation = collection_routing_build_metadata_validation(
        info,
        routing_build_metadata,
    )
    if created and routing_build_metadata is not None and not routing_metadata_validation["verified"]:
        raise RuntimeError(
            f"new collection {collection!r} did not preserve Orion routing build metadata: "
            f"{routing_metadata_validation['status']}"
        )
    cluster_summary = collection_cluster_summary(collection_cluster_info(base_url, collection))
    if force_hnsw_for_nonempty_shards:
        indexed_vectors = int(info.get("indexed_vectors_count") or 0)
        if indexed_vectors != expected_points:
            raise RuntimeError(
                "Orion requires every non-empty shard to finish HNSW indexing: "
                f"indexed_vectors_count={indexed_vectors}, expected={expected_points}"
            )
        expected_segments = int(num_shards) * 2
        actual_segments = int(info.get("segments_count") or 0)
        if actual_segments != expected_segments:
            raise RuntimeError(
                "Orion requires one complete HNSW segment and one appendable "
                f"segment per shard: segments_count={actual_segments}, "
                f"expected={expected_segments}"
            )
    return {
        "collection": collection,
        "points_count": int(info.get("points_count") or 0),
        "indexed_vectors_count": int(info.get("indexed_vectors_count") or 0),
        "segments_count": int(info.get("segments_count") or 0),
        "shard_placement": shard_placement,
        "discovered_peer_count": len(peer_ids),
        "logical_points_count": int(len(train)),
        "assigned_points_count": int(expected_points),
        "ingest_mode": "parallel_per_shard_ordered_batches",
        "max_buffered_point_ids": int(
            num_shards * (upload_queue_capacity_batches + 1) * upload_batch_size
        ),
        "upload_worker_count": int(worker_count),
        "upload_concurrency": int(min(num_shards, upload_concurrency)),
        "upload_queue_capacity_batches": int(upload_queue_capacity_batches),
        "upload_request_count": int(upload_request_count),
        "upload_backpressure_events": int(upload_backpressure_events),
        "max_observed_queue_batches": int(max_observed_queue_batches),
        "upload_elapsed_seconds": float(upload_elapsed_seconds),
        "upload_copies_per_second": (
            float(expected_points / upload_elapsed_seconds)
            if upload_elapsed_seconds > 0
            else None
        ),
        "lower_hnsw_construction": "independent_full_multilayer_per_shard",
        "lower_hnsw_retains_non_base_layers": True,
        "routed_search_start_level": 0,
        "force_hnsw_for_nonempty_shards": bool(force_hnsw_for_nonempty_shards),
        **routing_build_metadata_validation_fields(routing_metadata_validation),
        **cluster_summary,
    }


def search_batch(
    base_url: str,
    collection: str,
    searches: list[dict[str, Any]],
    timeout: float = 600.0,
    encoded_body: bytes | None = None,
) -> list[list[tuple[float, int]]]:
    path = f"/collections/{urllib.parse.quote(collection, safe='')}/points/search/batch"
    payload = (
        request_json_encoded(
            base_url,
            "POST",
            path,
            encoded_body,
            timeout=timeout,
        )
        if encoded_body is not None
        else request_json(
            base_url,
            "POST",
            path,
            body={"searches": searches},
            timeout=timeout,
        )
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


def standard_dense_vector_request(
    query: list[float],
    top_k: int,
    *,
    api: str,
    vector_name: str = "",
    hnsw_ef: int | None = None,
) -> dict[str, Any]:
    """Build a normal client request with no Orion/custom-shard routing hints."""
    if api not in {"search", "query"}:
        raise ValueError("api must be 'search' or 'query'")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if hnsw_ef is not None and (
        isinstance(hnsw_ef, bool) or not isinstance(hnsw_ef, int) or hnsw_ef <= 0
    ):
        raise ValueError("hnsw_ef must be a positive integer when provided")
    vector = [float(value) for value in query]
    if not vector or not all(np.isfinite(value) for value in vector):
        raise ValueError("query must be a non-empty finite dense vector")
    request: dict[str, Any] = {
        "limit": top_k,
        "with_payload": False,
        "with_vector": False,
    }
    if hnsw_ef is not None:
        # A single standard query-level EF is part of Qdrant's public API and is
        # useful for the HashAll baseline. Native Orion callers leave it unset so
        # the server-side artifact remains the sole source of per-shard Dynamic EF.
        request["params"] = {"hnsw_ef": hnsw_ef}
    if api == "search":
        request["vector"] = (
            {"name": vector_name, "vector": vector} if vector_name else vector
        )
    else:
        request["query"] = vector
        if vector_name:
            request["using"] = vector_name
    return request


def standard_dense_vector_batch(
    base_url: str,
    collection: str,
    queries: np.ndarray | list[list[float]],
    top_k: int,
    *,
    api: str = "search",
    vector_name: str = "",
    hnsw_ef: int | None = None,
    timeout: float = 600.0,
) -> list[list[tuple[float, int]]]:
    """Execute a standard Search or Query batch against the coordinator.

    The request deliberately omits shard selectors, custom entry points,
    per-shard EF maps, source-ID payloads, and source-ID dedup hints. Thus an
    Orion collection exercises only its native server-side routing path.
    """
    if api not in {"search", "query"}:
        raise ValueError("api must be 'search' or 'query'")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    query_rows = np.asarray(queries)
    if query_rows.ndim != 2:
        raise ValueError("queries must be a two-dimensional dense array")
    searches = [
        standard_dense_vector_request(
            row.tolist(),
            top_k,
            api=api,
            vector_name=vector_name,
            hnsw_ef=hnsw_ef,
        )
        for row in query_rows
    ]
    endpoint = "search" if api == "search" else "query"
    payload = request_json(
        base_url,
        "POST",
        f"/collections/{urllib.parse.quote(collection, safe='')}/points/{endpoint}/batch",
        body={"searches": searches},
        timeout=timeout,
    )
    raw_results = payload.get("result")
    if not isinstance(raw_results, list) or len(raw_results) != len(searches):
        raise RuntimeError(
            f"standard {api} batch returned {len(raw_results) if isinstance(raw_results, list) else 'invalid'} "
            f"rows for {len(searches)} requests"
        )

    rows: list[list[tuple[float, int]]] = []
    for raw_row in raw_results:
        points = raw_row if api == "search" else (raw_row or {}).get("points")
        if not isinstance(points, list):
            raise RuntimeError(f"standard {api} batch returned an invalid result row: {raw_row!r}")
        row: list[tuple[float, int]] = []
        for point in points:
            point_id = point.get("id") if isinstance(point, dict) else None
            if isinstance(point_id, bool) or not isinstance(point_id, int):
                raise RuntimeError(
                    "native numeric-shard evaluation requires integer external point IDs; "
                    f"got {point_id!r}"
                )
            row.append((float(point["score"]), point_id))
        rows.append(row)
    return rows


def evaluate_standard_dense_vector_batches(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    batch_size: int,
    *,
    api: str = "search",
    vector_name: str = "",
    hnsw_ef: int | None = None,
    timeout: float = 600.0,
    include_per_query_metrics: bool = False,
) -> dict[str, Any]:
    """Measure recall/QPS using only standard coordinator Search/Query batches."""
    if api not in {"search", "query"}:
        raise ValueError("api must be 'search' or 'query'")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    query_rows = np.asarray(queries)
    neighbor_rows = np.asarray(neighbors)
    if query_rows.ndim != 2:
        raise ValueError("queries must be a two-dimensional dense array")
    if neighbor_rows.ndim != 2 or len(neighbor_rows) != len(query_rows):
        raise ValueError("neighbors must be a two-dimensional array aligned with queries")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if neighbor_rows.shape[1] < top_k:
        raise ValueError("top_k must be positive and covered by every ground-truth row")

    all_top_ids: list[list[int]] = []
    batch_latencies_ms: list[float] = []
    started = time.perf_counter()
    for start_idx in range(0, len(query_rows), batch_size):
        end_idx = min(start_idx + batch_size, len(query_rows))
        batch_started = time.perf_counter()
        result_rows = standard_dense_vector_batch(
            base_url,
            collection,
            query_rows[start_idx:end_idx],
            top_k,
            api=api,
            vector_name=vector_name,
            hnsw_ef=hnsw_ef,
            timeout=timeout,
        )
        batch_latencies_ms.append((time.perf_counter() - batch_started) * 1000.0)
        all_top_ids.extend(
            [point_id for _score, point_id in result_row[:top_k]]
            for result_row in result_rows
        )
    wall_s = time.perf_counter() - started
    recall_rows = per_query_recall_rows(all_top_ids, neighbor_rows, top_k)
    hits = sum(int(row["hits_at_k"]) for row in recall_rows)
    query_count = len(query_rows)
    result: dict[str, Any] = {
        "api": api,
        "hits": hits,
        "query_count": query_count,
        "recall_at_k": float(hits / (query_count * top_k)) if query_count else 0.0,
        "qps": float(query_count / wall_s) if wall_s > 0 else 0.0,
        "wall_s": wall_s,
        "search_batch_calls": len(batch_latencies_ms),
        "search_request_count": query_count,
        "avg_search_requests_per_query": 1.0 if query_count else 0.0,
        "avg_returned_candidates_per_query": (
            float(sum(len(row) for row in all_top_ids) / query_count) if query_count else 0.0
        ),
        **latency_percentile_fields(batch_latencies_ms),
    }
    if include_per_query_metrics:
        result["per_query_rows"] = recall_rows
    return result


def restrict_search_to_shard_keys(
    search: dict[str, Any],
    shard_keys: list[str],
) -> dict[str, Any]:
    restricted = dict(search)
    normalized_keys = [str(shard_key) for shard_key in shard_keys]
    restricted["shard_key"] = normalized_keys
    for field_name in ("hnsw_entry_points_by_shard", "hnsw_ef_by_shard"):
        values_by_shard = search.get(field_name)
        if values_by_shard is None:
            continue
        restricted_values = {
            shard_key: values_by_shard[shard_key]
            for shard_key in normalized_keys
            if shard_key in values_by_shard
        }
        if restricted_values:
            restricted[field_name] = restricted_values
        else:
            restricted.pop(field_name, None)
    return restricted


def shard_major_searches_for_query(search: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    shard_keys = [str(shard_key) for shard_key in search.get("shard_key") or []]
    if not shard_keys:
        return [("", dict(search))]

    entry_points_by_shard = search.get("hnsw_entry_points_by_shard")
    ef_by_shard = search.get("hnsw_ef_by_shard")
    expanded: list[tuple[str, dict[str, Any]]] = []
    for shard_key in shard_keys:
        single = dict(search)
        single["shard_key"] = [shard_key]

        if entry_points_by_shard is not None:
            single.pop("hnsw_entry_points_by_shard", None)
            entry_points = entry_points_by_shard.get(shard_key)
            if entry_points:
                single["hnsw_entry_points"] = list(entry_points)
            else:
                single.pop("hnsw_entry_points", None)

        if ef_by_shard is not None:
            single.pop("hnsw_ef_by_shard", None)
            if shard_key in ef_by_shard:
                params = dict(single.get("params") or {})
                params["hnsw_ef"] = int(ef_by_shard[shard_key])
                single["params"] = params

        expanded.append((shard_key, single))
    return expanded


def shard_major_flattened_searches(
    query_positions: list[int],
    searches: list[dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    entries: list[tuple[str, int, int, dict[str, Any]]] = []
    for original_order, (query_idx, search) in enumerate(zip(query_positions, searches)):
        for shard_key, single_search in shard_major_searches_for_query(search):
            entries.append((shard_key, int(query_idx), original_order, single_search))
    entries.sort(key=lambda item: (item[0], item[2]))
    return [query_idx for _shard_key, query_idx, _order, _search in entries], [
        search for _shard_key, _query_idx, _order, search in entries
    ]


def executed_search_count_for_plans(
    query_plans: list[dict[str, Any]],
    lower_execution_order: str,
) -> int:
    if lower_execution_order == "shard_major":
        return sum(
            len(shard_major_searches_for_query(search))
            for plan in query_plans
            for search in plan["searches"]
        )
    return sum(len(plan["searches"]) for plan in query_plans)


def query_shard_cost_traces(query_plans: list[dict[str, Any]]) -> list[dict[str, float]]:
    traces: list[dict[str, float]] = []
    for plan in query_plans:
        trace: dict[str, float] = defaultdict(float)
        for search in plan.get("searches", []):
            shard_keys = [str(shard_key) for shard_key in search.get("shard_key") or []]
            if not shard_keys:
                continue
            hnsw_ef = float((search.get("params") or {}).get("hnsw_ef", 0))
            ef_by_shard = search.get("hnsw_ef_by_shard") or {}
            for shard_key in shard_keys:
                trace[shard_key] += float(ef_by_shard.get(shard_key, hnsw_ef))
        traces.append(dict(sorted(trace.items())))
    return traces


def average_physical_peers_for_plans(
    query_plans: list[dict[str, Any]],
    shard_key_to_peer: dict[str, int] | None,
) -> float | None:
    if not shard_key_to_peer or not query_plans:
        return None
    counts: list[int] = []
    for plan in query_plans:
        peers: set[int] = set()
        for search in plan.get("searches", []):
            for shard_key in search.get("shard_key") or []:
                if shard_key in shard_key_to_peer:
                    peers.add(int(shard_key_to_peer[shard_key]))
        counts.append(len(peers))
    return float(sum(counts) / len(counts))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (q / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def latency_percentile_fields(batch_latencies_ms: list[float]) -> dict[str, float]:
    return {
        "latency_p50_ms": percentile(batch_latencies_ms, 50),
        "latency_p95_ms": percentile(batch_latencies_ms, 95),
        "latency_p99_ms": percentile(batch_latencies_ms, 99),
    }


def evaluate_query_peer_loads(
    traces: list[dict[str, float]],
    placement: dict[str, int],
) -> dict[str, float]:
    if not traces:
        return {
            "query_count": 0,
            "avg_query_max_peer_load": 0.0,
            "p95_query_max_peer_load": 0.0,
            "max_query_max_peer_load": 0.0,
            "avg_active_peers_per_query": 0.0,
            "max_total_peer_load": 0.0,
            "min_total_peer_load": 0.0,
            "peer_load_cv_pct": 0.0,
        }

    total_peer_load: dict[int, float] = defaultdict(float)
    query_max_loads: list[float] = []
    active_peer_counts: list[float] = []
    for trace in traces:
        peer_loads: dict[int, float] = defaultdict(float)
        for shard_key, cost in trace.items():
            if shard_key not in placement:
                raise ValueError(f"missing placement for shard {shard_key}")
            peer_id = int(placement[shard_key])
            peer_loads[peer_id] += float(cost)
            total_peer_load[peer_id] += float(cost)
        query_max_loads.append(max(peer_loads.values()) if peer_loads else 0.0)
        active_peer_counts.append(float(len(peer_loads)))

    peer_values = list(total_peer_load.values())
    mean_peer_load = sum(peer_values) / len(peer_values) if peer_values else 0.0
    if mean_peer_load > 0:
        variance = sum((value - mean_peer_load) ** 2 for value in peer_values) / len(peer_values)
        peer_load_cv_pct = (variance ** 0.5) / mean_peer_load * 100.0
    else:
        peer_load_cv_pct = 0.0

    return {
        "query_count": len(traces),
        "avg_query_max_peer_load": sum(query_max_loads) / len(query_max_loads),
        "p95_query_max_peer_load": percentile(query_max_loads, 95),
        "max_query_max_peer_load": max(query_max_loads),
        "avg_active_peers_per_query": sum(active_peer_counts) / len(active_peer_counts),
        "max_total_peer_load": max(peer_values) if peer_values else 0.0,
        "min_total_peer_load": min(peer_values) if peer_values else 0.0,
        "peer_load_cv_pct": peer_load_cv_pct,
    }


def greedy_method4_aware_placement(
    traces: list[dict[str, float]],
    peer_count: int,
    initial_placement: dict[str, int] | None = None,
) -> dict[str, int]:
    if peer_count <= 0:
        raise ValueError("peer_count must be positive")

    shard_total_cost: dict[str, float] = defaultdict(float)
    for trace in traces:
        for shard_key, cost in trace.items():
            shard_total_cost[str(shard_key)] += float(cost)

    if initial_placement:
        for shard_key in initial_placement:
            shard_total_cost.setdefault(str(shard_key), 0.0)

    shard_order = sorted(shard_total_cost, key=lambda shard_key: (-shard_total_cost[shard_key], shard_key))
    placement: dict[str, int] = {}
    total_peer_load = [0.0 for _ in range(peer_count)]

    for shard_key in shard_order:
        best_peer = 0
        best_score: tuple[float, float, float, int] | None = None
        for peer_id in range(peer_count):
            candidate = dict(placement)
            candidate[shard_key] = peer_id
            projected_peer_load = list(total_peer_load)
            projected_peer_load[peer_id] += shard_total_cost[shard_key]

            query_max_loads: list[float] = []
            for trace in traces:
                peer_loads: dict[int, float] = defaultdict(float)
                for trace_shard, cost in trace.items():
                    assigned_peer = candidate.get(trace_shard)
                    if assigned_peer is not None:
                        peer_loads[assigned_peer] += float(cost)
                query_max_loads.append(max(peer_loads.values()) if peer_loads else 0.0)
            avg_query_max = sum(query_max_loads) / len(query_max_loads) if query_max_loads else 0.0
            score = (
                avg_query_max,
                max(projected_peer_load),
                projected_peer_load[peer_id],
                peer_id,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_peer = peer_id

        placement[shard_key] = best_peer
        total_peer_load[best_peer] += shard_total_cost[shard_key]

    return dict(sorted(placement.items()))


def shard_key_sort_key(shard_key: str) -> tuple[int, str]:
    try:
        return int(shard_key.rsplit("_", 1)[1]), shard_key
    except (IndexError, ValueError):
        return 0, shard_key


def round_robin_simulated_placement(shard_keys: list[str], peer_count: int) -> dict[str, int]:
    if peer_count <= 0:
        raise ValueError("peer_count must be positive")
    return {
        shard_key: idx % peer_count
        for idx, shard_key in enumerate(sorted(set(shard_keys), key=shard_key_sort_key))
    }


def placement_improvement_pct(before: dict[str, float], after: dict[str, float], metric: str) -> float:
    before_value = float(before.get(metric, 0.0))
    after_value = float(after.get(metric, 0.0))
    if before_value <= 0:
        return 0.0
    return (before_value - after_value) / before_value * 100.0


def placement_simulation_summary(
    query_plans: list[dict[str, Any]],
    peer_count: int,
) -> dict[str, Any]:
    traces = query_shard_cost_traces(query_plans)
    shard_keys = sorted({shard_key for trace in traces for shard_key in trace}, key=shard_key_sort_key)
    round_robin = round_robin_simulated_placement(shard_keys, peer_count)
    method4_aware = greedy_method4_aware_placement(
        traces,
        peer_count,
        initial_placement=round_robin,
    )
    round_robin_metrics = evaluate_query_peer_loads(traces, round_robin)
    method4_aware_metrics = evaluate_query_peer_loads(traces, method4_aware)
    return {
        "peer_count": peer_count,
        "query_count": len(traces),
        "shard_count": len(shard_keys),
        "round_robin": round_robin_metrics,
        "method4_aware": method4_aware_metrics,
        "improvement_pct": {
            "avg_query_max_peer_load": placement_improvement_pct(
                round_robin_metrics,
                method4_aware_metrics,
                "avg_query_max_peer_load",
            ),
            "p95_query_max_peer_load": placement_improvement_pct(
                round_robin_metrics,
                method4_aware_metrics,
                "p95_query_max_peer_load",
            ),
            "max_total_peer_load": placement_improvement_pct(
                round_robin_metrics,
                method4_aware_metrics,
                "max_total_peer_load",
            ),
        },
        "placements": {
            "round_robin": round_robin,
            "method4_aware": method4_aware,
        },
    }


def physical_execution_summary(
    query_plans: list[dict[str, Any]],
    shard_key_to_peer: dict[str, int],
    top_k: int,
) -> dict[str, Any]:
    traces = query_shard_cost_traces(query_plans)
    logical_counts: list[float] = []
    physical_counts: list[float] = []
    logical_candidate_counts: list[float] = []
    physical_candidate_counts: list[float] = []
    assigned_ef_sums: list[float] = []
    max_peer_ef_sums: list[float] = []

    for trace in traces:
        peer_loads: dict[int, float] = defaultdict(float)
        for shard_key, cost in trace.items():
            if shard_key not in shard_key_to_peer:
                raise ValueError(f"missing physical peer for shard {shard_key}")
            peer_loads[int(shard_key_to_peer[shard_key])] += float(cost)

        logical_count = float(len(trace))
        physical_count = float(len(peer_loads))
        logical_counts.append(logical_count)
        physical_counts.append(physical_count)
        logical_candidate_counts.append(logical_count * int(top_k))
        physical_candidate_counts.append(physical_count * int(top_k))
        assigned_ef_sums.append(sum(float(cost) for cost in trace.values()))
        max_peer_ef_sums.append(max(peer_loads.values()) if peer_loads else 0.0)

    query_count = len(traces)
    logical_candidates = sum(logical_candidate_counts)
    physical_candidates = sum(physical_candidate_counts)
    logical_streams = sum(logical_counts)
    physical_streams = sum(physical_counts)

    return {
        "query_count": query_count,
        "top_k": int(top_k),
        "peer_count": len(set(int(peer) for peer in shard_key_to_peer.values())),
        "mapped_shard_count": len(shard_key_to_peer),
        "avg_logical_shards_per_query": sum(logical_counts) / query_count if query_count else 0.0,
        "p95_logical_shards_per_query": percentile(logical_counts, 95),
        "avg_physical_peers_per_query": sum(physical_counts) / query_count if query_count else 0.0,
        "p95_physical_peers_per_query": percentile(physical_counts, 95),
        "avg_controller_merge_stream_reduction_pct": (
            (logical_streams - physical_streams) / logical_streams * 100.0
            if logical_streams > 0
            else 0.0
        ),
        "avg_controller_candidate_reduction_pct": (
            (logical_candidates - physical_candidates) / logical_candidates * 100.0
            if logical_candidates > 0
            else 0.0
        ),
        "avg_assigned_ef_sum_per_query": sum(assigned_ef_sums) / query_count if query_count else 0.0,
        "p95_assigned_ef_sum_per_query": percentile(assigned_ef_sums, 95),
        "avg_max_peer_assigned_ef_per_query": (
            sum(max_peer_ef_sums) / query_count if query_count else 0.0
        ),
        "p95_max_peer_assigned_ef_per_query": percentile(max_peer_ef_sums, 95),
        "placement_peer_loads": evaluate_query_peer_loads(traces, shard_key_to_peer),
    }


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


class CompactRoutingManifest:
    def __init__(
        self,
        point_offsets: np.ndarray,
        shard_ids: np.ndarray,
        shard_keys: list[str],
        num_shards: int,
        source_id_dedup_block_size: int | None = None,
    ) -> None:
        self.point_offsets = point_offsets
        self.shard_ids = shard_ids
        self.shard_keys = shard_keys
        self.num_shards = int(num_shards)
        self.source_id_dedup_block_size = source_id_dedup_block_size


def build_compact_routing_manifest(
    point_to_shards: list[list[int]],
    num_shards: int,
    source_id_dedup_block_size: int | None = None,
) -> CompactRoutingManifest:
    offsets = np.zeros(len(point_to_shards) + 1, dtype=np.int64)
    flat_shards: list[int] = []
    for point_id, shard_ids in enumerate(point_to_shards):
        offsets[point_id] = len(flat_shards)
        for shard_id in shard_ids:
            flat_shards.append(int(shard_id))
    offsets[len(point_to_shards)] = len(flat_shards)
    return CompactRoutingManifest(
        point_offsets=offsets,
        shard_ids=np.asarray(flat_shards, dtype=np.int32),
        shard_keys=[shard_key_for_id(shard_id) for shard_id in range(num_shards)],
        num_shards=int(num_shards),
        source_id_dedup_block_size=source_id_dedup_block_size,
    )


def compact_entry_point_id(
    source_id: int,
    shard_id: int,
    source_id_dedup_block_size: int | None,
) -> int:
    if source_id_dedup_block_size is None:
        return int(source_id)
    return int(shard_id) * int(source_id_dedup_block_size) + int(source_id) + 1


def compact_routed_search_plan(
    query: list[float],
    upper_labels: list[int] | np.ndarray,
    manifest: CompactRoutingManifest,
    top_k: int,
    base_ef: int,
    factor: int,
    search_all_shards: bool,
    use_payload_source_id: bool,
    routed_result_limit_mode: str = "top_k",
    routed_result_limit_multiplier: float = 1,
) -> dict[str, Any]:
    eps_by_shard: list[list[int]] = [[] for _ in range(manifest.num_shards)]
    touched = np.zeros(manifest.num_shards, dtype=np.bool_)
    touched_shards: list[int] = []

    for label in list(upper_labels):
        point_id = int(label)
        if point_id < 0 or point_id + 1 >= len(manifest.point_offsets):
            continue
        start = int(manifest.point_offsets[point_id])
        end = int(manifest.point_offsets[point_id + 1])
        for pos in range(start, end):
            shard_id = int(manifest.shard_ids[pos])
            if shard_id < 0 or shard_id >= manifest.num_shards:
                continue
            if not bool(touched[shard_id]):
                touched[shard_id] = True
                touched_shards.append(shard_id)
            eps_by_shard[shard_id].append(
                compact_entry_point_id(
                    point_id,
                    shard_id,
                    manifest.source_id_dedup_block_size,
                )
            )

    if search_all_shards:
        shard_ids = list(range(manifest.num_shards))
    else:
        shard_ids = sorted(touched_shards)
    shard_keys = [manifest.shard_keys[shard_id] for shard_id in shard_ids]
    ef_values = [int(base_ef) + int(factor) * len(eps_by_shard[shard_id]) for shard_id in shard_ids]

    searches: list[dict[str, Any]] = []
    if shard_keys:
        request = {
            "vector": query,
            "limit": routed_result_limit(
                top_k,
                len(shard_keys),
                routed_result_limit_mode,
                routed_result_limit_multiplier,
            ),
            "params": {"hnsw_ef": int(max(ef_values))},
            "with_payload": ["source_id"] if use_payload_source_id else False,
            "with_vector": False,
            "shard_key": shard_keys,
            "hnsw_entry_points_by_shard": {
                manifest.shard_keys[shard_id]: eps_by_shard[shard_id]
                for shard_id in shard_ids
            },
            "hnsw_ef_by_shard": {
                manifest.shard_keys[shard_id]: int(ef_value)
                for shard_id, ef_value in zip(shard_ids, ef_values)
            },
        }
        if manifest.source_id_dedup_block_size is not None:
            request["source_id_dedup_block_size"] = int(manifest.source_id_dedup_block_size)
        searches.append(request)

    return {
        "searches": searches,
        "visited_shards": len(shard_keys),
        "upper_hits": int(sum(len(eps_by_shard[shard_id]) for shard_id in touched_shards)),
        "assigned_ef_sum": int(sum(ef_values)),
        "assigned_ef_count": len(ef_values),
    }


def build_compact_routed_search_plans(
    queries: np.ndarray,
    upper_labels: np.ndarray,
    manifest: CompactRoutingManifest,
    top_k: int,
    base_ef: int,
    factor: int,
    search_all_shards: bool,
    use_payload_source_id: bool,
    routed_result_limit_mode: str = "top_k",
    routed_result_limit_multiplier: float = 1,
) -> list[dict[str, Any]]:
    return [
        compact_routed_search_plan(
            query.tolist(),
            labels_row,
            manifest,
            top_k,
            base_ef,
            factor,
            search_all_shards,
            use_payload_source_id,
            routed_result_limit_mode,
            routed_result_limit_multiplier,
        )
        for query, labels_row in zip(queries, upper_labels)
    ]


def merge_topk_candidates(
    candidate_groups: list[list[tuple[float, int]]],
    top_k: int,
    score_higher_is_better: bool = True,
) -> list[int]:
    return [
        point_id
        for _score, point_id in merge_topk_scored_candidates(
            candidate_groups,
            top_k,
            score_higher_is_better,
        )
    ]


def per_query_recall_rows(
    top_ids_by_query: list[list[int]],
    neighbors: Any,
    top_k: int,
    query_index_offset: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_idx, top_ids in enumerate(top_ids_by_query):
        gt_ids = [int(point_id) for point_id in list(neighbors[local_idx])[:top_k]]
        hits = len(set(map(int, top_ids)) & set(gt_ids))
        rows.append(
            {
                "query_index": int(query_index_offset + local_idx),
                "hits_at_k": int(hits),
                "recall_at_k": float(hits / top_k),
                "retrieved_ids": " ".join(str(int(point_id)) for point_id in top_ids),
                "ground_truth_ids": " ".join(str(int(point_id)) for point_id in gt_ids),
            }
        )
    return rows


def merge_topk_scored_candidates(
    candidate_groups: list[list[tuple[float, int]]],
    top_k: int,
    score_higher_is_better: bool = True,
) -> list[tuple[float, int]]:
    best_by_id: dict[int, float] = {}
    for group in candidate_groups:
        for score, point_id in group:
            current = best_by_id.get(point_id)
            if current is None or (
                score > current if score_higher_is_better else score < current
            ):
                best_by_id[point_id] = score
    ordered = sorted(
        best_by_id.items(),
        key=lambda item: item[1],
        reverse=score_higher_is_better,
    )
    return [(score, point_id) for point_id, score in ordered[:top_k]]


def peer_local_premerge_candidates(
    shard_results_by_peer: dict[int, list[list[tuple[float, int]]]],
    top_k: int,
    score_higher_is_better: bool = True,
) -> list[tuple[int, list[tuple[float, int]]]]:
    return [
        (
            int(peer_id),
            merge_topk_scored_candidates(
                candidate_groups,
                top_k,
                score_higher_is_better,
            ),
        )
        for peer_id, candidate_groups in sorted(shard_results_by_peer.items())
    ]


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
    lower_execution_order: str = "query_major",
    shard_key_to_peer: dict[str, int] | None = None,
    score_higher_is_better: bool = True,
) -> dict[str, Any]:
    if label_to_shard is None and point_to_shards is None:
        raise ValueError("either label_to_shard or point_to_shards must be provided")
    if point_to_shards is not None and num_shards is None:
        raise ValueError("num_shards is required with point_to_shards")
    if routed_execution_mode not in {"grouped_by_ef", "compact_query_ef", "per_shard_multi_ep", "compact_multi_ep"}:
        raise ValueError(f"unsupported routed execution mode: {routed_execution_mode}")
    if lower_execution_order not in {"query_major", "shard_major"}:
        raise ValueError(f"unsupported lower execution order: {lower_execution_order}")

    total_hits = 0
    total_queries = len(queries)
    total_visited_shards = 0
    total_upper_hits = 0
    total_assigned_ef = 0
    total_assigned_ef_count = 0
    total_physical_peers = 0
    search_batch_calls = 0
    search_request_count = 0
    batch_latencies_ms: list[float] = []
    start = time.perf_counter()

    for start_idx in range(0, total_queries, batch_size):
        batch_started = time.perf_counter()
        chunk = queries[start_idx : start_idx + batch_size]
        upper_labels = compute_upper_labels(upper_index, chunk, upper_k)

        per_query_candidates: list[list[list[tuple[float, int]]]] = [[] for _ in range(len(chunk))]
        flat_searches: list[dict[str, Any]] = []
        flat_query_positions: list[int] = []

        for local_idx, labels_row in enumerate(upper_labels):
            owner_routing = label_to_shard is not None
            if owner_routing:
                shard_hit_counts = Counter()
                shard_to_eps: dict[str, list[int]] = defaultdict(list)
                for label in labels_row.tolist():
                    shard_key = label_to_shard.get(int(label))
                    if shard_key is not None:
                        shard_hit_counts[shard_key] += 1
                        shard_to_eps[shard_key].append(int(label))

                shard_keys, ef_values = shard_efs_from_upper_hits(
                    dict(shard_hit_counts), base_ef, factor
                )
                total_upper_hits += int(sum(shard_hit_counts.values()))
            else:
                assert point_to_shards is not None
                shard_to_eps = route_upper_labels_to_shard_eps(labels_row.tolist(), point_to_shards)
                shard_keys, ef_values = shard_efs_from_routed_eps(
                    shard_to_eps,
                    int(num_shards),
                    base_ef,
                    factor,
                    search_all_shards=search_all_shards,
                )
                total_upper_hits += int(sum(len(eps) for eps in shard_to_eps.values()))
            total_visited_shards += len(shard_keys)
            if shard_key_to_peer:
                total_physical_peers += len(
                    {shard_key_to_peer[key] for key in shard_keys if key in shard_key_to_peer}
                )

            if not shard_keys:
                continue
            def shard_entry_points(shard_key: str) -> list[int]:
                if owner_routing:
                    return shard_to_eps.get(shard_key, [])
                shard_id = int(shard_key.rsplit("_", 1)[1])
                return shard_to_eps.get(shard_id, [])

            if routed_execution_mode == "per_shard_multi_ep":
                total_assigned_ef += sum(ef_values)
                total_assigned_ef_count += len(ef_values)
                for shard_key, ef_value in zip(shard_keys, ef_values):
                    flat_query_positions.append(local_idx)
                    flat_searches.append(
                        search_request(
                            chunk[local_idx].tolist(),
                            top_k,
                            ef_value,
                            [shard_key],
                        use_payload_source_id,
                        shard_entry_points(shard_key),
                        source_id_dedup_block_size=source_id_dedup_block_size,
                    )
                )
            elif routed_execution_mode == "compact_multi_ep":
                total_assigned_ef += sum(ef_values)
                total_assigned_ef_count += len(ef_values)
                entry_points_by_shard: dict[str, list[int]] = {}
                ef_by_shard: dict[str, int] = {}
                for shard_key, ef_value in zip(shard_keys, ef_values):
                    entry_points_by_shard[shard_key] = shard_entry_points(shard_key)
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
            if lower_execution_order == "shard_major":
                flat_query_positions, flat_searches = shard_major_flattened_searches(
                    flat_query_positions,
                    flat_searches,
                )
            search_batch_calls += 1
            search_request_count += len(flat_searches)
            results = search_batch(base_url, collection, flat_searches)
            for local_idx, result in zip(flat_query_positions, results):
                per_query_candidates[local_idx].append(result)

        for local_idx, candidate_groups in enumerate(per_query_candidates):
            top_ids = merge_topk_candidates(candidate_groups, top_k, score_higher_is_better)
            gt = set(map(int, neighbors[start_idx + local_idx]))
            total_hits += len(set(top_ids) & gt)
        batch_latencies_ms.append((time.perf_counter() - batch_started) * 1000.0)

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
        "avg_physical_peers_per_query": (
            total_physical_peers / total_queries if shard_key_to_peer else None
        ),
        **latency_percentile_fields(batch_latencies_ms),
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
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    include_per_query_metrics: bool = False,
    score_higher_is_better: bool = True,
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
    if label_to_shard is None:
        assert point_to_shards is not None
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
    candidate_group_count = 0
    returned_candidate_count = 0
    per_query_rows: list[dict[str, Any]] = []
    batch_latencies_ms: list[float] = []

    for start_idx in range(0, total_queries, batch_size):
        end_idx = min(start_idx + batch_size, total_queries)
        batch_started = time.perf_counter()
        per_query_kwargs = {}
        if include_per_query_metrics:
            per_query_kwargs = {
                "include_per_query_metrics": True,
                "query_index_offset": start_idx,
            }
        result = execute_query_plans_once(
            base_url,
            collection,
            query_plans[start_idx:end_idx],
            neighbors[start_idx:end_idx],
            top_k,
            direct_peer_urls=direct_peer_urls,
            shard_key_to_peer=shard_key_to_peer,
            lower_execution_order=lower_execution_order,
            direct_peer_local_premerge=direct_peer_local_premerge,
            score_higher_is_better=score_higher_is_better,
            **per_query_kwargs,
        )
        batch_latencies_ms.append(
            float(result.get("batch_latency_ms", (time.perf_counter() - batch_started) * 1000.0))
        )
        total_hits += int(result["hits"])
        total_executed_queries += int(result["query_count"])
        total_visited_shards += int(result["visited_shards"])
        total_upper_hits += int(result["upper_hits"])
        total_assigned_ef += int(result["assigned_ef_sum"])
        total_assigned_ef_count += int(result["assigned_ef_count"])
        search_batch_calls += int(result["search_batch_calls"])
        search_request_count += int(result["search_request_count"])
        candidate_group_count += int(result.get("candidate_group_count", result.get("search_request_count", 0)))
        returned_candidate_count += int(result.get("returned_candidate_count", 0))
        if include_per_query_metrics:
            per_query_rows.extend(result.get("per_query_rows", []))

    wall = time.perf_counter() - start
    output = {
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
        "avg_candidate_groups_per_query": candidate_group_count / total_executed_queries,
        "avg_returned_candidates_per_query": returned_candidate_count / total_executed_queries,
        "avg_physical_peers_per_query": average_physical_peers_for_plans(
            query_plans, shard_key_to_peer
        ),
        **latency_percentile_fields(batch_latencies_ms),
    }
    if include_per_query_metrics:
        output["per_query_rows"] = per_query_rows
    return output


def evaluate_config_pipelined_routing(
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
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    include_per_query_metrics: bool = False,
    score_higher_is_better: bool = True,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if label_to_shard is None and point_to_shards is None:
        raise ValueError("either label_to_shard or point_to_shards must be provided")
    if point_to_shards is not None and num_shards is None:
        raise ValueError("num_shards is required with point_to_shards")

    total_queries = len(queries)
    ranges = [
        (start_idx, min(start_idx + batch_size, total_queries))
        for start_idx in range(0, total_queries, batch_size)
    ]

    def plan_range(
        start_idx: int,
        end_idx: int,
    ) -> tuple[int, int, list[dict[str, Any]], list[bytes] | None]:
        batch_queries = queries[start_idx:end_idx]
        upper_labels = compute_upper_labels(upper_index, batch_queries, upper_k)
        if label_to_shard is None:
            assert point_to_shards is not None
            plans = build_routed_search_plans(
                batch_queries,
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
            plans = [
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
                for query, labels_row in zip(batch_queries, upper_labels)
            ]
        encoded_stage_bodies = None
        if direct_peer_urls is None:
            encoded_stage_bodies = []
            for stage_items in query_plan_search_stages(plans):
                _query_positions, searches = coordinator_search_stage(
                    stage_items,
                    lower_execution_order,
                )
                encoded_stage_bodies.append(encode_search_batch_body(searches))
        return start_idx, end_idx, plans, encoded_stage_bodies

    total_hits = 0
    total_executed_queries = 0
    total_visited_shards = 0
    total_upper_hits = 0
    total_assigned_ef = 0
    total_assigned_ef_count = 0
    total_physical_peers = 0.0
    search_batch_calls = 0
    search_request_count = 0
    candidate_group_count = 0
    returned_candidate_count = 0
    per_query_rows: list[dict[str, Any]] = []
    batch_latencies_ms: list[float] = []

    start = time.perf_counter()
    if ranges:
        with ThreadPoolExecutor(max_workers=1) as planner:
            planned = planner.submit(plan_range, *ranges[0])
            for range_idx in range(len(ranges)):
                start_idx, end_idx, query_plans, encoded_stage_bodies = planned.result()
                if range_idx + 1 < len(ranges):
                    planned = planner.submit(plan_range, *ranges[range_idx + 1])

                batch_started = time.perf_counter()
                per_query_kwargs = {}
                if include_per_query_metrics:
                    per_query_kwargs = {
                        "include_per_query_metrics": True,
                        "query_index_offset": start_idx,
                    }
                result = execute_query_plans_once(
                    base_url,
                    collection,
                    query_plans,
                    neighbors[start_idx:end_idx],
                    top_k,
                    direct_peer_urls=direct_peer_urls,
                    shard_key_to_peer=shard_key_to_peer,
                    lower_execution_order=lower_execution_order,
                    direct_peer_local_premerge=direct_peer_local_premerge,
                    score_higher_is_better=score_higher_is_better,
                    preencoded_search_stage_bodies=encoded_stage_bodies,
                    **per_query_kwargs,
                )
                batch_latencies_ms.append(
                    float(
                        result.get(
                            "batch_latency_ms",
                            (time.perf_counter() - batch_started) * 1000.0,
                        )
                    )
                )
                total_hits += int(result["hits"])
                total_executed_queries += int(result["query_count"])
                total_visited_shards += int(result["visited_shards"])
                total_upper_hits += int(result["upper_hits"])
                total_assigned_ef += int(result["assigned_ef_sum"])
                total_assigned_ef_count += int(result["assigned_ef_count"])
                search_batch_calls += int(result["search_batch_calls"])
                search_request_count += int(result["search_request_count"])
                candidate_group_count += int(
                    result.get("candidate_group_count", result.get("search_request_count", 0))
                )
                returned_candidate_count += int(result.get("returned_candidate_count", 0))
                if shard_key_to_peer:
                    total_physical_peers += average_physical_peers_for_plans(
                        query_plans,
                        shard_key_to_peer,
                    ) * len(query_plans)
                if include_per_query_metrics:
                    per_query_rows.extend(result.get("per_query_rows", []))

    wall = time.perf_counter() - start
    output = {
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
        "avg_candidate_groups_per_query": candidate_group_count / total_executed_queries,
        "avg_returned_candidates_per_query": returned_candidate_count / total_executed_queries,
        "avg_physical_peers_per_query": (
            total_physical_peers / total_executed_queries if shard_key_to_peer else None
        ),
        **latency_percentile_fields(batch_latencies_ms),
    }
    if include_per_query_metrics:
        output["per_query_rows"] = per_query_rows
    return output


def evaluate_config_compact_materialized_routing(
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
    routed_execution_mode: str = "compact_multi_ep",
    compact_ef_mode: str = "max",
    routed_result_limit_mode: str = "top_k",
    routed_result_limit_multiplier: float = 1,
    source_id_dedup_block_size: int | None = None,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    include_per_query_metrics: bool = False,
    score_higher_is_better: bool = True,
) -> dict[str, Any]:
    if routed_execution_mode != "compact_multi_ep":
        raise ValueError("compact_materialized planning requires --routed-execution-mode compact_multi_ep")
    if compact_ef_mode != "max":
        raise ValueError("compact_materialized planning does not support compact EF reduction modes")
    if num_shards is None:
        raise ValueError("compact_materialized planning requires num_shards")
    if label_to_shard is not None:
        point_count = (
            len(point_to_shards)
            if point_to_shards is not None
            else max((int(label) for label in label_to_shard), default=-1) + 1
        )
        route_memberships: list[list[int]] = [[] for _ in range(point_count)]
        for label, shard_key in label_to_shard.items():
            point_id = int(label)
            if point_id < 0 or point_id >= point_count:
                raise ValueError(f"upper label {point_id} is outside compact route manifest")
            route_memberships[point_id] = [int(str(shard_key).rsplit("_", 1)[1])]
    elif point_to_shards is not None:
        route_memberships = point_to_shards
    else:
        raise ValueError(
            "compact_materialized planning requires label_to_shard or point_to_shards"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    manifest = build_compact_routing_manifest(
        route_memberships,
        int(num_shards),
        source_id_dedup_block_size,
    )

    total_queries = len(queries)
    start = time.perf_counter()
    upper_labels = compute_upper_labels(upper_index, queries, upper_k)
    query_plans = build_compact_routed_search_plans(
        queries,
        upper_labels,
        manifest,
        top_k,
        base_ef,
        factor,
        search_all_shards,
        use_payload_source_id,
        routed_result_limit_mode,
        routed_result_limit_multiplier,
    )

    total_hits = 0
    total_executed_queries = 0
    total_visited_shards = 0
    total_upper_hits = 0
    total_assigned_ef = 0
    total_assigned_ef_count = 0
    search_batch_calls = 0
    search_request_count = 0
    candidate_group_count = 0
    returned_candidate_count = 0
    per_query_rows: list[dict[str, Any]] = []
    batch_latencies_ms: list[float] = []

    for start_idx in range(0, total_queries, batch_size):
        end_idx = min(start_idx + batch_size, total_queries)
        batch_started = time.perf_counter()
        per_query_kwargs = {}
        if include_per_query_metrics:
            per_query_kwargs = {
                "include_per_query_metrics": True,
                "query_index_offset": start_idx,
            }
        result = execute_query_plans_once(
            base_url,
            collection,
            query_plans[start_idx:end_idx],
            neighbors[start_idx:end_idx],
            top_k,
            direct_peer_urls=direct_peer_urls,
            shard_key_to_peer=shard_key_to_peer,
            lower_execution_order=lower_execution_order,
            direct_peer_local_premerge=direct_peer_local_premerge,
            score_higher_is_better=score_higher_is_better,
            **per_query_kwargs,
        )
        batch_latencies_ms.append(
            float(result.get("batch_latency_ms", (time.perf_counter() - batch_started) * 1000.0))
        )
        total_hits += int(result["hits"])
        total_executed_queries += int(result["query_count"])
        total_visited_shards += int(result["visited_shards"])
        total_upper_hits += int(result["upper_hits"])
        total_assigned_ef += int(result["assigned_ef_sum"])
        total_assigned_ef_count += int(result["assigned_ef_count"])
        search_batch_calls += int(result["search_batch_calls"])
        search_request_count += int(result["search_request_count"])
        candidate_group_count += int(result.get("candidate_group_count", result.get("search_request_count", 0)))
        returned_candidate_count += int(result.get("returned_candidate_count", 0))
        if include_per_query_metrics:
            per_query_rows.extend(result.get("per_query_rows", []))

    wall = time.perf_counter() - start
    output = {
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
        "avg_candidate_groups_per_query": candidate_group_count / total_executed_queries,
        "avg_returned_candidates_per_query": returned_candidate_count / total_executed_queries,
        "avg_physical_peers_per_query": average_physical_peers_for_plans(
            query_plans, shard_key_to_peer
        ),
        **latency_percentile_fields(batch_latencies_ms),
    }
    if include_per_query_metrics:
        output["per_query_rows"] = per_query_rows
    return output


def routed_evaluator_for_planning_mode(mode: str) -> Any:
    if mode == "materialized":
        return evaluate_config_materialized_routing
    if mode == "compact_materialized":
        return evaluate_config_compact_materialized_routing
    if mode == "pipelined":
        return evaluate_config_pipelined_routing
    if mode == "per_batch":
        return evaluate_config
    raise ValueError(f"unsupported routed planning mode: {mode}")


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
    lower_execution_order: str = "query_major",
    include_per_query_metrics: bool = False,
    score_higher_is_better: bool = True,
    fixed_ef_shard_chunk_size: int = 0,
) -> dict[str, Any]:
    if lower_execution_order not in {"query_major", "shard_major"}:
        raise ValueError(f"unsupported lower execution order: {lower_execution_order}")
    total_hits = 0
    total_queries = len(queries)
    shard_keys, _ef_values = all_shard_keys_and_ef(num_shards, hnsw_ef)
    shard_key_chunks = fixed_ef_shard_key_chunks(
        shard_keys,
        fixed_ef_shard_chunk_size,
    )
    search_batch_calls = 0
    search_request_count = 0
    per_query_rows: list[dict[str, Any]] = []
    batch_latencies_ms: list[float] = []
    start = time.perf_counter()

    for start_idx in range(0, total_queries, batch_size):
        batch_started = time.perf_counter()
        chunk = queries[start_idx : start_idx + batch_size]
        per_query_candidates: list[list[list[tuple[float, int]]]] = [[] for _ in range(len(chunk))]
        for shard_key_chunk in shard_key_chunks:
            searches = [
                fixed_ef_search_requests(
                    query.tolist(),
                    top_k,
                    hnsw_ef,
                    shard_key_chunk,
                    use_payload_source_id,
                    source_id_dedup_block_size,
                    shard_chunk_size=0,
                    include_hnsw_ef_by_shard=True,
                )[0]
                for query in chunk
            ]
            query_positions = list(range(len(searches)))
            if lower_execution_order == "shard_major":
                query_positions, searches = shard_major_flattened_searches(
                    query_positions,
                    searches,
                )
            search_request_count += len(searches)
            if not searches:
                continue
            search_batch_calls += 1
            results = search_batch(base_url, collection, searches)
            for local_idx, result in zip(query_positions, results):
                per_query_candidates[local_idx].append(result)
        for local_idx, candidate_groups in enumerate(per_query_candidates):
            top_ids = merge_topk_candidates(candidate_groups, top_k, score_higher_is_better)
            gt = set(map(int, neighbors[start_idx + local_idx]))
            total_hits += len(set(top_ids) & gt)
        if include_per_query_metrics:
            per_query_rows.extend(
                per_query_recall_rows(
                    [
                        merge_topk_candidates(candidate_groups, top_k, score_higher_is_better)
                        for candidate_groups in per_query_candidates
                    ],
                    neighbors[start_idx : start_idx + len(chunk)],
                    top_k,
                    start_idx,
                )
            )
        batch_latencies_ms.append((time.perf_counter() - batch_started) * 1000.0)

    wall = time.perf_counter() - start
    output = {
        "recall_at_k": total_hits / (total_queries * top_k),
        "qps": total_queries / wall,
        "wall_s": wall,
        "avg_visited_shards": float(num_shards),
        "avg_upper_hits": 0.0,
        "avg_assigned_ef_per_visited_shard": float(hnsw_ef),
        "search_batch_calls": search_batch_calls,
        "avg_search_requests_per_query": search_request_count / total_queries,
        **latency_percentile_fields(batch_latencies_ms),
    }
    if include_per_query_metrics:
        output["per_query_rows"] = per_query_rows
    return output


def evaluate_kmeans_simple_nprobe_config(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    centroids: np.ndarray,
    nprobe: int,
    hnsw_ef: int,
    batch_size: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    include_per_query_metrics: bool = False,
    score_higher_is_better: bool = True,
    fixed_ef_shard_chunk_size: int = 0,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total_queries = len(queries)
    start = time.perf_counter()
    query_plans = build_kmeans_simple_nprobe_search_plans(
        queries,
        centroids,
        nprobe,
        top_k,
        hnsw_ef,
        use_payload_source_id,
        source_id_dedup_block_size,
        fixed_ef_shard_chunk_size,
    )

    total_hits = 0
    total_executed_queries = 0
    total_visited_shards = 0
    total_upper_hits = 0
    total_assigned_ef = 0
    total_assigned_ef_count = 0
    search_batch_calls = 0
    search_request_count = 0
    candidate_group_count = 0
    returned_candidate_count = 0
    per_query_rows: list[dict[str, Any]] = []
    batch_latencies_ms: list[float] = []

    for start_idx in range(0, total_queries, batch_size):
        end_idx = min(start_idx + batch_size, total_queries)
        batch_started = time.perf_counter()
        per_query_kwargs = {}
        if include_per_query_metrics:
            per_query_kwargs = {
                "include_per_query_metrics": True,
                "query_index_offset": start_idx,
            }
        result = execute_query_plans_once(
            base_url,
            collection,
            query_plans[start_idx:end_idx],
            neighbors[start_idx:end_idx],
            top_k,
            direct_peer_urls=direct_peer_urls,
            shard_key_to_peer=shard_key_to_peer,
            lower_execution_order=lower_execution_order,
            direct_peer_local_premerge=direct_peer_local_premerge,
            score_higher_is_better=score_higher_is_better,
            **per_query_kwargs,
        )
        batch_latencies_ms.append(
            float(result.get("batch_latency_ms", (time.perf_counter() - batch_started) * 1000.0))
        )
        total_hits += int(result["hits"])
        total_executed_queries += int(result["query_count"])
        total_visited_shards += int(result["visited_shards"])
        total_upper_hits += int(result["upper_hits"])
        total_assigned_ef += int(result["assigned_ef_sum"])
        total_assigned_ef_count += int(result["assigned_ef_count"])
        search_batch_calls += int(result["search_batch_calls"])
        search_request_count += int(result["search_request_count"])
        candidate_group_count += int(result.get("candidate_group_count", result.get("search_request_count", 0)))
        returned_candidate_count += int(result.get("returned_candidate_count", 0))
        if include_per_query_metrics:
            per_query_rows.extend(result.get("per_query_rows", []))

    wall = time.perf_counter() - start
    output = {
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
        "avg_candidate_groups_per_query": candidate_group_count / total_executed_queries,
        "avg_returned_candidates_per_query": returned_candidate_count / total_executed_queries,
        "avg_physical_peers_per_query": average_physical_peers_for_plans(
            query_plans, shard_key_to_peer
        ),
        **latency_percentile_fields(batch_latencies_ms),
    }
    if include_per_query_metrics:
        output["per_query_rows"] = per_query_rows
    return output


def evaluate_kmeans_simple_nprobe_config_batch(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: Any,
    top_k: int,
    centroids: np.ndarray,
    nprobe: int,
    hnsw_ef: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    score_higher_is_better: bool = True,
    fixed_ef_shard_chunk_size: int = 0,
) -> dict[str, Any]:
    query_plans = build_kmeans_simple_nprobe_search_plans(
        queries,
        centroids,
        nprobe,
        top_k,
        hnsw_ef,
        use_payload_source_id,
        source_id_dedup_block_size,
        fixed_ef_shard_chunk_size,
    )
    return execute_query_plans_once(
        base_url,
        collection,
        query_plans,
        neighbors,
        top_k,
        direct_peer_urls=direct_peer_urls,
        shard_key_to_peer=shard_key_to_peer,
        lower_execution_order=lower_execution_order,
        direct_peer_local_premerge=direct_peer_local_premerge,
        score_higher_is_better=score_higher_is_better,
    )


def evaluate_kmeans_simple_nprobe_config_concurrent(
    base_url: str,
    collection: str,
    queries: np.ndarray,
    neighbors: np.ndarray,
    top_k: int,
    centroids: np.ndarray,
    nprobe: int,
    hnsw_ef: int,
    batch_size: int,
    concurrency: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    score_higher_is_better: bool = True,
    fixed_ef_shard_chunk_size: int = 0,
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
        return evaluate_kmeans_simple_nprobe_config_batch(
            base_url,
            collection,
            queries[start_idx:end_idx],
            neighbors[start_idx:end_idx],
            top_k,
            centroids,
            nprobe,
            hnsw_ef,
            use_payload_source_id,
            source_id_dedup_block_size,
            direct_peer_urls=direct_peer_urls,
            shard_key_to_peer=shard_key_to_peer,
            lower_execution_order=lower_execution_order,
            direct_peer_local_premerge=direct_peer_local_premerge,
            score_higher_is_better=score_higher_is_better,
            fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
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


def fixed_ef_search_requests(
    query: list[float],
    top_k: int,
    hnsw_ef: int,
    shard_keys: list[str],
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
    shard_chunk_size: int = 0,
    include_hnsw_ef_by_shard: bool = True,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for shard_chunk in fixed_ef_shard_key_chunks(shard_keys, shard_chunk_size):
        per_shard_ef = (
            {shard_key: int(hnsw_ef) for shard_key in shard_chunk}
            if include_hnsw_ef_by_shard
            else None
        )
        requests.append(
            search_request(
                query,
                top_k,
                hnsw_ef,
                shard_chunk,
                use_payload_source_id,
                hnsw_ef_by_shard=per_shard_ef,
                source_id_dedup_block_size=source_id_dedup_block_size,
            )
        )
    return requests


def all_shard_search_plan(
    query: list[float],
    top_k: int,
    num_shards: int,
    hnsw_ef: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
    fixed_ef_shard_chunk_size: int = 0,
) -> dict[str, Any]:
    shard_keys, _ef_values = all_shard_keys_and_ef(num_shards, hnsw_ef)
    searches = fixed_ef_search_requests(
        query,
        top_k,
        hnsw_ef,
        shard_keys,
        use_payload_source_id,
        source_id_dedup_block_size,
        fixed_ef_shard_chunk_size,
    )
    # The uniform per-shard map is a transport hint for the native shard-major
    # peer-premerge path. Every shard still receives the same fixed EF, so the
    # Naive baseline remains an all-shards + fixed-EF algorithm.
    return {
        "searches": searches,
        "separate_http_search_stages": len(searches) > 1,
        "visited_shards": int(num_shards),
        "upper_hits": 0,
        "assigned_ef_sum": int(hnsw_ef) * int(num_shards),
        "assigned_ef_count": int(num_shards),
    }


def kmeans_simple_nprobe_shard_ids(
    query: np.ndarray,
    centroids: np.ndarray,
    nprobe: int,
) -> list[int]:
    if nprobe <= 0:
        raise ValueError("nprobe must be positive")
    if centroids.ndim != 2 or len(centroids) == 0:
        raise ValueError("centroids must be a non-empty 2D array")
    distances = squared_l2_distances(
        np.asarray(query, dtype=np.float32).reshape(1, -1),
        centroids.astype(np.float32, copy=False),
    )[0]
    limit = min(int(nprobe), len(centroids))
    return [int(shard_id) for shard_id in np.argsort(distances, kind="stable")[:limit].tolist()]


def kmeans_simple_nprobe_search_plan(
    query: np.ndarray | list[float],
    centroids: np.ndarray,
    nprobe: int,
    top_k: int,
    hnsw_ef: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
    fixed_ef_shard_chunk_size: int = 0,
) -> dict[str, Any]:
    shard_ids = kmeans_simple_nprobe_shard_ids(
        np.asarray(query, dtype=np.float32),
        centroids,
        nprobe,
    )
    shard_keys = [shard_key_for_id(shard_id) for shard_id in shard_ids]
    searches = fixed_ef_search_requests(
        np.asarray(query, dtype=np.float32).tolist(),
        top_k,
        hnsw_ef,
        shard_keys,
        use_payload_source_id,
        source_id_dedup_block_size,
        fixed_ef_shard_chunk_size,
    )
    # Keep the baseline's one fixed EF while expressing it per selected shard.
    # This lets the distributed controller group the request by physical peer
    # instead of materializing thousands of independent remote shard calls.
    return {
        "searches": searches,
        "separate_http_search_stages": len(searches) > 1,
        "visited_shards": len(shard_keys),
        "upper_hits": 0,
        "assigned_ef_sum": int(hnsw_ef) * len(shard_keys),
        "assigned_ef_count": len(shard_keys),
    }


def build_kmeans_simple_nprobe_search_plans(
    queries: np.ndarray,
    centroids: np.ndarray,
    nprobe: int,
    top_k: int,
    hnsw_ef: int,
    use_payload_source_id: bool,
    source_id_dedup_block_size: int | None = None,
    fixed_ef_shard_chunk_size: int = 0,
) -> list[dict[str, Any]]:
    return [
        kmeans_simple_nprobe_search_plan(
            query,
            centroids,
            nprobe,
            top_k,
            hnsw_ef,
            use_payload_source_id,
            source_id_dedup_block_size,
            fixed_ef_shard_chunk_size,
        )
        for query in queries
    ]


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
    fixed_ef_shard_chunk_size: int = 0,
) -> list[dict[str, Any]]:
    return [
        all_shard_search_plan(
            query.tolist(),
            top_k,
            num_shards,
            hnsw_ef,
            use_payload_source_id,
            source_id_dedup_block_size,
            fixed_ef_shard_chunk_size,
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


def query_plan_search_stages(
    query_plans: list[dict[str, Any]],
) -> list[list[tuple[int, dict[str, Any]]]]:
    if not any(bool(plan.get("separate_http_search_stages")) for plan in query_plans):
        flattened = [
            (query_idx, search)
            for query_idx, plan in enumerate(query_plans)
            for search in plan["searches"]
        ]
        return [flattened] if flattened else []

    stage_count = max(
        (
            len(plan["searches"])
            if bool(plan.get("separate_http_search_stages"))
            else 1
        )
        for plan in query_plans
    )
    stages: list[list[tuple[int, dict[str, Any]]]] = [
        [] for _ in range(stage_count)
    ]
    for query_idx, plan in enumerate(query_plans):
        searches = plan["searches"]
        if bool(plan.get("separate_http_search_stages")):
            for stage_idx, search in enumerate(searches):
                stages[stage_idx].append((query_idx, search))
        else:
            stages[0].extend((query_idx, search) for search in searches)
    return [stage for stage in stages if stage]


def coordinator_search_stage(
    stage_items: list[tuple[int, dict[str, Any]]],
    lower_execution_order: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    query_positions = [query_idx for query_idx, _search in stage_items]
    searches = [search for _query_idx, search in stage_items]
    if lower_execution_order == "shard_major":
        query_positions, searches = shard_major_flattened_searches(
            query_positions,
            searches,
        )
    return query_positions, searches


def execute_query_plans_once(
    base_url: str,
    collection: str,
    query_plans: list[dict[str, Any]],
    neighbors: Any,
    top_k: int,
    direct_peer_urls: dict[int, str] | None = None,
    shard_key_to_peer: dict[str, int] | None = None,
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    include_per_query_metrics: bool = False,
    query_index_offset: int = 0,
    score_higher_is_better: bool = True,
    preencoded_search_stage_bodies: list[bytes] | None = None,
) -> dict[str, Any]:
    batch_started = time.perf_counter()
    if lower_execution_order not in {"query_major", "shard_major"}:
        raise ValueError(f"unsupported lower execution order: {lower_execution_order}")
    if direct_peer_local_premerge and direct_peer_urls is None:
        raise ValueError("direct peer local pre-merge requires direct_peer_urls")
    if preencoded_search_stage_bodies is not None and direct_peer_urls is not None:
        raise ValueError("pre-encoded coordinator search bodies cannot be used with direct_peer_urls")

    per_query_candidates: list[list[list[tuple[float, int]]]] = [
        [] for _ in range(len(query_plans))
    ]

    search_batch_calls = 0
    search_request_count = 0
    search_stages = query_plan_search_stages(query_plans)
    if (
        preencoded_search_stage_bodies is not None
        and len(preencoded_search_stage_bodies) != len(search_stages)
    ):
        raise ValueError("pre-encoded search body count must match query-plan search stages")

    if direct_peer_urls is not None:
        if shard_key_to_peer is None:
            raise ValueError("shard_key_to_peer is required with direct_peer_urls")

        def run_peer_batch(
            peer_id: int,
            items: list[tuple[int, dict[str, Any]]],
        ) -> list[tuple[int, int, list[tuple[float, int]]]]:
            results = search_batch(
                direct_peer_urls[peer_id],
                collection,
                [search for _query_idx, search in items],
            )
            return [
                (peer_id, query_idx, result)
                for (query_idx, _search), result in zip(items, results)
            ]

        peer_results: list[tuple[int, int, list[tuple[float, int]]]] = []
        for stage_items in search_stages:
            peer_batches: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
            for query_idx, search in stage_items:
                if lower_execution_order == "shard_major":
                    for shard_key, single_search in shard_major_searches_for_query(search):
                        if not shard_key:
                            raise ValueError("shard_major direct_peer execution requires shard_key")
                        if shard_key not in shard_key_to_peer:
                            raise ValueError(f"missing peer mapping for shard key {shard_key}")
                        peer_id = int(shard_key_to_peer[shard_key])
                        if peer_id not in direct_peer_urls:
                            raise ValueError(f"missing direct HTTP URL for peer {peer_id}")
                        peer_batches[peer_id].append((query_idx, single_search))
                else:
                    keys_by_peer: dict[int, list[str]] = defaultdict(list)
                    for shard_key in search.get("shard_key") or []:
                        if shard_key not in shard_key_to_peer:
                            raise ValueError(f"missing peer mapping for shard key {shard_key}")
                        keys_by_peer[int(shard_key_to_peer[shard_key])].append(shard_key)
                    for peer_id, shard_keys in keys_by_peer.items():
                        if peer_id not in direct_peer_urls:
                            raise ValueError(f"missing direct HTTP URL for peer {peer_id}")
                        split_search = restrict_search_to_shard_keys(search, shard_keys)
                        peer_batches[peer_id].append((query_idx, split_search))

            if lower_execution_order == "shard_major":
                for items in peer_batches.values():
                    items.sort(
                        key=lambda item: (
                            (item[1].get("shard_key") or [""])[0],
                            item[0],
                        )
                    )

            search_batch_calls += len(peer_batches)
            search_request_count += sum(len(items) for items in peer_batches.values())

            if len(peer_batches) == 1:
                stage_results = [
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
                    stage_results = [
                        result
                        for future in futures
                        for result in future.result()
                    ]
            peer_results.extend(stage_results)

        if direct_peer_local_premerge:
            per_query_peer_candidates: list[dict[int, list[list[tuple[float, int]]]]] = [
                defaultdict(list) for _ in range(len(query_plans))
            ]
            for peer_id, local_idx, result in peer_results:
                per_query_peer_candidates[local_idx][peer_id].append(result)
            for local_idx, shard_results_by_peer in enumerate(per_query_peer_candidates):
                for _peer_id, premerged in peer_local_premerge_candidates(
                    shard_results_by_peer,
                    top_k,
                    score_higher_is_better,
                ):
                    per_query_candidates[local_idx].append(premerged)
        else:
            for _peer_id, local_idx, result in peer_results:
                per_query_candidates[local_idx].append(result)
    else:
        for stage_idx, stage_items in enumerate(search_stages):
            query_positions, flat_searches = coordinator_search_stage(
                stage_items,
                lower_execution_order,
            )

            search_request_count += len(flat_searches)
            if not flat_searches:
                continue
            search_batch_calls += 1
            encoded_body = (
                preencoded_search_stage_bodies[stage_idx]
                if preencoded_search_stage_bodies is not None
                else None
            )
            results = (
                search_batch(
                    base_url,
                    collection,
                    flat_searches,
                    encoded_body=encoded_body,
                )
                if encoded_body is not None
                else search_batch(base_url, collection, flat_searches)
            )
            for local_idx, result in zip(query_positions, results):
                per_query_candidates[local_idx].append(result)

    candidate_group_count = sum(len(candidate_groups) for candidate_groups in per_query_candidates)
    returned_candidate_count = sum(
        len(group)
        for candidate_groups in per_query_candidates
        for group in candidate_groups
    )

    hits = 0
    top_ids_by_query: list[list[int]] = []
    for offset, candidate_groups in enumerate(per_query_candidates):
        top_ids = merge_topk_candidates(candidate_groups, top_k, score_higher_is_better)
        top_ids_by_query.append(top_ids)
        gt = set(map(int, neighbors[offset]))
        hits += len(set(top_ids) & gt)
    result = {
        "hits": hits,
        "query_count": len(query_plans),
        "visited_shards": sum(int(plan["visited_shards"]) for plan in query_plans),
        "upper_hits": sum(int(plan["upper_hits"]) for plan in query_plans),
        "assigned_ef_sum": sum(int(plan["assigned_ef_sum"]) for plan in query_plans),
        "assigned_ef_count": sum(int(plan["assigned_ef_count"]) for plan in query_plans),
        "search_batch_calls": search_batch_calls,
        "search_request_count": search_request_count,
        "candidate_group_count": candidate_group_count,
        "returned_candidate_count": returned_candidate_count,
        "avg_candidate_groups_per_query": (
            candidate_group_count / len(query_plans) if query_plans else 0.0
        ),
        "avg_returned_candidates_per_query": (
            returned_candidate_count / len(query_plans) if query_plans else 0.0
        ),
        "batch_latency_ms": (time.perf_counter() - batch_started) * 1000.0,
    }
    if include_per_query_metrics:
        result["per_query_rows"] = per_query_recall_rows(
            top_ids_by_query,
            neighbors,
            top_k,
            query_index_offset,
        )
    return result


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
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    score_higher_is_better: bool = True,
) -> dict[str, Any]:
    if label_to_shard is None and point_to_shards is None:
        raise ValueError("either label_to_shard or point_to_shards must be provided")
    if point_to_shards is not None and num_shards is None:
        raise ValueError("num_shards is required with point_to_shards")

    upper_labels = compute_upper_labels(upper_index, queries, upper_k)
    if label_to_shard is None:
        assert point_to_shards is not None
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
        lower_execution_order=lower_execution_order,
        direct_peer_local_premerge=direct_peer_local_premerge,
        score_higher_is_better=score_higher_is_better,
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
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    score_higher_is_better: bool = True,
    fixed_ef_shard_chunk_size: int = 0,
) -> dict[str, Any]:
    query_plans = build_all_shard_search_plans(
        queries,
        top_k,
        num_shards,
        hnsw_ef,
        use_payload_source_id,
        source_id_dedup_block_size,
        fixed_ef_shard_chunk_size,
    )
    return execute_query_plans_once(
        base_url,
        collection,
        query_plans,
        neighbors,
        top_k,
        direct_peer_urls=direct_peer_urls,
        shard_key_to_peer=shard_key_to_peer,
        lower_execution_order=lower_execution_order,
        direct_peer_local_premerge=direct_peer_local_premerge,
        score_higher_is_better=score_higher_is_better,
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
    candidate_group_count = sum(
        int(result.get("candidate_group_count", result.get("search_request_count", 0)))
        for result in batch_results
    )
    returned_candidate_count = sum(
        int(result.get("returned_candidate_count", 0))
        for result in batch_results
    )
    batch_latencies_ms = [
        float(result["batch_latency_ms"])
        for result in batch_results
        if isinstance(result.get("batch_latency_ms"), (int, float))
    ]

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
        "avg_candidate_groups_per_query": candidate_group_count / total_queries,
        "avg_returned_candidates_per_query": returned_candidate_count / total_queries,
        **latency_percentile_fields(batch_latencies_ms),
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
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    score_higher_is_better: bool = True,
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
            lower_execution_order=lower_execution_order,
            direct_peer_local_premerge=direct_peer_local_premerge,
            score_higher_is_better=score_higher_is_better,
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
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    score_higher_is_better: bool = True,
    fixed_ef_shard_chunk_size: int = 0,
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
            lower_execution_order=lower_execution_order,
            direct_peer_local_premerge=direct_peer_local_premerge,
            score_higher_is_better=score_higher_is_better,
            fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
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
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    score_higher_is_better: bool = True,
) -> dict[str, Any]:
    result = execute_query_plans_once(
        base_url,
        collection,
        query_plans[start_idx:end_idx],
        neighbors[start_idx:end_idx],
        top_k,
        direct_peer_urls=direct_peer_urls,
        shard_key_to_peer=shard_key_to_peer,
        lower_execution_order=lower_execution_order,
        direct_peer_local_premerge=direct_peer_local_premerge,
        score_higher_is_better=score_higher_is_better,
    )
    return result


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
    lower_execution_order: str = "query_major",
    direct_peer_local_premerge: bool = False,
    score_higher_is_better: bool = True,
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

    start = time.perf_counter()
    if concurrency == 1:
        batch_results = []
        for start_idx, end_idx in ranges:
            result = evaluate_preplanned_search_batch(
                base_url,
                collection,
                query_plans,
                neighbors,
                top_k,
                start_idx,
                end_idx,
                direct_peer_urls,
                shard_key_to_peer,
                lower_execution_order,
                direct_peer_local_premerge,
                score_higher_is_better,
            )
            batch_results.append(result)
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
                    lower_execution_order,
                    direct_peer_local_premerge,
                    score_higher_is_better,
                )
                for start_idx, end_idx in ranges
            ]
            batch_results = [future.result() for future in futures]

    wall = time.perf_counter() - start
    return aggregate_timed_batch_results(batch_results, wall, top_k)


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
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_provenance(repo: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo or Path(__file__).resolve().parents[1])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, text=True, capture_output=True
        ).stdout
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        status_lines = [line for line in status.splitlines() if line.strip()]
        tracked_lines = [line for line in tracked_status.splitlines() if line.strip()]
        return {
            "root": str(root),
            "commit": commit,
            "dirty": bool(status_lines),
            "tracked_dirty": bool(tracked_lines),
            "untracked_entry_count": max(0, len(status_lines) - len(tracked_lines)),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"root": str(root), "error": str(exc)}


def load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def worker_shard_point_summary(
    sample_rows: list[dict[str, Any]],
    shard_key_to_peer: dict[str, int],
) -> dict[str, Any]:
    by_peer: dict[int, dict[str, int]] = defaultdict(lambda: {"shard_count": 0, "points_count": 0})
    for row in sample_rows:
        shard_key = str(row.get("shard_key") or "")
        if shard_key not in shard_key_to_peer:
            continue
        peer_id = int(shard_key_to_peer[shard_key])
        by_peer[peer_id]["shard_count"] += 1
        points_count = row.get("points_count")
        if isinstance(points_count, (int, float)):
            by_peer[peer_id]["points_count"] += int(points_count)
    return {str(peer_id): values for peer_id, values in sorted(by_peer.items())}


def capture_controller_transport_resources(
    deployment_manifest: dict[str, Any] | None,
    p2p_port: int = 6335,
) -> dict[str, Any]:
    nodes = (deployment_manifest or {}).get("nodes") or []
    controller = next(
        (node for node in nodes if str(node.get("role") or "") == "controller"),
        None,
    )
    if controller is None:
        return {
            "available": False,
            "error": "deployment manifest has no controller node",
        }
    host = str(controller.get("ssh_host") or "localhost")
    container = str(controller.get("container_name") or "")
    if not container:
        return {
            "available": False,
            "error": "controller node has no container_name",
        }

    remote_port_hex = f"{int(p2p_port):04X}"
    script = (
        "fd=$(ls /proc/1/fd | wc -l); "
        "p2p=$(awk 'NR > 1 && $4 == \"01\" && $3 ~ /:"
        f"{remote_port_hex}"
        "$/ { count++ } END { print count + 0 }' "
        "/proc/1/net/tcp /proc/1/net/tcp6 2>/dev/null); "
        "echo \"$fd $p2p\""
    )
    docker_command = [
        "sudo",
        "-n",
        "docker",
        "exec",
        container,
        "sh",
        "-lc",
        script,
    ]
    if host in {"localhost", "127.0.0.1", socket.gethostname(), socket.getfqdn()}:
        command = docker_command
    else:
        command = ["ssh", "-o", "BatchMode=yes", host, shlex.join(docker_command)]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20.0)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "host": host,
            "container": container,
            "error": str(exc),
        }
    if result.returncode != 0:
        return {
            "available": False,
            "host": host,
            "container": container,
            "returncode": result.returncode,
            "stderr": result.stderr.strip(),
        }
    fields = result.stdout.split()
    if len(fields) != 2:
        return {
            "available": False,
            "host": host,
            "container": container,
            "error": f"unexpected transport resource output: {result.stdout!r}",
        }
    try:
        fd_count, p2p_established = map(int, fields)
    except ValueError:
        return {
            "available": False,
            "host": host,
            "container": container,
            "error": f"non-integer transport resource output: {result.stdout!r}",
        }
    worker_count = sum(
        1 for node in nodes if str(node.get("role") or "").startswith("qdrant_shard_")
    )
    return {
        "available": True,
        "host": host,
        "container": container,
        "p2p_port": int(p2p_port),
        "fd_count": fd_count,
        "p2p_established": p2p_established,
        "worker_count": worker_count,
        "expected_steady_state_p2p_connections": worker_count * 2,
    }


def audit_transport_resources(
    start: dict[str, Any],
    end: dict[str, Any],
) -> dict[str, Any]:
    if not start.get("available") or not end.get("available"):
        return {
            "valid": False,
            "start": start,
            "end": end,
            "error": "controller transport resources could not be captured",
        }
    expected = max(
        int(start.get("expected_steady_state_p2p_connections") or 0),
        int(end.get("expected_steady_state_p2p_connections") or 0),
    )
    tolerance = max(2, expected)
    clean_limit = expected + tolerance
    start_p2p = int(start["p2p_established"])
    end_p2p = int(end["p2p_established"])
    fd_delta = int(end["fd_count"]) - int(start["fd_count"])
    p2p_delta = end_p2p - start_p2p
    checks = {
        "start_p2p_within_clean_limit": start_p2p <= clean_limit,
        "end_p2p_within_clean_limit": end_p2p <= clean_limit,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "start": start,
        "end": end,
        "delta": {
            "fd_count": fd_delta,
            "p2p_established": p2p_delta,
        },
        "expected_steady_state_p2p_connections": expected,
        "tolerance": tolerance,
        "clean_limit": clean_limit,
        "note": (
            "The current Qdrant transport pool maintains two steady-state channels per "
            "worker URI. The tolerance permits one additional pool-width while rejecting "
            "the hundreds of transient connections produced by ungrouped shard fan-out."
        ),
    }


def collect_container_log_tails(
    deployment_manifest: dict[str, Any] | None,
    tail_lines: int = 200,
    since_epoch: int | None = None,
) -> dict[str, Any]:
    logs: dict[str, Any] = {}
    for node in (deployment_manifest or {}).get("nodes") or []:
        role = str(node.get("role") or node.get("ssh_host") or "unknown")
        host = str(node.get("ssh_host") or "localhost")
        container = str(node.get("container_name") or "")
        if not container:
            continue
        docker_command = [
            "sudo",
            "-n",
            "docker",
            "logs",
        ]
        if since_epoch is not None:
            docker_command.extend(["--since", str(int(since_epoch))])
        docker_command.extend(["--tail", str(tail_lines), container])
        if host in {"localhost", "127.0.0.1", socket.gethostname(), socket.getfqdn()}:
            command = docker_command
        else:
            command = ["ssh", "-o", "BatchMode=yes", host, shlex.join(docker_command)]
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=20.0)
            logs[role] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except (OSError, subprocess.SubprocessError) as exc:
            logs[role] = {"error": str(exc)}
    return logs


def audit_runtime_logs(logs: dict[str, Any]) -> dict[str, Any]:
    matches: dict[str, list[dict[str, str]]] = {
        name: [] for name in RUNTIME_LOG_ERROR_PATTERNS
    }
    collection_errors: list[dict[str, str]] = []
    for role, payload in logs.items():
        if payload.get("error"):
            collection_errors.append({"role": role, "error": str(payload["error"])})
            continue
        if int(payload.get("returncode") or 0) != 0:
            collection_errors.append(
                {
                    "role": role,
                    "error": f"docker logs returned {payload.get('returncode')}",
                }
            )
        combined = "\n".join(
            str(payload.get(stream) or "") for stream in ("stdout", "stderr")
        )
        for line in combined.splitlines():
            for name, pattern in RUNTIME_LOG_ERROR_PATTERNS.items():
                if pattern.search(line) and len(matches[name]) < 20:
                    matches[name].append({"role": role, "line": line})
    nonempty_matches = {name: rows for name, rows in matches.items() if rows}
    return {
        "valid": not collection_errors and not nonempty_matches,
        "collection_errors": collection_errors,
        "error_matches": nonempty_matches,
    }


def audit_runtime_health(
    logs: dict[str, Any],
    cluster_result: dict[str, Any] | None,
    collection: dict[str, Any],
    collection_cluster: dict[str, Any],
    transport_resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    log_audit = audit_runtime_logs(logs)
    raft_info = (cluster_result or {}).get("raft_info") or {}
    consensus = (cluster_result or {}).get("consensus_thread_status") or {}
    message_send_failures = (cluster_result or {}).get("message_send_failures") or {}
    optimizer_status = collection.get("optimizer_status")
    optimizer_ok = optimizer_status is None or optimizer_status == "ok" or (
        isinstance(optimizer_status, dict) and optimizer_status.get("ok") is True
    )
    local_shards = collection_cluster.get("local_shards") or []
    remote_shards = collection_cluster.get("remote_shards") or []
    transfers = collection_cluster.get("shard_transfers") or []
    all_active = bool(remote_shards) and all(
        shard.get("state") == "Active" for shard in local_shards + remote_shards
    )
    checks = {
        "logs_clean": bool(log_audit["valid"]),
        "consensus_working": (
            str(consensus.get("consensus_thread_status") or "") == "working"
        ),
        "pending_operations_zero": int(raft_info.get("pending_operations") or 0) == 0,
        "message_send_failures_empty": not message_send_failures,
        "collection_green": str(collection.get("status") or "").lower() == "green",
        "optimizer_ok": bool(optimizer_ok),
        "update_queue_empty": int((collection.get("update_queue") or {}).get("length") or 0)
        == 0,
        "all_shards_active": all_active,
        "controller_has_no_lower_local_shards": not local_shards,
        "no_shard_transfers": not transfers,
    }
    if transport_resources is not None:
        checks["transport_resources_clean"] = bool(transport_resources.get("valid"))
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "log_audit": log_audit,
        "message_send_failures": message_send_failures,
        "pending_operations": int(raft_info.get("pending_operations") or 0),
        "consensus_thread_status": consensus.get("consensus_thread_status"),
        "optimizer_status": optimizer_status,
        "update_queue_length": int(
            (collection.get("update_queue") or {}).get("length") or 0
        ),
        "shard_transfers": transfers,
        "transport_resources": transport_resources,
    }


def main() -> int:
    runtime_log_since_epoch = int(time.time())
    args = parse_args()
    validate_args(args)
    distance_config = vector_distance_config(args.vector_distance)
    configured_upper_search_ef = effective_upper_search_ef(args)
    if args.recover_routing_from_collection and (
        args.routing_mode != "faithful_original_rest" or not args.reuse_existing
    ):
        raise ValueError(
            "--recover-routing-from-collection requires --routing-mode faithful_original_rest "
            "and --reuse-existing"
        )
    if args.direct_peer_local_premerge and args.search_dispatch_mode != "direct_peer":
        raise ValueError("--direct-peer-local-premerge requires --search-dispatch-mode direct_peer")
    output_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)

    topology: dict[str, Any] | None = None
    cluster_preflight: dict[str, Any] | None = None
    if args.cluster_topology:
        topology = load_cluster_topology(args.cluster_topology)
        cluster_preflight = validate_cluster_preflight(args.base_url, topology)
    deployment_manifest = load_optional_json(args.deployment_manifest)
    transport_resources_start = (
        capture_controller_transport_resources(deployment_manifest)
        if deployment_manifest is not None
        else None
    )
    if getattr(args, "require_clean_runtime", False) and transport_resources_start is not None:
        initial_transport_audit = audit_transport_resources(
            transport_resources_start,
            transport_resources_start,
        )
        if not initial_transport_audit["valid"]:
            raise RuntimeError(
                "distributed controller transport resources are not clean before the run: "
                f"{initial_transport_audit}"
            )

    with h5py.File(args.hdf5_path, "r") as handle:
        train = slice_train_rows(handle["train"], args.train_limit)[:].astype(np.float32, copy=True)
        tuning_queries = handle["test"][: args.tuning_query_count].astype(np.float32, copy=True)
        tuning_neighbors = handle["neighbors"][: args.tuning_query_count, : args.top_k].astype(np.int32, copy=True)
        eval_queries = handle["test"][: args.eval_query_count].astype(np.float32, copy=True)
        eval_neighbors = handle["neighbors"][: args.eval_query_count, : args.top_k].astype(np.int32, copy=True)
        warmup_queries = handle["test"][: args.warmup_query_count].astype(np.float32, copy=True)
        warmup_neighbors = handle["neighbors"][: args.warmup_query_count, : args.top_k].astype(np.int32, copy=True)

    train = prepare_vectors_for_distance(train, distance_config["name"])
    tuning_queries = prepare_vectors_for_distance(tuning_queries, distance_config["name"])
    eval_queries = prepare_vectors_for_distance(eval_queries, distance_config["name"])
    warmup_queries = prepare_vectors_for_distance(warmup_queries, distance_config["name"])

    if topology is not None:
        peers = cluster_peer_ids(args.base_url, exact_uris=topology["worker_uris"])
        if cluster_preflight is None or peers != cluster_preflight["worker_peer_ids"]:
            raise RuntimeError(
                f"worker placement peer mismatch: preflight={cluster_preflight}, selected={peers}"
            )
    else:
        peers = cluster_peer_ids(args.base_url, args.placement_peer_uri_contains)
    shard_placement_map: dict[str, int] | None = None
    if args.shard_placement == "map":
        if not args.shard_placement_map:
            raise ValueError("--shard-placement map requires --shard-placement-map")
        shard_placement_map = load_shard_placement_map(
            args.shard_placement_map,
            args.shard_placement_map_name,
        )
    elif args.shard_placement_map:
        raise ValueError("--shard-placement-map requires --shard-placement map")

    label_to_shard: dict[int, str] | None = None
    point_to_shards: list[list[int]] | None = None
    routing_state: OriginalRoutingState | None = None
    upper_index: Any | None = None
    upper_ids = np.asarray([], dtype=np.int64)
    simple_kmeans_centroids: np.ndarray | None = None
    simple_kmeans_total_assigned: int | None = None
    simple_kmeans_expansion_ratio: float | None = None
    simple_kmeans_shard_counts: np.ndarray | None = None
    recovered_original_total_assigned: int | None = None
    recovered_original_expansion_ratio: float | None = None
    use_payload_source_id = False
    routing_build_metadata: dict[str, Any] | None = None

    if args.routing_mode == "cpp_kmeans_baseline":
        assignments, _centroids = build_cpp_kmeans_baseline_assignments(
            train,
            args.num_shards,
            args.cpp_kmeans_train_size,
            args.kmeans_iters,
            args.kmeans_rand_seed,
        )
        routing_build_metadata = build_routing_build_metadata(
            args,
            train_count=len(train),
            effective_num_shards=args.num_shards,
            vector_distance=distance_config["qdrant_distance"],
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
            shard_placement_map,
            vector_distance=distance_config["qdrant_distance"],
            routing_build_metadata=routing_build_metadata,
        )

        upper_vectors, upper_ids, label_to_shard, sample_rows = sample_cpp_kmeans_upper_points(
            train,
            assignments,
            args.num_shards,
            args.sample_denominator,
            args.upper_sample_seed,
        )
        if len(upper_ids) == 0:
            raise ValueError(
                "cpp_kmeans_baseline sampled no upper points; decrease --sample-denominator "
                "or increase the training set size"
            )
        upper_index = build_upper_index(
            upper_vectors,
            upper_ids,
            train.shape[1],
            args.upper_m,
            args.upper_ef_construction,
            configured_upper_search_ef,
            distance_config["hnsw_space"],
        )
        effective_num_shards = args.num_shards
    elif args.routing_mode == "kmeans_simple_nprobe":
        assignments, simple_kmeans_centroids = build_cpp_kmeans_baseline_assignments(
            train,
            args.num_shards,
            args.cpp_kmeans_train_size,
            args.kmeans_iters,
            args.kmeans_rand_seed,
        )
        routing_build_metadata = build_routing_build_metadata(
            args,
            train_count=len(train),
            effective_num_shards=args.num_shards,
            vector_distance=distance_config["qdrant_distance"],
        )
        simple_kmeans_multi_assign = float(args.simple_kmeans_multi_assign_alpha) > 1.0
        if simple_kmeans_multi_assign:
            point_to_shards = simple_kmeans_point_to_shards_by_distance_alpha(
                train,
                simple_kmeans_centroids,
                args.simple_kmeans_multi_assign_alpha,
                args.simple_kmeans_multi_assign_chunk_size,
            )
            empty_upper_indices = np.asarray([], dtype=np.int64)
            build_row = ensure_collection_from_point_shards(
                args.base_url,
                args.collection,
                train,
                point_to_shards,
                empty_upper_indices,
                args.num_shards,
                args.hnsw_m,
                args.ef_construct,
                args.upload_batch_size,
                args.reuse_existing,
                args.shard_placement,
                peers,
                shard_placement_map,
                vector_distance=distance_config["qdrant_distance"],
                routing_build_metadata=routing_build_metadata,
                upload_concurrency=args.upload_concurrency,
                upload_queue_capacity_batches=args.upload_queue_capacity_batches,
            )
            shard_counts = shard_counts_from_point_to_shards(point_to_shards, args.num_shards)
            use_payload_source_id = True
        else:
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
                shard_placement_map,
                vector_distance=distance_config["qdrant_distance"],
                routing_build_metadata=routing_build_metadata,
            )
            shard_counts = np.bincount(assignments, minlength=args.num_shards)
        simple_kmeans_shard_counts = shard_counts
        simple_kmeans_total_assigned = int(np.sum(shard_counts))
        simple_kmeans_expansion_ratio = float(simple_kmeans_total_assigned / len(train))
        sample_rows = [
            {
                "shard_id": shard_id,
                "shard_key": shard_key_for_id(shard_id),
                "points_count": int(shard_counts[shard_id]),
                "sample_count": 0,
            }
            for shard_id in range(args.num_shards)
        ]
        effective_num_shards = args.num_shards
    elif args.routing_mode == "legacy_centroid":
        assignments, _shard_centroids = build_assignments_and_centroids(
            train,
            args.num_shards,
            args.sample_size,
            args.kmeans_iters,
            args.seed,
        )
        routing_build_metadata = build_routing_build_metadata(
            args,
            train_count=len(train),
            effective_num_shards=args.num_shards,
            vector_distance=distance_config["qdrant_distance"],
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
            shard_placement_map,
            vector_distance=distance_config["qdrant_distance"],
            routing_build_metadata=routing_build_metadata,
            upload_concurrency=args.upload_concurrency,
            upload_queue_capacity_batches=args.upload_queue_capacity_batches,
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
            configured_upper_search_ef,
            distance_config["hnsw_space"],
        )
        effective_num_shards = args.num_shards
    elif args.routing_mode == "naive_hash_all_shards":
        point_to_shards = hash_point_to_shards(len(train), args.num_shards)
        effective_num_shards = args.num_shards
        routing_build_metadata = build_routing_build_metadata(
            args,
            train_count=len(train),
            effective_num_shards=effective_num_shards,
            vector_distance=distance_config["qdrant_distance"],
        )
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
            shard_placement_map,
            vector_distance=distance_config["qdrant_distance"],
            routing_build_metadata=routing_build_metadata,
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
            configured_upper_search_ef,
            distance_config["hnsw_space"],
        )
        if args.recover_routing_from_collection:
            collection_cluster = collection_cluster_info(args.base_url, args.collection)
            cluster_summary = collection_cluster_summary(collection_cluster)
            effective_num_shards = int(cluster_summary["cluster_shard_count"] or args.num_shards)
            routing_build_metadata = build_routing_build_metadata(
                args,
                train_count=len(train),
                effective_num_shards=effective_num_shards,
                vector_distance=distance_config["qdrant_distance"],
            )
            info = collection_info(args.base_url, args.collection)
            routing_metadata_validation = collection_routing_build_metadata_validation(
                info,
                routing_build_metadata,
            )
            if routing_metadata_validation["status"] == "mismatch":
                raise RuntimeError(
                    f"refusing to reuse collection {args.collection!r}: "
                    + "; ".join(
                        f"routing build metadata {item}"
                        for item in routing_metadata_validation["mismatches"]
                    )
                )
            point_to_shards = recover_upper_point_to_shards_from_collection(
                args.base_url,
                args.collection,
                upper_indices,
                len(train),
                effective_num_shards,
            )
            recovered_original_total_assigned = int(info.get("points_count") or 0)
            recovered_original_expansion_ratio = expansion_ratio_from_assigned_points(
                recovered_original_total_assigned,
                len(train),
            )
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
                **routing_build_metadata_validation_fields(routing_metadata_validation),
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
            upper_index.set_ef(configured_upper_search_ef)
            if args.claim_a_partition_family != "none":
                routing_state = build_claim_a_partition_routing_state(
                    args.claim_a_partition_family,
                    train,
                    upper_indices,
                    point_to_l1s,
                    args.num_shards,
                    args.kmeans_iters,
                    args.kmeans_rand_seed,
                    args.topology_iters,
                    use_multi_assign=not args.disable_multi_assign,
                    multi_assign_min_max_vote=args.orion_multi_assign_min_max_vote,
                    multi_assign_vote_delta=args.orion_multi_assign_vote_delta,
                    multi_assign_max_shards=args.orion_multi_assign_max_shards,
                    random_seed=args.claim_a_random_seed,
                )
            else:
                routing_state = build_canonical_routing_state(
                    train,
                    upper_indices,
                    point_to_l1s,
                    args.num_shards,
                    args.kmeans_iters,
                    args.kmeans_rand_seed,
                    args.topology_iters,
                )
            point_to_shards = routing_state.point_to_shards
            label_to_shard = {
                int(point_id): shard_key_for_id(
                    int(routing_state.l1_to_shard[int(point_id)])
                )
                for point_id in routing_state.upper_indices.tolist()
            }
            effective_num_shards = routing_state.num_shards
            routing_build_metadata = build_routing_build_metadata(
                args,
                train_count=len(train),
                effective_num_shards=effective_num_shards,
                vector_distance=distance_config["qdrant_distance"],
            )
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
                shard_placement_map,
                vector_distance=distance_config["qdrant_distance"],
                routing_build_metadata=routing_build_metadata,
                force_hnsw_for_nonempty_shards=True,
                upload_concurrency=args.upload_concurrency,
                upload_queue_capacity_batches=args.upload_queue_capacity_batches,
            )
            sample_rows = original_upper_sample_rows(routing_state)
        upper_ids = upper_indices
        use_payload_source_id = True

    if topology is not None:
        validate_existing_collection(
            args.base_url,
            args.collection,
            expected_dimension=train.shape[1],
            expected_distance=distance_config["qdrant_distance"],
            expected_hnsw_m=args.hnsw_m,
            expected_ef_construct=args.ef_construct,
            expected_points_count=int(build_row["points_count"]),
            expected_shard_count=effective_num_shards,
            allowed_peer_ids=peers,
            expected_routing_build_metadata=routing_build_metadata,
        )

    source_id_dedup_block_size = args.source_id_dedup_block_size
    if source_id_dedup_block_size is None and use_payload_source_id:
        source_id_dedup_block_size = len(train) + 1
    fixed_ef_shard_chunk_size = int(
        getattr(args, "fixed_ef_shard_chunk_size", 0)
    )

    direct_peer_urls: dict[int, str] | None = None
    shard_key_to_peer: dict[str, int] | None = collection_shard_key_to_peer(
        args.base_url, args.collection
    )
    if args.search_dispatch_mode == "direct_peer":
        direct_peer_urls = parse_peer_http_urls(args.direct_peer_http_urls)
        if not direct_peer_urls:
            raise ValueError("--search-dispatch-mode direct_peer requires --direct-peer-http-urls")
        missing_peer_urls = set(shard_key_to_peer.values()) - set(direct_peer_urls)
        if missing_peer_urls:
            raise ValueError(f"missing direct HTTP URLs for peers: {sorted(missing_peer_urls)}")

    def run_warmup(upper_k: int, base_ef: int, factor: int) -> None:
        if args.warmup_query_count <= 0 or len(warmup_queries) == 0:
            return
        if args.routing_mode == "naive_hash_all_shards":
            if args.search_dispatch_mode == "direct_peer":
                evaluate_all_shards_config_concurrent(
                    args.base_url,
                    args.collection,
                    warmup_queries,
                    warmup_neighbors,
                    args.top_k,
                    effective_num_shards,
                    base_ef,
                    args.batch_size,
                    1,
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    direct_peer_urls=direct_peer_urls,
                    shard_key_to_peer=shard_key_to_peer,
                    lower_execution_order=args.lower_execution_order,
                    direct_peer_local_premerge=args.direct_peer_local_premerge,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                    fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
                )
            else:
                evaluate_all_shards_config(
                    args.base_url,
                    args.collection,
                    warmup_queries,
                    warmup_neighbors,
                    args.top_k,
                    effective_num_shards,
                    base_ef,
                    args.batch_size,
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    lower_execution_order=args.lower_execution_order,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                    fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
                )
            return
        if args.routing_mode == "kmeans_simple_nprobe":
            assert simple_kmeans_centroids is not None
            evaluate_kmeans_simple_nprobe_config(
                args.base_url,
                args.collection,
                warmup_queries,
                warmup_neighbors,
                args.top_k,
                simple_kmeans_centroids,
                upper_k,
                base_ef,
                args.batch_size,
                use_payload_source_id=use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
                direct_peer_urls=direct_peer_urls,
                shard_key_to_peer=shard_key_to_peer,
                lower_execution_order=args.lower_execution_order,
                direct_peer_local_premerge=args.direct_peer_local_premerge,
                score_higher_is_better=distance_config["score_higher_is_better"],
                fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
            )
            return
        assert upper_index is not None
        if args.search_dispatch_mode == "direct_peer":
            evaluate_config_concurrent(
                args.base_url,
                args.collection,
                warmup_queries,
                warmup_neighbors,
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
                lower_execution_order=args.lower_execution_order,
                direct_peer_local_premerge=args.direct_peer_local_premerge,
                score_higher_is_better=distance_config["score_higher_is_better"],
            )
        else:
            routed_evaluator = routed_evaluator_for_planning_mode(args.routed_planning_mode)
            routed_evaluator(
                args.base_url,
                args.collection,
                warmup_queries,
                warmup_neighbors,
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
                lower_execution_order=args.lower_execution_order,
                shard_key_to_peer=shard_key_to_peer,
                score_higher_is_better=distance_config["score_higher_is_better"],
            )

    tuning_rows: list[dict[str, Any]] = []
    if args.routing_mode == "naive_hash_all_shards":
        tuning_parameter_rows = [(0, base_ef, 0) for base_ef in args.base_ef_candidates]
    elif args.routing_mode == "kmeans_simple_nprobe":
        nprobe_candidates = args.nprobe_candidates if args.nprobe_candidates is not None else args.upper_k_candidates
        tuning_parameter_rows = [
            (nprobe, base_ef, 0)
            for nprobe in nprobe_candidates
            for base_ef in args.base_ef_candidates
        ]
    else:
        tuning_parameter_rows = [
            (upper_k, base_ef, factor)
            for upper_k in args.upper_k_candidates
            for base_ef in args.base_ef_candidates
            for factor in args.factor_candidates
        ]

    for upper_k, base_ef, factor in tuning_parameter_rows:
        run_warmup(int(upper_k), int(base_ef), int(factor))
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
                    lower_execution_order=args.lower_execution_order,
                    direct_peer_local_premerge=args.direct_peer_local_premerge,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                    fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
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
                    lower_execution_order=args.lower_execution_order,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                    fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
                )
        elif args.routing_mode == "kmeans_simple_nprobe":
            assert simple_kmeans_centroids is not None
            result = evaluate_kmeans_simple_nprobe_config(
                args.base_url,
                args.collection,
                tuning_queries,
                tuning_neighbors,
                args.top_k,
                simple_kmeans_centroids,
                upper_k,
                base_ef,
                args.batch_size,
                use_payload_source_id=use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
                direct_peer_urls=direct_peer_urls,
                shard_key_to_peer=shard_key_to_peer,
                lower_execution_order=args.lower_execution_order,
                direct_peer_local_premerge=args.direct_peer_local_premerge,
                score_higher_is_better=distance_config["score_higher_is_better"],
                fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
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
                    lower_execution_order=args.lower_execution_order,
                    direct_peer_local_premerge=args.direct_peer_local_premerge,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                )
            else:
                routed_evaluator = routed_evaluator_for_planning_mode(args.routed_planning_mode)
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
                    lower_execution_order=args.lower_execution_order,
                    shard_key_to_peer=shard_key_to_peer,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                )
        row = {
            "upper_k": upper_k,
            "nprobe": upper_k if args.routing_mode == "kmeans_simple_nprobe" else None,
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
            "estimated_ef_sum_per_query": (
                result["avg_assigned_ef_per_visited_shard"] * result["avg_visited_shards"]
            ),
            "avg_physical_peers_per_query": (
                result.get("avg_physical_peers_per_query")
                if result.get("avg_physical_peers_per_query") is not None
                else (
                    float(len(set((shard_key_to_peer or {}).values())))
                    if args.routing_mode == "naive_hash_all_shards"
                    else None
                )
            ),
            "latency_p50_ms": result.get("latency_p50_ms"),
            "latency_p95_ms": result.get("latency_p95_ms"),
            "latency_p99_ms": result.get("latency_p99_ms"),
            "search_batch_calls": result["search_batch_calls"],
            "avg_search_requests_per_query": result["avg_search_requests_per_query"],
            "avg_candidate_groups_per_query": result.get("avg_candidate_groups_per_query"),
            "avg_returned_candidates_per_query": result.get("avg_returned_candidates_per_query"),
        }
        tuning_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    best = choose_best_matched_recall(tuning_rows, args.target_recall)
    run_warmup(int(best["upper_k"]), int(best["base_ef"]), int(best["factor"]))

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
                lower_execution_order=args.lower_execution_order,
                direct_peer_local_premerge=args.direct_peer_local_premerge,
                score_higher_is_better=distance_config["score_higher_is_better"],
                fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
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
                lower_execution_order=args.lower_execution_order,
                include_per_query_metrics=args.write_per_query_metrics,
                score_higher_is_better=distance_config["score_higher_is_better"],
                fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
            )
    elif args.routing_mode == "kmeans_simple_nprobe":
        assert simple_kmeans_centroids is not None
        final_result = evaluate_kmeans_simple_nprobe_config(
            args.base_url,
            args.collection,
            eval_queries,
            eval_neighbors,
            args.top_k,
            simple_kmeans_centroids,
            int(best["upper_k"]),
            int(best["base_ef"]),
            args.batch_size,
            use_payload_source_id=use_payload_source_id,
            source_id_dedup_block_size=source_id_dedup_block_size,
            direct_peer_urls=direct_peer_urls,
            shard_key_to_peer=shard_key_to_peer,
            lower_execution_order=args.lower_execution_order,
            direct_peer_local_premerge=args.direct_peer_local_premerge,
            include_per_query_metrics=args.write_per_query_metrics,
            score_higher_is_better=distance_config["score_higher_is_better"],
            fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
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
                lower_execution_order=args.lower_execution_order,
                direct_peer_local_premerge=args.direct_peer_local_premerge,
                score_higher_is_better=distance_config["score_higher_is_better"],
            )
        else:
            routed_evaluator = routed_evaluator_for_planning_mode(args.routed_planning_mode)
            final_extra_kwargs = {}
            if args.write_per_query_metrics and args.routed_planning_mode in {
                "materialized",
                "compact_materialized",
                "pipelined",
            }:
                final_extra_kwargs["include_per_query_metrics"] = True
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
                lower_execution_order=args.lower_execution_order,
                shard_key_to_peer=shard_key_to_peer,
                score_higher_is_better=distance_config["score_higher_is_better"],
                **final_extra_kwargs,
            )
    final_row = {
        "upper_k": int(best["upper_k"]),
        "nprobe": int(best["upper_k"]) if args.routing_mode == "kmeans_simple_nprobe" else None,
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
        "estimated_ef_sum_per_query": (
            final_result["avg_assigned_ef_per_visited_shard"]
            * final_result["avg_visited_shards"]
        ),
        "avg_physical_peers_per_query": (
            final_result.get("avg_physical_peers_per_query")
            if final_result.get("avg_physical_peers_per_query") is not None
            else (
                float(len(set((shard_key_to_peer or {}).values())))
                if args.routing_mode == "naive_hash_all_shards"
                else None
            )
        ),
        "latency_p50_ms": final_result.get("latency_p50_ms"),
        "latency_p95_ms": final_result.get("latency_p95_ms"),
        "latency_p99_ms": final_result.get("latency_p99_ms"),
        "search_batch_calls": final_result["search_batch_calls"],
        "avg_search_requests_per_query": final_result["avg_search_requests_per_query"],
        "avg_candidate_groups_per_query": final_result.get("avg_candidate_groups_per_query"),
        "avg_returned_candidates_per_query": final_result.get("avg_returned_candidates_per_query"),
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
                    lower_execution_order=args.lower_execution_order,
                    direct_peer_local_premerge=args.direct_peer_local_premerge,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                    fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
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
                    lower_execution_order=args.lower_execution_order,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                    fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
                )
        elif args.routing_mode == "kmeans_simple_nprobe":
            assert simple_kmeans_centroids is not None
            result = evaluate_kmeans_simple_nprobe_config(
                args.base_url,
                args.collection,
                eval_queries,
                eval_neighbors,
                args.top_k,
                simple_kmeans_centroids,
                int(best["upper_k"]),
                int(best["base_ef"]),
                args.batch_size,
                use_payload_source_id=use_payload_source_id,
                source_id_dedup_block_size=source_id_dedup_block_size,
                direct_peer_urls=direct_peer_urls,
                shard_key_to_peer=shard_key_to_peer,
                lower_execution_order=args.lower_execution_order,
                direct_peer_local_premerge=args.direct_peer_local_premerge,
                score_higher_is_better=distance_config["score_higher_is_better"],
                fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
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
                    lower_execution_order=args.lower_execution_order,
                    direct_peer_local_premerge=args.direct_peer_local_premerge,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                )
            else:
                routed_evaluator = routed_evaluator_for_planning_mode(args.routed_planning_mode)
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
                    lower_execution_order=args.lower_execution_order,
                    shard_key_to_peer=shard_key_to_peer,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                )
        row = {
            "run": run_idx,
            "upper_k": int(best["upper_k"]),
            "nprobe": int(best["upper_k"]) if args.routing_mode == "kmeans_simple_nprobe" else None,
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
            "estimated_ef_sum_per_query": (
                result["avg_assigned_ef_per_visited_shard"] * result["avg_visited_shards"]
            ),
            "avg_physical_peers_per_query": (
                result.get("avg_physical_peers_per_query")
                if result.get("avg_physical_peers_per_query") is not None
                else (
                    float(len(set((shard_key_to_peer or {}).values())))
                    if args.routing_mode == "naive_hash_all_shards"
                    else None
                )
            ),
            "latency_p50_ms": result.get("latency_p50_ms"),
            "latency_p95_ms": result.get("latency_p95_ms"),
            "latency_p99_ms": result.get("latency_p99_ms"),
            "search_batch_calls": result["search_batch_calls"],
            "avg_search_requests_per_query": result["avg_search_requests_per_query"],
            "avg_candidate_groups_per_query": result.get("avg_candidate_groups_per_query"),
            "avg_returned_candidates_per_query": result.get("avg_returned_candidates_per_query"),
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
                    fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
                )
            elif args.routing_mode == "kmeans_simple_nprobe":
                assert simple_kmeans_centroids is not None
                query_plans = build_kmeans_simple_nprobe_search_plans(
                    eval_queries,
                    simple_kmeans_centroids,
                    int(best["upper_k"]),
                    args.top_k,
                    int(best["base_ef"]),
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
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
                    lower_execution_order=args.lower_execution_order,
                    direct_peer_local_premerge=args.direct_peer_local_premerge,
                    score_higher_is_better=distance_config["score_higher_is_better"],
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
                    lower_execution_order=args.lower_execution_order,
                    direct_peer_local_premerge=args.direct_peer_local_premerge,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                    fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
                )
            elif args.routing_mode == "kmeans_simple_nprobe":
                assert simple_kmeans_centroids is not None
                result = evaluate_kmeans_simple_nprobe_config_concurrent(
                    args.base_url,
                    args.collection,
                    eval_queries,
                    eval_neighbors,
                    args.top_k,
                    simple_kmeans_centroids,
                    int(best["upper_k"]),
                    int(best["base_ef"]),
                    args.batch_size,
                    int(concurrency),
                    use_payload_source_id=use_payload_source_id,
                    source_id_dedup_block_size=source_id_dedup_block_size,
                    direct_peer_urls=direct_peer_urls,
                    shard_key_to_peer=shard_key_to_peer,
                    lower_execution_order=args.lower_execution_order,
                    direct_peer_local_premerge=args.direct_peer_local_premerge,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                    fixed_ef_shard_chunk_size=fixed_ef_shard_chunk_size,
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
                    lower_execution_order=args.lower_execution_order,
                    direct_peer_local_premerge=args.direct_peer_local_premerge,
                    score_higher_is_better=distance_config["score_higher_is_better"],
                )
            row = {
                "concurrency": int(concurrency),
                "concurrency_evaluation_mode": args.concurrency_evaluation_mode,
                "search_dispatch_mode": args.search_dispatch_mode,
                "lower_execution_order": args.lower_execution_order,
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
                "avg_candidate_groups_per_query": result.get("avg_candidate_groups_per_query"),
                "avg_returned_candidates_per_query": result.get("avg_returned_candidates_per_query"),
                "routing_plan_wall_s_excluded_from_qps": concurrency_plan_wall_s,
            }
            concurrency_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    needs_architecture_query_plans = args.placement_simulation or args.physical_execution_trace
    placement_simulation: dict[str, Any] | None = None
    physical_execution_trace: dict[str, Any] | None = None
    architecture_query_plans: list[dict[str, Any]] | None = None
    if needs_architecture_query_plans:
        if args.routing_mode != "faithful_original_rest":
            raise ValueError("method4 architecture traces are only supported for faithful_original_rest")
        if args.routed_planning_mode not in {"materialized", "pipelined"}:
            raise ValueError(
                "method4 architecture traces require --routed-planning-mode "
                "materialized or pipelined"
            )
        if point_to_shards is None:
            raise ValueError("method4 architecture traces require point_to_shards routing metadata")
        assert upper_index is not None
        upper_labels = compute_upper_labels(upper_index, eval_queries, int(best["upper_k"]))
        architecture_query_plans = build_routed_search_plans(
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

    if args.placement_simulation:
        if args.placement_simulation_peer_count <= 0:
            raise ValueError("--placement-simulation-peer-count must be positive")
        assert architecture_query_plans is not None
        placement_simulation = placement_simulation_summary(
            architecture_query_plans,
            args.placement_simulation_peer_count,
        )

    if args.physical_execution_trace:
        # Placement is available for both a collection created in this invocation and
        # one accepted through --reuse-existing.  Always read the live cluster view;
        # limiting this to reuse mode incorrectly rejected newly deployed collections.
        actual_shard_key_to_peer = collection_shard_key_to_peer(args.base_url, args.collection)
        if not actual_shard_key_to_peer:
            raise ValueError("--physical-execution-trace requires a deployed clustered collection")
        assert architecture_query_plans is not None
        physical_execution_trace = physical_execution_summary(
            architecture_query_plans,
            actual_shard_key_to_peer,
            args.top_k,
        )

    write_csv(output_dir / "builds.csv", [build_row])
    write_csv(output_dir / "upper_sample_stats.csv", sample_rows)
    write_csv(output_dir / "routing_tuning.csv", tuning_rows)
    write_csv(output_dir / "final_metrics.csv", [final_row])
    write_csv(output_dir / "final_per_query_metrics.csv", final_result.get("per_query_rows", []))
    write_csv(output_dir / "stability_runs.csv", stability_rows)
    write_csv(output_dir / "concurrency_runs.csv", concurrency_rows)
    if placement_simulation is not None:
        (output_dir / "placement_simulation.json").write_text(
            json.dumps(placement_simulation, indent=2),
            encoding="utf-8",
        )
    if physical_execution_trace is not None:
        (output_dir / "physical_execution_trace.json").write_text(
            json.dumps(physical_execution_trace, indent=2),
            encoding="utf-8",
        )

    transport_resources_end = (
        capture_controller_transport_resources(deployment_manifest)
        if transport_resources_start is not None
        else None
    )
    transport_resource_audit = (
        audit_transport_resources(transport_resources_start, transport_resources_end)
        if transport_resources_start is not None and transport_resources_end is not None
        else None
    )
    collection_info_now = collection_info(args.base_url, args.collection)
    collection_cluster_snapshot = collection_cluster_info(args.base_url, args.collection)
    cluster_summary_now = collection_cluster_summary(collection_cluster_snapshot)
    cluster_snapshot_now = None
    try:
        cluster_snapshot_now = request_json(args.base_url, "GET", "/cluster").get("result")
    except RuntimeError:
        pass
    runtime_logs = collect_container_log_tails(
        deployment_manifest,
        since_epoch=runtime_log_since_epoch,
    )
    runtime_health = audit_runtime_health(
        runtime_logs,
        cluster_snapshot_now,
        collection_info_now,
        collection_cluster_snapshot or {},
        transport_resource_audit,
    )
    actual_dataset_sha256 = args.dataset_sha256 or sha256_path(args.hdf5_path)
    if args.dataset_sha256 and actual_dataset_sha256 != sha256_path(args.hdf5_path):
        raise RuntimeError("--dataset-sha256 does not match the HDF5 file")
    deployment_image = (deployment_manifest or {}).get("image") or {}
    resolved_image_tag = args.image_tag or deployment_image.get("tag")
    resolved_image_digest = args.image_digest or deployment_image.get("id") or deployment_image.get("digest")
    placement_summary = worker_shard_point_summary(sample_rows, shard_key_to_peer or {})
    routing_build_metadata_status = build_row.get(
        "routing_build_metadata_status",
        "not_requested",
    )
    routing_build_metadata_note = {
        "verified": (
            "The live collection contains Orion harness routing-build metadata whose "
            "canonical fields and SHA-256 fingerprint exactly match this run."
        ),
        "missing_unverified": (
            "The collection predates the routing-build metadata guard. Schema, point-count, "
            "and placement checks passed, but seed/partition semantics are unverified."
        ),
        "not_requested": "No routing-build metadata expectation was supplied.",
    }.get(str(routing_build_metadata_status), "Routing-build metadata validation failed.")

    summary = {
        "base_url": args.base_url,
        "collection": args.collection,
        "collection_routing_build_metadata": {
            "status": routing_build_metadata_status,
            "verified": bool(build_row.get("routing_build_metadata_verified", False)),
            "expected_fingerprint": build_row.get(
                "routing_build_metadata_expected_fingerprint"
            ),
            "actual_fingerprint": build_row.get(
                "routing_build_metadata_actual_fingerprint"
            ),
            "expected_fields": routing_build_metadata,
            "note": routing_build_metadata_note,
        },
        "hdf5_path": args.hdf5_path,
        "dataset_sha256": actual_dataset_sha256,
        "dataset_size_bytes": Path(args.hdf5_path).stat().st_size,
        "repository": repository_provenance(),
        "image_tag": resolved_image_tag,
        "image_digest": resolved_image_digest,
        "deployment_manifest_path": args.deployment_manifest,
        "deployment_manifest": deployment_manifest,
        "runtime_log_since_epoch": runtime_log_since_epoch,
        "container_log_tails": runtime_logs,
        "runtime_health_audit": runtime_health,
        "command": [sys.executable, *sys.argv],
        "process_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cluster_topology_path": args.cluster_topology,
        "cluster_preflight": cluster_preflight,
        "cluster_snapshot": cluster_snapshot_now,
        "vector_distance": distance_config["name"],
        "upper_hnsw_space": distance_config["hnsw_space"],
        "qdrant_vector_distance": distance_config["qdrant_distance"],
        "normalize_vectors": distance_config["normalize_vectors"],
        "score_higher_is_better": distance_config["score_higher_is_better"],
        "train_limit": args.train_limit,
        "routing_mode": args.routing_mode,
        "initial_num_shards": args.num_shards,
        "num_shards": effective_num_shards,
        "shard_placement": args.shard_placement,
        "shard_placement_map": args.shard_placement_map if args.shard_placement == "map" else None,
        "shard_placement_map_name": args.shard_placement_map_name if args.shard_placement == "map" else None,
        "placement_peer_uri_contains": args.placement_peer_uri_contains,
        "discovered_peer_ids": peers,
        "collection_cluster": cluster_summary_now,
        "collection_cluster_snapshot": collection_cluster_snapshot,
        "worker_shard_points": placement_summary,
        "placement_valid": cluster_summary_now.get("cluster_placement_valid"),
        "warmup_query_count": args.warmup_query_count,
        "sample_denominator": args.sample_denominator,
        "k_overlap": args.k_overlap,
        "search_all_shards": args.search_all_shards,
        "concurrency_evaluation_mode": args.concurrency_evaluation_mode,
        "search_dispatch_mode": args.search_dispatch_mode,
        "lower_execution_order": args.lower_execution_order,
        "fixed_ef_shard_chunk_size": fixed_ef_shard_chunk_size,
        "direct_peer_http_urls": args.direct_peer_http_urls if args.search_dispatch_mode == "direct_peer" else None,
        "direct_peer_local_premerge": args.direct_peer_local_premerge,
        "routed_execution_mode": args.routed_execution_mode,
        "routed_planning_mode": args.routed_planning_mode,
        "pipelined_execution": (
            {
                "enabled": True,
                "lookahead_batches": 1,
                "preencoded_search_batch_body": args.search_dispatch_mode == "coordinator",
                "note": (
                    "Planning and JSON encoding for batch N+1 run on the benchmark-client "
                    "CPU set while the controller and workers execute lower search for batch N. "
                    "Only one lower HTTP batch remains in flight, and routed shards, MultiEP, "
                    "Dynamic EF, source-ID dedup, and final global top-k are unchanged."
                ),
            }
            if args.routed_planning_mode == "pipelined"
            else None
        ),
        "placement_simulation": placement_simulation,
        "physical_execution_trace": physical_execution_trace,
        "compact_ef_mode": args.compact_ef_mode if args.routed_execution_mode == "compact_query_ef" else None,
        "routed_result_limit_mode": args.routed_result_limit_mode,
        "routed_result_limit_multiplier": (
            args.routed_result_limit_multiplier if args.routed_result_limit_mode == "fixed_multiplier" else None
        ),
        "distributed_fixed_ef_peer_compaction": (
            {
                "enabled": True,
                "uniform_ef": True,
                "note": (
                    "The harness repeats the baseline's one fixed EF in the per-shard "
                    "request map so Qdrant can group remote work by physical peer. This "
                    "changes transport fan-out only; nprobe/all-shards selection and the "
                    "fixed lower HNSW EF are unchanged."
                ),
            }
            if args.routing_mode in {"kmeans_simple_nprobe", "naive_hash_all_shards"}
            else None
        ),
        "source_id_dedup_block_size": source_id_dedup_block_size,
        "multi_assign": args.routing_mode == "faithful_original_rest",
        "orion_multi_assign_min_max_vote": (
            2 if args.routing_mode == "faithful_original_rest" else None
        ),
        "orion_multi_assign_vote_delta": (
            0 if args.routing_mode == "faithful_original_rest" else None
        ),
        "orion_multi_assign_max_shards": (
            0 if args.routing_mode == "faithful_original_rest" else None
        ),
        "fission_enabled": False,
        "orion_balance": (
            {"mode": "none", "intervention": False}
            if args.routing_mode == "faithful_original_rest"
            else None
        ),
        "fair_architecture_note": (
            "faithful_original_rest, kmeans_simple_nprobe, and naive_hash_all_shards all use Qdrant "
            "custom shard keys, the same Docker cluster shape, the same batch search endpoint, and "
            "the same native physical-peer grouping path, and the same client result merge path. "
            "Simple and Naive express their one fixed EF as a uniform per-shard transport map; "
            "this does not make their EF dynamic. The intended algorithmic difference is shard "
            "selection: Orion routes to upper-selected shards with Method4 routing metadata, "
            "kmeans_simple_nprobe probes the nearest centroid shards by nprobe, and "
            "naive_hash_all_shards searches every shard."
            if args.routing_mode in {"faithful_original_rest", "kmeans_simple_nprobe", "naive_hash_all_shards"}
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
            "(upper sample seed 100 and KMeans random initialization stream), but "
            "Python/NumPy RNG streams are not bit-identical "
            "to libstdc++ std::mt19937/std::shuffle or C rand()."
            if args.routing_mode == "faithful_original_rest"
            else None
        ),
        "upper_hnsw": (
            {
                "m": args.upper_m,
                "ef_construction": args.upper_ef_construction,
                "search_ef": configured_upper_search_ef,
                "partition_build_efs": (
                    args.k_overlap
                    if args.routing_mode == "faithful_original_rest"
                    else None
                ),
                "sample_points": int(len(upper_ids)),
            }
            if upper_index is not None
            else None
        ),
        "kmeans_simple_nprobe": (
            {
                "centroid_count": int(len(simple_kmeans_centroids)),
                "nprobe": int(best["upper_k"]),
                "hnsw_ef": int(best["base_ef"]),
                "cpp_kmeans_train_size": args.cpp_kmeans_train_size,
                "kmeans_iters": args.kmeans_iters,
                "kmeans_rand_seed": args.kmeans_rand_seed,
                "multi_assign": bool(args.simple_kmeans_multi_assign_alpha > 1.0),
                "multi_assign_alpha": float(args.simple_kmeans_multi_assign_alpha),
                "total_assigned": simple_kmeans_total_assigned,
                "expansion_ratio": simple_kmeans_expansion_ratio,
                "shard_count_min": int(np.min(simple_kmeans_shard_counts)) if simple_kmeans_shard_counts is not None else None,
                "shard_count_max": int(np.max(simple_kmeans_shard_counts)) if simple_kmeans_shard_counts is not None else None,
                "dynamic_ef": False,
                "upper_hnsw_entry_points": False,
                "source_id_dedup": bool(use_payload_source_id),
            }
            if args.routing_mode == "kmeans_simple_nprobe" and simple_kmeans_centroids is not None
            else None
        ),
        "original_routing": (
            {
                "topology_iterations": routing_state.topology_iterations,
                "total_assigned": routing_state.total_assigned,
                "expansion_ratio": routing_state.expansion_ratio,
                "multi_assign_min_max_vote": 2,
                "multi_assign_vote_delta": 0,
                "multi_assign_max_shards": 0,
                "assignment_policy": "all_max_vote_ties_else_nearest_when_max_vote_is_one",
                "initial_partition": "kmeans_without_capacity_correction",
                "fission_events": routing_state.fission_events,
                "shard_count_min": int(np.min(routing_state.shard_counts)),
                "shard_count_max": int(np.max(routing_state.shard_counts)),
                "claim_a_partition_family": routing_state.claim_a_partition_family,
                "claim_a_partition_note": routing_state.claim_a_partition_note,
                "balance_diagnostics": routing_state.balance_diagnostics,
                "claim_a_random_seed": (
                    args.claim_a_random_seed
                    if routing_state.claim_a_partition_family == "random_balanced_46"
                    else None
                ),
            }
            if routing_state is not None
            else (
                {
                    "topology_iterations": None,
                    "total_assigned": recovered_original_total_assigned,
                    "expansion_ratio": recovered_original_expansion_ratio,
                    "multi_assign_min_max_vote": 2,
                    "multi_assign_vote_delta": 0,
                    "multi_assign_max_shards": 0,
                    "assignment_policy": "all_max_vote_ties_else_nearest_when_max_vote_is_one",
                    "fission_events": None,
                    "shard_count_min": None,
                    "shard_count_max": None,
                    "recovered_from_collection": True,
                }
                if args.routing_mode == "faithful_original_rest"
                and args.recover_routing_from_collection
                and recovered_original_total_assigned is not None
                else None
            )
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
    if args.require_clean_runtime and not runtime_health["valid"]:
        raise RuntimeError(
            "distributed runtime health audit failed; result was retained for diagnosis: "
            f"{runtime_health}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
