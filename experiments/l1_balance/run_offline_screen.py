#!/usr/bin/env python3
"""Screen frozen-upper-graph L1 partitioners before building distributed indexes.

Candidate partitioners receive only the production upper graph (and, for the
geometry control, upper vectors).  Full-dataset attachments are consumed only
after each owner array has been frozen, to evaluate the unchanged Orion
multi-assignment rule and downstream routing consequences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance.l1_partitioner import (  # noqa: E402
    SUPPORTED_ALGORITHMS,
    PartitionResult,
    partition_l1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--attachments", required=True)
    parser.add_argument("--attachments-manifest", required=True)
    parser.add_argument("--row-count", type=int, required=True)
    parser.add_argument("--attachment-k", type=int, default=10)
    parser.add_argument("--num-partitions", type=int, default=32)
    parser.add_argument("--query-hits")
    parser.add_argument("--query-hits-manifest")
    parser.add_argument("--ground-truth")
    parser.add_argument("--ground-truth-width", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(SUPPORTED_ALGORITHMS),
        choices=SUPPORTED_ALGORITHMS,
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checked_binary_matrix(
    path: Path,
    *,
    rows: int,
    width: int,
    dtype: str,
) -> np.memmap:
    if rows <= 0 or width <= 0:
        raise ValueError("binary matrix rows and width must be positive")
    itemsize = np.dtype(dtype).itemsize
    expected = rows * width * itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path} has {actual} bytes; expected {expected}")
    return np.memmap(path, dtype=dtype, mode="r", shape=(rows, width))


def load_upper(
    artifact_path: Path,
) -> tuple[
    dict[str, Any],
    list[list[int]],
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
    str,
]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    upper_nodes = artifact.get("upper_nodes")
    graph = artifact.get("upper_graph")
    if not isinstance(upper_nodes, list) or not upper_nodes:
        raise ValueError("artifact has no upper_nodes")
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ValueError("artifact has no production upper_graph")
    labels = np.asarray([int(node["label"]) for node in upper_nodes], dtype=np.int64)
    if len(np.unique(labels)) != len(labels):
        raise ValueError("upper labels are not unique")
    label_to_local = {int(label): index for index, label in enumerate(labels.tolist())}
    graph_nodes = graph["nodes"]
    if len(graph_nodes) != len(labels):
        raise ValueError("upper graph node count differs from upper node count")
    adjacency: list[list[int]] = [[] for _ in upper_nodes]
    for graph_node in graph_nodes:
        node = label_to_local.get(int(graph_node["label"]))
        if node is None:
            raise ValueError("upper graph contains an unknown node label")
        levels = graph_node.get("neighbors_by_level")
        if not isinstance(levels, list) or not levels:
            raise ValueError("upper graph node lacks level zero")
        try:
            adjacency[node] = [label_to_local[int(label)] for label in levels[0]]
        except KeyError as exc:
            raise ValueError(f"upper graph contains unknown neighbor {exc.args[0]}") from exc
    vectors = np.asarray([node["vector"] for node in upper_nodes], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(labels):
        raise ValueError("upper vectors are not rectangular")
    entry_label = int(graph["entry_point"])
    if entry_label not in label_to_local:
        raise ValueError("upper graph entry point is unknown")
    entry_point = label_to_local[entry_label]

    undirected = set()
    for left, row in enumerate(adjacency):
        for right in row:
            if left != right:
                undirected.add((min(left, right), max(left, right)))
    edge_left = np.fromiter((edge[0] for edge in sorted(undirected)), dtype=np.int32)
    edge_right = np.fromiter((edge[1] for edge in sorted(undirected)), dtype=np.int32)
    navigator = {
        "vector_schema": artifact["vector_schema"],
        "upper_labels": labels.tolist(),
        "upper_vectors": [node["vector"] for node in upper_nodes],
        "upper_graph": graph,
    }
    return (
        artifact,
        adjacency,
        vectors,
        labels,
        entry_point,
        edge_left,
        edge_right,
        canonical_sha256(navigator),
    )


def local_hits(
    hits: np.ndarray, labels: np.ndarray, row_count: int
) -> np.ndarray:
    max_label = max(int(labels.max()), int(np.max(hits)))
    dense = np.full(max_label + 1, -1, dtype=np.int32)
    dense[labels] = np.arange(len(labels), dtype=np.int32)
    mapped = dense[np.asarray(hits)]
    if np.any(mapped < 0):
        first = int(np.argwhere(mapped < 0)[0, 1])
        raise ValueError(f"hit matrix contains a label outside the upper graph near column {first}")
    if mapped.shape[0] != row_count:
        raise ValueError("hit matrix row count mismatch")
    return mapped


def physical_membership(
    owner: np.ndarray,
    attachment_local: np.ndarray,
    num_partitions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the unchanged delta=0/min-max-vote=2 multi-assignment rule."""

    hit_owner = owner[attachment_local]
    rows = hit_owner.shape[0]
    votes = np.zeros((rows, num_partitions), dtype=np.uint8)
    row_ids = np.arange(rows, dtype=np.int64)
    for column in range(hit_owner.shape[1]):
        np.add.at(votes, (row_ids, hit_owner[:, column]), 1)
    maximum = votes.max(axis=1)
    membership = votes == maximum[:, None]
    fallback = maximum < 2
    membership[fallback] = False
    membership[row_ids[fallback], hit_owner[fallback, 0]] = True
    copy_count = membership.sum(axis=1, dtype=np.int16)
    loads = membership.sum(axis=0, dtype=np.int64)
    return membership, copy_count, loads


def distribution(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    values = np.asarray(values)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_min": int(np.min(values)),
        f"{prefix}_max": int(np.max(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
    }


def topology_metrics(
    owner: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    node_count: int,
    num_partitions: int,
) -> dict[str, float | int]:
    internal = owner[edge_left] == owner[edge_right]
    internal_degree = np.bincount(
        np.concatenate((edge_left[internal], edge_right[internal])),
        minlength=node_count,
    )
    degree = np.bincount(
        np.concatenate((edge_left, edge_right)), minlength=node_count
    )
    retained = np.divide(
        internal_degree,
        degree,
        out=np.ones(node_count, dtype=np.float64),
        where=degree > 0,
    )

    parent = np.arange(node_count, dtype=np.int32)
    size = np.ones(node_count, dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[node]) != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    for raw_left, raw_right in zip(edge_left[internal], edge_right[internal]):
        left = find(int(raw_left))
        right = find(int(raw_right))
        if left == right:
            continue
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]
    largest = np.zeros(num_partitions, dtype=np.int32)
    partition_sizes = np.bincount(owner, minlength=num_partitions)
    for node in range(node_count):
        root = find(node)
        largest[owner[node]] = max(largest[owner[node]], size[root])
    largest_fraction = largest / partition_sizes
    return {
        "upper_edge_count": int(len(edge_left)),
        "upper_edge_cut_ratio": float(1.0 - np.mean(internal)),
        "retained_degree_mean": float(np.mean(retained)),
        "retained_degree_p10": float(np.percentile(retained, 10)),
        "nodes_losing_ge_25pct_neighbors": int(np.count_nonzero(retained <= 0.75)),
        "nodes_losing_ge_50pct_neighbors": int(np.count_nonzero(retained <= 0.50)),
        "nodes_losing_ge_75pct_neighbors": int(np.count_nonzero(retained <= 0.25)),
        "upper_isolated_fraction": float(np.mean(internal_degree == 0)),
        "largest_component_fraction_mean": float(np.mean(largest_fraction)),
        "largest_component_fraction_min": float(np.min(largest_fraction)),
    }


def query_metrics(
    owner: np.ndarray,
    query_local: np.ndarray,
    upper_membership: np.ndarray,
    full_membership: np.ndarray,
    ground_truth: np.ndarray | None,
    dynamic_ef_base: int,
    dynamic_ef_factor: int,
) -> dict[str, float | int]:
    owner_path = owner[query_local]
    owner_unique = np.asarray(
        [len(np.unique(row)) for row in owner_path], dtype=np.int16
    )
    transitions = np.count_nonzero(owner_path[:, 1:] != owner_path[:, :-1], axis=1)
    hit_membership = upper_membership[query_local]
    routed = np.any(hit_membership, axis=1)
    entry_counts_by_shard = hit_membership.sum(axis=1, dtype=np.int16)
    routed_shards = routed.sum(axis=1, dtype=np.int16)
    entry_counts = entry_counts_by_shard.sum(axis=1, dtype=np.int16)
    ef_sum = (
        routed_shards.astype(np.int64) * int(dynamic_ef_base)
        + entry_counts.astype(np.int64) * int(dynamic_ef_factor)
    )
    result: dict[str, float | int] = {
        **distribution(owner_unique, "query_owner_shards"),
        **distribution(transitions, "query_owner_transitions"),
        **distribution(routed_shards, "routed_shards"),
        **distribution(entry_counts, "route_entry_points"),
        **distribution(ef_sum, "route_ef_sum"),
    }
    if ground_truth is not None:
        gt_membership = full_membership[np.asarray(ground_truth)]
        covered = np.any(gt_membership & routed[:, None, :], axis=2)
        per_query = covered.mean(axis=1)
        result.update(
            {
                "gt_routing_coverage_mean": float(np.mean(per_query)),
                "gt_routing_coverage_min": float(np.min(per_query)),
                "gt_queries_full_coverage_fraction": float(np.mean(np.all(covered, axis=1))),
                "gt_queries_zero_coverage": int(np.count_nonzero(~np.any(covered, axis=1))),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.row_count <= 0 or args.attachment_k <= 0 or args.num_partitions <= 0:
        raise ValueError("row-count, attachment-k, and num-partitions must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    artifact_path = Path(args.artifact).expanduser().resolve()
    attachments_path = Path(args.attachments).expanduser().resolve()
    attachments_manifest_path = Path(args.attachments_manifest).expanduser().resolve()
    attachments_manifest = json.loads(attachments_manifest_path.read_text(encoding="utf-8"))
    expected_attachment = {
        "artifact_sha256": sha256_path(artifact_path),
        "row_count": args.row_count,
        "top_k": args.attachment_k,
        "search_ef": 100,
        "hits_sha256": sha256_path(attachments_path),
    }
    mismatches = {
        key: {"expected": value, "actual": attachments_manifest.get(key)}
        for key, value in expected_attachment.items()
        if attachments_manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"attachment manifest mismatch: {mismatches}")

    (
        artifact,
        adjacency,
        vectors,
        labels,
        entry_point,
        edge_left,
        edge_right,
        navigator_sha256,
    ) = load_upper(artifact_path)
    if int(artifact["shard_count"]) != args.num_partitions:
        raise ValueError("artifact shard_count differs from num-partitions")
    attachment_hits = checked_binary_matrix(
        attachments_path,
        rows=args.row_count,
        width=args.attachment_k,
        dtype="<u8",
    )
    attachment_local = local_hits(attachment_hits, labels, args.row_count)

    query_local = None
    query_manifest = None
    ground_truth = None
    if bool(args.query_hits) != bool(args.query_hits_manifest):
        raise ValueError("query-hits and query-hits-manifest must be provided together")
    if args.query_hits:
        query_hits_path = Path(args.query_hits).expanduser().resolve()
        query_manifest_path = Path(args.query_hits_manifest).expanduser().resolve()
        query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
        query_rows = int(query_manifest["row_count"])
        query_k = int(query_manifest["top_k"])
        query_hits = checked_binary_matrix(
            query_hits_path, rows=query_rows, width=query_k, dtype="<u8"
        )
        if query_manifest.get("artifact_sha256") != sha256_path(artifact_path):
            raise ValueError("query-hit artifact checksum mismatch")
        if query_manifest.get("hits_sha256") != sha256_path(query_hits_path):
            raise ValueError("query-hit checksum mismatch")
        query_local = local_hits(query_hits, labels, query_rows)
        if args.ground_truth:
            ground_truth = checked_binary_matrix(
                Path(args.ground_truth).expanduser().resolve(),
                rows=query_rows,
                width=args.ground_truth_width,
                dtype="<u4",
            )

    results: list[tuple[str, np.ndarray, PartitionResult | None, bool]] = []
    source_memberships = [node["shard_membership"] for node in artifact["upper_nodes"]]
    if all(isinstance(row, list) and len(row) == 1 for row in source_memberships):
        source_owner = np.asarray([int(row[0]) for row in source_memberships], dtype=np.int32)
        results.append(("source-l0-informed-owner", source_owner, None, False))
    for method in args.methods:
        partition = partition_l1(
            adjacency,
            args.num_partitions,
            algorithm=method,
            vectors=vectors if method == "balanced-kmeans" else None,
            entry_point=entry_point,
        )
        results.append((method, np.asarray(partition.owner, dtype=np.int32), partition, True))

    output_dir.mkdir(parents=True, exist_ok=False)
    owners_dir = output_dir / "owners"
    owners_dir.mkdir()
    rows: list[dict[str, Any]] = []
    copy_histograms: dict[str, dict[str, int]] = {}
    for method, owner, partition, eligible in results:
        membership, copy_count, loads = physical_membership(
            owner, attachment_local, args.num_partitions
        )
        upper_membership = membership[labels]
        owner_path = owners_dir / f"{method}.owner.i32le"
        owner.astype("<i4", copy=False).tofile(owner_path)
        owner_sha256 = sha256_path(owner_path)
        load_mean = float(np.mean(loads))
        row: dict[str, Any] = {
            "method": method,
            "new_algorithm_eligible": eligible,
            "owner_sha256": owner_sha256,
            "owner_path": str(owner_path),
            "l1_node_count": len(owner),
            "l1_min_size": int(np.min(np.bincount(owner, minlength=args.num_partitions))),
            "l1_max_size": int(np.max(np.bincount(owner, minlength=args.num_partitions))),
            "partition_seconds": float(partition.elapsed_seconds) if partition else None,
            "partition_edge_visits": int(partition.edge_visits) if partition else -1,
            "physical_point_count": int(np.sum(loads)),
            "expansion_ratio": float(np.mean(copy_count)),
            "load_min": int(np.min(loads)),
            "load_max": int(np.max(loads)),
            "load_mean": load_mean,
            "load_cv": float(np.std(loads) / load_mean),
            "load_max_over_mean": float(np.max(loads) / load_mean),
            "load_min_over_mean": float(np.min(loads) / load_mean),
            "empty_shards": int(np.count_nonzero(loads == 0)),
            **topology_metrics(
                owner,
                edge_left,
                edge_right,
                len(owner),
                args.num_partitions,
            ),
        }
        if query_local is not None:
            row.update(
                query_metrics(
                    owner,
                    query_local,
                    upper_membership,
                    membership,
                    ground_truth,
                    int(artifact["dynamic_ef_base"]),
                    int(artifact["dynamic_ef_factor"]),
                )
            )
        rows.append(row)
        unique, counts = np.unique(copy_count, return_counts=True)
        copy_histograms[method] = {
            str(int(value)): int(count)
            for value, count in zip(unique.tolist(), counts.tolist())
        }
        np.savez_compressed(
            owners_dir / f"{method}.evaluation.npz",
            loads=loads,
            upper_membership=np.packbits(upper_membership, axis=1),
            copy_count_hist_values=unique,
            copy_count_hist_counts=counts,
        )
        print(
            f"method={method} cut={row['upper_edge_cut_ratio']:.6f} "
            f"load_max_mean={row['load_max_over_mean']:.6f} "
            f"expansion={row['expansion_ratio']:.6f}",
            flush=True,
        )

    write_csv(output_dir / "summary.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "format_version": 1,
        "contract": {
            "partition_input_scope": "production_upper_graph_only",
            "partitioner_reads_full_attachments": False,
            "partitioner_reads_l0_load": False,
            "partitioner_reads_multi_assignment_state": False,
            "l0_assignment_after_owner_freeze": True,
            "l0_repair_or_flow": False,
            "multi_assignment": {
                "enabled": True,
                "min_max_vote": 2,
                "vote_delta": 0,
                "max_shards": 0,
            },
        },
        "inputs": {
            "artifact": str(artifact_path),
            "artifact_sha256": sha256_path(artifact_path),
            "navigator_sha256": navigator_sha256,
            "attachments": str(attachments_path),
            "attachments_sha256": sha256_path(attachments_path),
            "attachments_manifest": str(attachments_manifest_path),
            "attachments_manifest_sha256": sha256_path(attachments_manifest_path),
            "query_hits_manifest": str(Path(args.query_hits_manifest).resolve())
            if args.query_hits_manifest
            else None,
            "ground_truth": str(Path(args.ground_truth).resolve())
            if args.ground_truth
            else None,
        },
        "parameters": {
            "row_count": args.row_count,
            "attachment_k": args.attachment_k,
            "attachment_search_ef": 100,
            "num_partitions": args.num_partitions,
            "methods": args.methods,
        },
        "copy_count_histograms": copy_histograms,
        "outputs": {
            "summary_csv": "summary.csv",
            "summary_json": "summary.json",
            "owners_dir": "owners",
        },
    }
    (output_dir / "screen-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
