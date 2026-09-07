"""Query load skew across shards, which is what caps throughput.

`fanout_headroom.py` reports how *many* shards a query must touch. It says
nothing about *which* ones, so it cannot see whether some shards absorb far more
of the workload than others. Throughput is set by the busiest shard, so a layout
with even shard sizes can still scale badly if queries concentrate.

Size skew is a poor proxy for this on a graph index. At fixed `ef` an HNSW search
costs roughly O(ef * M * log n) distance computations, so doubling a shard's size
barely changes its per-query cost; an oversized shard mainly costs memory. What
does hurt is receiving a disproportionate share of the queries.

Reported for each ordering at the fixed budget that reaches the recall target:

    load_skew = max_s (queries touching s) / mean_s (queries touching s)

A skew of 2.0 means the busiest shard sees twice the average, so aggregate
throughput is capped near half of what a perfectly spread workload would allow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from fanout_headroom import (
    Layout,
    build_layout,
    greedy_coverage_order,
    neighbor_masks,
    popcount_table,
    recall_curve,
    shard_centroids,
)


def centroid_order(queries: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Rank shards by query-to-centroid distance, without materializing (n, P, d)."""
    squared = (
        (queries**2).sum(axis=1)[:, None]
        - 2.0 * queries @ centroids.T
        + (centroids**2).sum(axis=1)[None, :]
    )
    return np.argsort(squared, axis=1, kind="stable")


def fixed_budget(cumulative_recall: np.ndarray, target: float) -> int:
    means = cumulative_recall.mean(axis=0)
    reached = np.nonzero(means >= target)[0]
    return int(reached[0]) + 1 if reached.size else cumulative_recall.shape[1]


def skew_at(order: np.ndarray, budget: int, shards: int) -> dict[str, float]:
    touches = np.bincount(order[:, :budget].ravel(), minlength=shards)
    mean = touches.mean()
    return {
        "budget": budget,
        "max_over_mean": float(touches.max() / mean),
        "min_over_mean": float(touches.min() / mean),
        "cv": float(touches.std() / mean),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--layout-file")
    parser.add_argument("--shards", type=int, default=0)
    parser.add_argument(
        "--method",
        default="layout_file",
        choices=("layout_file", "kmeans", "balanced_kmeans"),
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as handle:
        train = np.asarray(handle["train"], dtype=np.float32)
        queries = np.asarray(handle["test"], dtype=np.float32)
        ground_truth = np.asarray(handle["neighbors"], dtype=np.int64)
    if args.normalize:
        for matrix in (train, queries):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.divide(matrix, np.maximum(norms, 1e-12), out=matrix)

    if args.method == "layout_file":
        layout = Layout.load(args.layout_file)
        centroids = shard_centroids(layout, train)
    else:
        assignment, centroids, _ = build_layout(
            train, args.shards, args.method, args.seed
        )
        layout = Layout.from_assignment(assignment, args.shards)

    sizes = layout.copies_per_shard().astype(np.float64)
    masks = neighbor_masks(layout, ground_truth, args.top_k)
    orders = {
        "oracle": greedy_coverage_order(masks, args.top_k),
        "centroid": centroid_order(queries, centroids),
    }

    report: dict[str, object] = {
        "dataset": Path(args.hdf5).name,
        "layout": args.label,
        "shards": layout.shards,
        "recall_target": args.target,
        "size_skew_max_over_mean": float(sizes.max() / sizes.mean()),
        "expansion_ratio": layout.copies / layout.points,
        "orderings": {},
    }
    for name, order in orders.items():
        curve = recall_curve(masks, order, args.top_k)
        budget = fixed_budget(curve, args.target)
        report["orderings"][name] = skew_at(order, budget, layout.shards)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    oracle = report["orderings"]["oracle"]
    centroid = report["orderings"]["centroid"]
    print(
        f"{report['dataset']:24} P={layout.shards:2d} {args.label:16} "
        f"size_skew={report['size_skew_max_over_mean']:.2f} | "
        f"oracle B={oracle['budget']} load_skew={oracle['max_over_mean']:.2f} | "
        f"centroid B={centroid['budget']} load_skew={centroid['max_over_mean']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
