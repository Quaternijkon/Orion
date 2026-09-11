"""Build a balanced-k-means layout whose per-shard *quota* is either point count
(the usual size balance) or a per-point load weight (query-load balance).

Same geometric centroids in both modes, so the two layouts differ only in what
they equalize across shards. Balancing summed load weight should flatten query
load skew (the throughput cap) at some cost in fan-out; measuring which wins is
the point of the probe. Saved in the CSR Layout format used by fanout_headroom.

Run WITHOUT LD_PRELOAD (uses sklearn).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def weighted_balanced_assign(
    train: np.ndarray, centroids: np.ndarray, weights: np.ndarray,
    max_load_ratio: float,
) -> np.ndarray:
    """Nearest-centroid assignment under a per-shard summed-weight capacity.

    Points are ordered by how strongly they prefer their best centroid over the
    second best (stable, decisive points first) and placed on the nearest
    centroid that still has weight capacity; leftovers go to the lightest shard.
    With unit weights this reduces to standard equal-size balancing.
    """
    n = train.shape[0]
    shards = centroids.shape[0]
    distances = np.empty((n, shards), dtype=np.float32)
    for i in range(0, n, 100_000):
        w = train[i : i + 100_000]
        distances[i : i + 100_000] = (
            (w**2).sum(1)[:, None] - 2.0 * w @ centroids.T + (centroids**2).sum(1)[None, :]
        )

    order_by_shard = np.argsort(distances, axis=1)
    best = distances[np.arange(n), order_by_shard[:, 0]]
    second = distances[np.arange(n), order_by_shard[:, 1]]
    priority = np.argsort(second - best)[::-1]

    total_w = float(weights.sum())
    capacity = (total_w / shards) * float(max_load_ratio)
    load = np.zeros(shards, dtype=np.float64)
    assignment = np.full(n, -1, dtype=np.int32)
    for point in priority:
        pw = float(weights[point])
        for shard in order_by_shard[point]:
            if load[shard] + pw <= capacity:
                assignment[point] = shard
                load[shard] += pw
                break
        if assignment[point] < 0:
            shard = int(np.argmin(load))
            assignment[point] = shard
            load[shard] += pw
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--weights-path",
        help="npy of per-point load weights; omit for count (size) balance",
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-load-ratio", type=float, default=1.10,
        help="soft per-shard capacity = ratio * (total_weight / shards)",
    )
    args = parser.parse_args()

    from sklearn.cluster import MiniBatchKMeans

    train = np.load(args.train_path).astype(np.float32)
    if args.normalize:
        norms = np.linalg.norm(train, axis=1, keepdims=True)
        train = train / np.maximum(norms, 1e-12)
    n = train.shape[0]

    if args.weights_path:
        weights = np.load(args.weights_path).astype(np.float64)
        # +1 so zero-in-degree points still occupy a little capacity (avoid a
        # single shard vacuuming up all cold points and blowing up size skew).
        weights = weights + 1.0
        mode = "load"
    else:
        weights = np.ones(n, dtype=np.float64)
        mode = "count"

    t = time.perf_counter()
    centroids = MiniBatchKMeans(
        n_clusters=args.shards, random_state=args.seed, n_init=3,
        batch_size=4096, max_iter=200,
    ).fit(train).cluster_centers_.astype(np.float32)
    assignment = weighted_balanced_assign(train, centroids, weights, args.max_load_ratio)

    counts = np.bincount(assignment, minlength=args.shards).astype(np.float64)
    wsums = np.zeros(args.shards)
    np.add.at(wsums, assignment, weights)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        indptr=np.arange(n + 1, dtype=np.int64),
        indices=assignment.astype(np.int32),
        shard_count=np.int64(args.shards),
    )
    print(
        f"{mode}-balanced [{Path(args.out).name}] in {time.perf_counter()-t:.1f}s: "
        f"size max/mean={counts.max()/counts.mean():.2f}  "
        f"load max/mean={wsums.max()/wsums.mean():.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
