"""Build a data-neighborhood multi-assign layout: replicate boundary points onto
their next-nearest shards so the navigation router covers a query's neighbors in
fewer probes (lower fan-out at fixed recall) -- the lever the L1-only replication
lacked, and without its recall penalty.

Home shard of every point is taken from a single-assign base layout (so arm A and
this arm share the same base). A point is "boundary" when its 2nd-nearest shard
centroid is nearly as close as its home; those points, ranked by closeness ratio
d2/d1, receive extra copies (2nd then 3rd nearest) until the copy budget implied
by --expansion is spent. Output is a CSR layout .npz for upload_pts.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from fanout_headroom import Layout, shard_centroids  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-path", required=True)
    p.add_argument("--base-layout", required=True, help="single-assign base .npz")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--expansion", type=float, default=1.4, help="target copies/point")
    p.add_argument("--max-replicas", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    train = np.load(args.train_path).astype(np.float32)
    if args.normalize:
        train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    base = Layout.load(args.base_layout)
    n, shards = base.points, base.shards
    home = base.indices.astype(np.int64)
    centroids = shard_centroids(base, train)

    # distance from every point to every shard centroid; rank shards per point
    d = ((train**2).sum(1)[:, None] - 2.0 * train @ centroids.T
         + (centroids**2).sum(1)[None, :])
    order = np.argsort(d, axis=1)  # nearest-first shard ids per point
    dsort = np.take_along_axis(d, order, axis=1)
    # closeness ratio of 2nd-nearest vs nearest (>=1); ~1 => on a boundary
    ratio = np.sqrt(np.maximum(dsort[:, 1], 0) / np.maximum(dsort[:, 0], 1e-12))

    lists = [[int(home[i])] for i in range(n)]
    budget = int((args.expansion - 1.0) * n)
    # spend budget on the most-boundary points first (smallest ratio)
    boundary_rank = np.argsort(ratio)
    spent = 0
    ri = 2  # start by adding 2nd nearest (rank index 1), escalate to 3rd, ...
    for replica_slot in range(1, args.max_replicas):
        if spent >= budget:
            break
        for pt in boundary_rank:
            if spent >= budget:
                break
            cand = int(order[pt, replica_slot])
            if cand not in lists[pt]:
                lists[pt].append(cand)
                spent += 1
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(x) for x in lists])
    indices = np.fromiter((s for x in lists for s in x), dtype=np.int32,
                          count=int(indptr[-1]))
    np.savez(args.out, indptr=indptr, indices=indices, shard_count=np.int64(shards))
    print(f"multi-assign -> {args.out}: copies={len(indices)} points={n} "
          f"expansion={len(indices)/n:.4f} (budget {budget}, spent {spent})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
