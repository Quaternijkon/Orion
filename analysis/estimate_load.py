"""Estimate a per-point query-load weight (hubness) offline from the train set.

Throughput is capped by the busiest shard, and a shard is busy when it holds
points that are retrieved by many queries. We cannot see online queries at build
time, so we estimate each point's popularity by its in-degree in the train k-NN
graph: treat every train point as a proxy query, take its top-k neighbors, and
count how often each point is retrieved. High-in-degree points are "hubs" that
attract disproportionate load (pronounced in high-dim angular data like GloVe).

Weights are estimated from TRAIN only, so a layout balanced on them can be
evaluated for load skew on the held-out TEST queries without leakage.

Requires system libstdc++ preloaded for hnswlib:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python estimate_load.py ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True, help="train.npy (row i = point i)")
    parser.add_argument("--out", required=True, help="npy of per-point load weights")
    parser.add_argument("--top-k", type=int, default=10, help="neighbors per proxy query")
    parser.add_argument("--space", choices=("cosine", "l2", "ip"), default="cosine")
    parser.add_argument("--normalize", action="store_true", help="unit-normalize (angular)")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--threads", type=int, default=0, help="0 = all cores")
    args = parser.parse_args()

    import hnswlib

    train = np.load(args.train_path).astype(np.float32)
    n, dim = train.shape
    if args.normalize:
        norms = np.linalg.norm(train, axis=1, keepdims=True)
        train = train / np.maximum(norms, 1e-12)

    index = hnswlib.Index(space=args.space, dim=dim)
    index.init_index(max_elements=n, ef_construction=args.ef_construction, M=args.m)
    if args.threads > 0:
        index.set_num_threads(args.threads)
    built = time.perf_counter()
    index.add_items(train, np.arange(n))
    index.set_ef(max(args.ef_search, args.top_k + 1))
    print(f"index built in {time.perf_counter() - built:.1f}s", flush=True)

    queried = time.perf_counter()
    # top-k+1 because the first hit is the point itself; drop it.
    labels, _ = index.knn_query(train, k=args.top_k + 1)
    neighbors = labels[:, 1:]  # drop self
    in_degree = np.bincount(neighbors.ravel(), minlength=n).astype(np.int64)
    print(f"queried {n} proxy queries in {time.perf_counter() - queried:.1f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, in_degree)
    nz = in_degree[in_degree > 0]
    print(
        f"wrote {args.out}: load weight (in-degree) "
        f"mean={in_degree.mean():.2f} max={in_degree.max()} "
        f"p99={np.percentile(in_degree, 99):.0f} zeros={int((in_degree == 0).sum())} "
        f"skew(max/mean)={in_degree.max()/max(in_degree.mean(),1e-9):.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
