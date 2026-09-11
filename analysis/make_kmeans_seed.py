"""Precompute an alternative k-means initial L1->shard seed for the
"is Orion's custom balanced k-means a good seed?" ablation, in a plain process
WITHOUT LD_PRELOAD (so sklearn imports cleanly). build_orion_layout.py loads it
via --init-file under LD_PRELOAD.

Two standard alternatives to Orion's custom global-sorted-greedy capacity
assignment (`cpp_style_predict_balanced`):

- ``standard_balanced``: MiniBatchKMeans centroids + exact equal-quota
  assignment resolving contention by (second-best - best) distance priority
  (the ELKI-style same-size heuristic; reused from fanout_headroom.balanced_kmeans).
  Hard equal-size *by point count*.
- ``plain``: plain nearest-centroid k-means, no balancing at all.

The upper node set is reproduced with the same deterministic sampler
build_orion_layout uses (global_upper_indices), so the seed aligns bit-for-bit
with the upper_indices computed inside the layout build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True, help="normalized train.npy")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out", required=True, help="npy seed (length N, -1 non-upper)")
    parser.add_argument(
        "--method", choices=("standard_balanced", "plain"), required=True
    )
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    import fanout_headroom as fh

    train = np.load(args.train_path, mmap_mode="r")
    n = int(train.shape[0])
    upper_indices = experiment.global_upper_indices(
        n, int(args.sample_denominator), int(args.upper_sample_seed)
    )
    features = np.asarray(train[upper_indices], dtype=np.float32)
    print(f"upper nodes: {features.shape[0]} x {features.shape[1]}", flush=True)

    if args.method == "standard_balanced":
        labels = fh.balanced_kmeans(features, int(args.shards), int(args.seed))
    else:
        from sklearn.cluster import KMeans

        labels = KMeans(
            n_clusters=int(args.shards), random_state=int(args.seed),
            n_init=1, max_iter=50,
        ).fit_predict(features).astype(np.int32)

    seed = np.full(n, -1, dtype=np.int64)
    seed[upper_indices] = labels.astype(np.int64)
    sizes = np.bincount(labels, minlength=int(args.shards))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, seed)
    print(
        f"wrote {args.out} ({args.method}): {upper_indices.size} upper nodes -> "
        f"{args.shards} shards; size min={sizes.min()} max={sizes.max()} "
        f"skew={sizes.max()/sizes.mean():.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
