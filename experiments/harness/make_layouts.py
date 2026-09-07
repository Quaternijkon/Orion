"""Produce k-means and hash point->shard layouts (.npz) for the baseline arms.

Both are single-assignment (one shard per point). k-means gives locality (a
router can exploit it); hash is uniform with no locality (broadcast only). Saved
in the CSR Layout format used by fanout_headroom.Layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def save_layout(path: Path, assignment: np.ndarray, shards: int) -> None:
    n = assignment.size
    np.savez(
        path,
        indptr=np.arange(n + 1, dtype=np.int64),
        indices=assignment.astype(np.int32, copy=False),
        shard_count=np.int64(shards),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True, help="normalized train.npy")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kmeans-batch", type=int, default=10000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = np.load(args.train_path, mmap_mode="r")
    n = train.shape[0]

    # Hash: uniform, no locality (matches a default hash-sharded collection's
    # distribution). id % P keeps it deterministic and roughly balanced.
    hash_assignment = (np.arange(n) % args.shards).astype(np.int32)
    save_layout(out_dir / f"hash_p{args.shards}.npz", hash_assignment, args.shards)
    print(f"hash: {n} points -> {args.shards} shards")

    # k-means: MiniBatchKMeans on the (already normalized) vectors.
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=args.shards,
        random_state=args.seed,
        batch_size=args.kmeans_batch,
        n_init=3,
        max_iter=100,
    )
    km.fit(np.asarray(train, dtype=np.float32))
    km_assignment = km.predict(np.asarray(train, dtype=np.float32)).astype(np.int32)
    save_layout(out_dir / f"kmeans_p{args.shards}.npz", km_assignment, args.shards)
    sizes = np.bincount(km_assignment, minlength=args.shards)
    print(
        f"kmeans: {n} points -> {args.shards} shards; "
        f"size min={sizes.min()} max={sizes.max()} skew={sizes.max()/sizes.mean():.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
