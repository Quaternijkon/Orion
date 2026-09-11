"""Precompute a spectral (normalized-cut) initial L1->shard seed for the init
ablation, in a plain process WITHOUT LD_PRELOAD (so sklearn/pandas import
cleanly). build_orion_layout.py then loads it via --init-file under LD_PRELOAD.

The upper node set is reproduced with the same deterministic sampler
build_orion_layout uses (global_upper_indices), so the seed aligns bit-for-bit
with the upper_indices computed inside the layout build. The seed is a
normalized-cut partition of the upper points' kNN graph: a graph-native warm
start that targets the same cut objective the refinement minimizes, unlike the
geometric k-means seed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True, help="normalized train.npy")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out", required=True, help="npy seed (length N, -1 non-upper)")
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--spectral-knn", type=int, default=15)
    parser.add_argument("--init-seed", type=int, default=1)
    args = parser.parse_args()

    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import SpectralClustering

    train = np.load(args.train_path, mmap_mode="r")
    n = int(train.shape[0])
    upper_indices = experiment.global_upper_indices(
        n, int(args.sample_denominator), int(args.upper_sample_seed)
    )
    features = np.asarray(train[upper_indices], dtype=np.float32)
    print(f"upper nodes: {features.shape[0]} x {features.shape[1]}", flush=True)

    affinity = kneighbors_graph(
        features, n_neighbors=int(args.spectral_knn), mode="connectivity",
        include_self=False, n_jobs=-1,
    )
    affinity = 0.5 * (affinity + affinity.T)
    print("kNN affinity built; running spectral clustering...", flush=True)
    labels = SpectralClustering(
        n_clusters=int(args.shards), affinity="precomputed",
        assign_labels="kmeans", random_state=int(args.init_seed),
    ).fit_predict(affinity)

    seed = np.full(n, -1, dtype=np.int64)
    seed[upper_indices] = labels.astype(np.int64)
    sizes = np.bincount(labels, minlength=int(args.shards))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, seed)
    print(
        f"wrote {args.out}: {upper_indices.size} upper nodes -> {args.shards} shards; "
        f"cluster size min={sizes.min()} max={sizes.max()} "
        f"skew={sizes.max()/sizes.mean():.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
