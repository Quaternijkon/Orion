"""Check how far the greedy coverage ordering is from exact max-coverage.

The headroom analysis uses a greedy ordering to stand in for the best possible
shard ordering. With disjoint shards greedy is exactly optimal at every prefix,
but under replication maximum coverage is a set-cover style problem where greedy
is only a (1-1/e) approximation. If greedy were loose, the "oracle" columns
would overstate the cost floor of a replicated layout and the comparison against
single-assignment k-means would be unfair.

This enumerates the exact optimum by brute force over the shards that actually
hold a neighbor (a handful per query) and reports the gap.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

from fanout_headroom import (
    Layout,
    greedy_coverage_order,
    neighbor_masks,
    popcount_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--layout-file", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-prefix", type=int, default=4)
    parser.add_argument("--sample", type=int, default=2000)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as handle:
        ground_truth = np.asarray(handle["neighbors"], dtype=np.int64)
    layout = Layout.load(args.layout_file)
    masks = neighbor_masks(layout, ground_truth, args.top_k)

    rng = np.random.default_rng(0)
    sample = (
        rng.choice(masks.shape[0], args.sample, replace=False)
        if masks.shape[0] > args.sample
        else np.arange(masks.shape[0])
    )
    masks = masks[sample]
    table = popcount_table(args.top_k)
    greedy = greedy_coverage_order(masks, args.top_k)

    print(f"{Path(args.layout_file).name}  P={layout.shards}  n={len(sample)}")
    print(f"{'prefix':>6} {'greedy cov':>11} {'exact cov':>10} {'suboptimal':>11}")
    for prefix in range(1, args.max_prefix + 1):
        greedy_cov = np.bitwise_or.reduce(
            masks[np.arange(len(sample))[:, None], greedy[:, :prefix]], axis=1
        )
        greedy_hits = table[greedy_cov].astype(np.int32)

        exact_hits = np.empty(len(sample), dtype=np.int32)
        for row in range(len(sample)):
            candidates = np.nonzero(masks[row])[0]
            if candidates.size <= prefix:
                exact_hits[row] = table[np.bitwise_or.reduce(masks[row, candidates])]
                continue
            row_masks = masks[row, candidates]
            best = 0
            for choice in combinations(range(candidates.size), prefix):
                covered = np.bitwise_or.reduce(row_masks[list(choice)])
                best = max(best, int(table[covered]))
                if best == args.top_k:
                    break
            exact_hits[row] = best

        worse = int((greedy_hits < exact_hits).sum())
        print(
            f"{prefix:6d} {greedy_hits.mean() / args.top_k:11.4f} "
            f"{exact_hits.mean() / args.top_k:10.4f} "
            f"{worse:8d} ({100.0 * worse / len(sample):.2f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
