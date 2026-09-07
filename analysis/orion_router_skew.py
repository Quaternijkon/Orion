"""Measure Orion's own graph-navigation router, not a stand-in.

Every other script here orders shards by oracle coverage or by query-to-centroid
distance. Neither is what Orion deploys. Orion routes by navigating the upper
graph: a query's nearest upper-tier (L1) nodes are looked up, and the shards
that own those nodes are the shards it probes. `route_upper_labels_to_shard_eps`
in the harness does exactly this. The per-query fan-out is therefore emergent --
it is however many distinct shards the query's nearest L1 nodes happen to fall
on -- not a prefix length someone picks.

This script reconstructs that router offline and compares three routers on the
*same* Orion layout, all driven to the same recall target:

    oracle    -- greedy coverage, the router-independent lower bound
    centroid  -- rank shards by query-to-centroid distance (a k-means router)
    orion     -- first-appearance order of shards along the query's ranked
                 upper-graph L1 neighbours (Orion's actual routing signal)

For each it reports the fixed shard budget that reaches the target and the query
load skew at that budget -- max/mean over shards of how many queries touch each.
Load skew is what caps throughput, so this is the test of whether the navigation
graph buys routing quality that centroid distance does not.

Fidelity caveat: the upper graph is rebuilt with hnswlib using the same
parameters as build_orion_layout.py. This is the legacy dual-graph variant that
agent.md flags as an ablation; it is a faithful encoding of the routing *rule*,
not a bit-for-bit match of the Rust router. Good enough for the structural
question, not for deployable numbers.

Needs the system libstdc++ ahead of conda's for hnswlib:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 orion_router_skew.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402

from fanout_headroom import (  # noqa: E402
    Layout,
    greedy_coverage_order,
    neighbor_masks,
    shard_centroids,
)
from load_skew import centroid_order, router_stats  # noqa: E402


def orion_navigation_order(
    upper_labels: np.ndarray, layout: Layout
) -> np.ndarray:
    """Shard order = first appearance of each shard along the ranked L1 hits.

    Shards a query never navigates to are appended in id order so the ordering
    spans all P shards; those tail shards only matter if the target is
    unreachable within the navigated set, which the recall curve then exposes.
    """
    queries = upper_labels.shape[0]
    order = np.empty((queries, layout.shards), dtype=np.int32)
    for query in range(queries):
        seen = np.zeros(layout.shards, dtype=bool)
        cursor = 0
        for label in upper_labels[query]:
            point = int(label)
            if not (0 <= point < layout.points):
                continue
            for shard in layout.shards_of(point):
                if not seen[shard]:
                    seen[shard] = True
                    order[query, cursor] = shard
                    cursor += 1
        if cursor < layout.shards:
            for shard in range(layout.shards):
                if not seen[shard]:
                    order[query, cursor] = shard
                    cursor += 1
    return order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--layout-file", required=True)
    parser.add_argument("--vector-distance", choices=("cosine", "euclid", "l2"), required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--upper-query-k", type=int, default=200)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-graph-seed", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as handle:
        train = np.asarray(handle["train"], dtype=np.float32)
        queries = np.asarray(handle["test"], dtype=np.float32)
        ground_truth = np.asarray(handle["neighbors"], dtype=np.int64)

    layout = Layout.load(args.layout_file)
    if layout.points != train.shape[0]:
        raise SystemExit(
            f"layout covers {layout.points} points, dataset has {train.shape[0]}"
        )

    distance_config = experiment.vector_distance_config(args.vector_distance)
    upper_indices = experiment.global_upper_indices(
        len(train), int(args.sample_denominator), int(args.upper_sample_seed)
    )
    upper_index = experiment.build_upper_index(
        train[upper_indices],
        upper_indices.astype(np.int64, copy=False),
        int(train.shape[1]),
        int(args.upper_m),
        int(args.upper_ef_construction),
        int(args.upper_query_k),
        distance_config["hnsw_space"],
        random_seed=int(args.upper_graph_seed),
        construction_threads=-1,
    )
    query_k = min(int(args.upper_query_k), upper_index.get_current_count())
    upper_index.set_ef(max(query_k, int(args.upper_ef_construction)))
    upper_labels, _ = upper_index.knn_query(queries, k=query_k)

    masks = neighbor_masks(layout, ground_truth, args.top_k)
    centroids = shard_centroids(layout, train)
    orders = {
        "oracle": greedy_coverage_order(masks, args.top_k),
        "centroid": centroid_order(queries, centroids),
        "orion": orion_navigation_order(upper_labels, layout),
    }

    report: dict[str, object] = {
        "dataset": Path(args.hdf5).name,
        "layout": Path(args.layout_file).stem,
        "shards": layout.shards,
        "recall_target": args.target,
        "upper_query_k": query_k,
        "size_skew_max_over_mean": float(
            layout.copies_per_shard().max() / layout.copies_per_shard().mean()
        ),
        "routers": {
            name: router_stats(order, masks, args.top_k, args.target, layout.shards)
            for name, order in orders.items()
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{report['dataset']:24} P={layout.shards} target={args.target}")
    print(f"{'router':10} {'fixed_B':>8} {'load_skew':>10} {'adapt_fanout':>13} {'unreached':>10}")
    for name, stats in report["routers"].items():
        print(
            f"{name:10} {stats['fixed_budget']:8d} "
            f"{stats['load_skew_max_over_mean']:10.2f} "
            f"{stats['adaptive_mean_fanout']:13.2f} "
            f"{stats['queries_target_unreachable']:10d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
