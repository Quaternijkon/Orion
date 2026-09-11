"""Offline de-risk before building a replicated collection: does load-aware
least-loaded-replica routing lower W under Orion's *real* navigation router
(not the oracle-coverage router of analysis/replication_load_balance.py)?

Difference from the oracle probe: here the probed set is what Orion actually
probes online -- the shards that the query's top upper-k L1 hits land on -- and
we replicate only the *hot upper (L1) nodes* onto their nearest shards. The nav
router then has a genuine per-hit choice, which we use to (1) consolidate hits
onto already-probed shards (keep fan-out low) and (2) send new shards to the
least-loaded replica (balance load). If this reproduces the W drop with realistic
routing, a cluster build is justified; if not, it is not.

Arms (knob = upper-k, swept), scored on TEST queries:
  A  base single-assign      + oblivious (each L1 hit -> its one home shard)
  D  hot-L1 replication       + load-aware (consolidate, then least-loaded replica)

Hotness of an upper node = how often it appears in the top upper-k of TRAIN proxy
queries (no test leakage). Recall = coverage of ground-truth neighbors by the
probed shard set under the *base* single-assign membership of all points.

Needs system libstdc++ ahead of conda for hnswlib:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 nav_load_probe.py ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402
from fanout_headroom import Layout, shard_centroids  # noqa: E402


def upper_hit_counts(index, train, rng, n_proxy, upper_k, n_upper):
    """Hotness proxy: hit frequency of each upper node over random train queries."""
    if n_proxy < train.shape[0]:
        sample = rng.choice(train.shape[0], size=n_proxy, replace=False)
        probe = train[sample]
    else:
        probe = train
    labels, _ = index.knn_query(probe, k=upper_k)
    counts = np.bincount(labels.reshape(-1), minlength=n_upper).astype(np.int64)
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--train-path", required=True)
    p.add_argument("--layout-file", required=True, help="base single-assign kmeans .npz")
    p.add_argument("--vector-distance", choices=("cosine", "euclid", "l2"), default="cosine")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--upper-k", type=int, nargs="+", default=[8, 12, 16, 24])
    p.add_argument("--replicate-frac", type=float, default=0.10, help="frac of upper nodes")
    p.add_argument("--replicas", type=int, default=3)
    p.add_argument("--n-proxy", type=int, default=100000)
    p.add_argument("--dump-layout", default="", help="write the replicated CSR layout .npz and continue")
    # upper-graph params: keep in sync with build_routed_requests defaults
    p.add_argument("--sample-denominator", type=int, default=32)
    p.add_argument("--upper-sample-seed", type=int, default=100)
    p.add_argument("--upper-m", type=int, default=32)
    p.add_argument("--upper-ef-construction", type=int, default=100)
    p.add_argument("--upper-graph-seed", type=int, default=100)
    p.add_argument("--upper-query-k", type=int, default=200)
    args = p.parse_args()

    train = np.load(args.train_path).astype(np.float32)
    if args.normalize:
        train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    with h5py.File(args.hdf5, "r") as h:
        queries = np.asarray(h["test"], dtype=np.float32)
        ground_truth = np.asarray(h["neighbors"], dtype=np.int64)[:, : args.top_k]
    if args.normalize:
        queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)

    dist = experiment.vector_distance_config(args.vector_distance)
    base = Layout.load(args.layout_file)
    n = base.points
    shards = base.shards
    home = base.indices.astype(np.int64)  # single-assign home shard per point
    centroids = shard_centroids(base, train)

    # Build the deployed upper graph and route the test queries.
    upper_idx = experiment.global_upper_indices(
        n, int(args.sample_denominator), int(args.upper_sample_seed)
    )
    index = experiment.build_upper_index(
        train[upper_idx], upper_idx.astype(np.int64), int(train.shape[1]),
        int(args.upper_m), int(args.upper_ef_construction), int(args.upper_query_k),
        dist["hnsw_space"], random_seed=int(args.upper_graph_seed), construction_threads=-1,
    )
    n_upper = index.get_current_count()
    index.set_ef(max(int(args.upper_query_k), int(args.upper_ef_construction)))
    max_uk = max(args.upper_k)
    # knn_query returns GLOBAL point ids (the labels the index was built with).
    test_labels, _ = index.knn_query(queries, k=min(int(args.upper_query_k), n_upper))
    test_labels = test_labels.astype(np.int64)  # global point ids per query

    # Hotness in global-id space, from train proxy queries.
    rng = np.random.default_rng(0)
    counts = upper_hit_counts(index, train, rng, int(args.n_proxy),
                              min(max_uk, n_upper), n)  # counts[global_id]

    # owners: dict global_id -> shards holding it. Default (non-replicated) owner
    # is the point's base home shard; only hot upper nodes get extra copies.
    owners: dict[int, list[int]] = {}
    upper_home = home[upper_idx]
    hot_rank = np.argsort(counts[upper_idx])[::-1]
    n_hot = int(args.replicate_frac * n_upper)
    hot_local = hot_rank[:n_hot]
    hot_global = upper_idx[hot_local]
    hv = train[hot_global]
    d = ((hv**2).sum(1)[:, None] - 2 * hv @ centroids.T + (centroids**2).sum(1)[None, :])
    nearest = np.argsort(d, axis=1)[:, : args.replicas]
    extra_copies = 0
    for row, gid in enumerate(hot_global.tolist()):
        picks = [int(home[gid])]
        for s in nearest[row].tolist():
            if s not in picks:
                picks.append(int(s))
            if len(picks) >= args.replicas:
                break
        extra_copies += len(picks) - 1
        owners[gid] = picks
    expansion = 1.0 + extra_copies / n  # overhead is tiny (only hot upper nodes)

    def owner_of(gid: int) -> list[int]:
        o = owners.get(gid)
        return o if o is not None else [int(home[gid])]

    if args.dump_layout:
        lists = [[int(home[i])] for i in range(n)]
        for gid, picks in owners.items():
            lists[gid] = list(picks)
        indptr = np.zeros(n + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(x) for x in lists])
        indices = np.fromiter((s for x in lists for s in x), dtype=np.int32,
                              count=int(indptr[-1]))
        np.savez(args.dump_layout, indptr=indptr, indices=indices,
                 shard_count=np.int64(shards))
        print(f"dumped replicated layout -> {args.dump_layout} "
              f"copies={len(indices)} points={n} expansion={len(indices)/n:.4f}")

    # base membership of ALL points, for recall coverage
    gt_home = home[ground_truth]  # (Q, k) home shard of each true neighbor

    def recall_of(probed_sets):
        cov = np.zeros(len(queries))
        for q in range(len(queries)):
            ps = probed_sets[q]
            cov[q] = np.isin(gt_home[q], list(ps)).mean()
        return float(cov.mean())

    print(f"P={shards} n_upper={n_upper} hot={n_hot} expansion={expansion:.4f}")
    print(f"{'arm':22s} {'uk':>3s} {'fanout':>7s} {'skew':>6s} {'W':>7s} {'recall':>7s}")
    for uk in args.upper_k:
        hits = test_labels[:, :uk]
        # Arm A: oblivious single-assign
        loadA = np.zeros(shards, dtype=np.int64)
        probedA = []
        foA = np.zeros(len(queries))
        for q in range(len(queries)):
            ps = set(int(home[u]) for u in hits[q])
            probedA.append(ps)
            for s in ps:
                loadA[s] += 1
            foA[q] = len(ps)
        skewA = loadA.max() / loadA.mean()
        recA = recall_of(probedA)
        WA = foA.mean() * skewA
        print(f"{'A single/oblivious':22s} {uk:3d} {foA.mean():7.2f} {skewA:6.2f} "
              f"{WA:7.2f} {recA:7.3f}")

        # Arm D: hot-L1 replication + load-aware (consolidate, then least-loaded)
        loadD = np.zeros(shards, dtype=np.int64)
        probedD = []
        foD = np.zeros(len(queries))
        for q in range(len(queries)):
            ps: set[int] = set()
            for u in hits[q]:
                own = owner_of(int(u))
                if len(own) == 1:
                    ps.add(own[0])
                    continue
                inter = [s for s in own if s in ps]
                if inter:
                    continue  # already covered -> no new shard (consolidate)
                # pick least-loaded replica
                s = min(own, key=lambda x: loadD[x])
                ps.add(int(s))
            probedD.append(ps)
            for s in ps:
                loadD[s] += 1
            foD[q] = len(ps)
        skewD = loadD.max() / loadD.mean()
        recD = recall_of(probedD)
        WD = foD.mean() * skewD
        print(f"{'D repl/load-aware':22s} {uk:3d} {foD.mean():7.2f} {skewD:6.2f} "
              f"{WD:7.2f} {recD:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
