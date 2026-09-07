"""Tier 2: measure adaptive vs uniform within-shard ef on real per-shard HNSW.

Finding 6 argued, under a linear-ef cost model, that Orion's per-shard
`ef_s = base_ef + factor * hits_s` spends less within-shard work than one global
ef. That was a model bound. This script removes the model: it builds a real
hnswlib index per shard, routes each query with the upper graph to a probed
shard set, and searches those shards two ways on the *same* routing:

  uniform  -- every probed shard at one global ef E (standard scatter-gather)
  adaptive -- shard s at ef_s = round(alpha * (base_ef + factor * hits_s))

Both merge per-shard top-k into a global top-k and score Recall@10 against
ground truth. Work is total candidate depth per query, Sum_s ef_s, the standard
HNSW cost proxy. Sweeping E and alpha traces two recall-vs-work frontiers; the
ratio of uniform work to adaptive work at a matched recall is the real speedup.

Scope: this isolates ef *strength*. It does NOT model the entry-point mechanism
(stock hnswlib enters from the top of its own hierarchy and takes no custom entry
points); that gain needs the patched Qdrant and belongs to the system test.

Needs the system libstdc++ ahead of conda's for hnswlib:
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 within_shard_measure.py ...
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

from fanout_headroom import Layout  # noqa: E402


def points_by_shard(layout) -> list[np.ndarray]:
    """Invert the point->shard CSR into shard->points (a point may appear twice)."""
    owners = np.repeat(np.arange(layout.points), np.diff(layout.indptr))
    order = np.argsort(layout.indices, kind="stable")
    sorted_shards = layout.indices[order]
    sorted_owners = owners[order]
    bounds = np.searchsorted(sorted_shards, np.arange(layout.shards + 1))
    return [sorted_owners[bounds[s] : bounds[s + 1]] for s in range(layout.shards)]


def build_shard_indices(train, layout, space, m, ef_construction, seed):
    import hnswlib

    members = points_by_shard(layout)
    indices = []
    for shard in range(layout.shards):
        point_ids = members[shard]
        index = hnswlib.Index(space=space, dim=train.shape[1])
        index.init_index(
            max_elements=len(point_ids), ef_construction=ef_construction, M=m, random_seed=seed
        )
        index.add_items(train[point_ids], point_ids, num_threads=-1)
        indices.append(index)
    return indices


def evaluate(indices, queries, probed, ef_table, gt_sets, top_k, threads):
    """Return (mean_recall, mean_total_work) for a per-(query,shard) ef assignment.

    probed[q]     : int array of shards query q searches
    ef_table[q]   : int array of ef for each of those shards (aligned with probed[q])
    """
    n_q = len(queries)
    # Gather, per shard, the queries that touch it and their ef, then batch by ef.
    per_shard: dict[int, list[tuple[int, int]]] = {}
    total_work = np.zeros(n_q, dtype=np.int64)
    for q in range(n_q):
        for s, ef in zip(probed[q], ef_table[q]):
            per_shard.setdefault(int(s), []).append((q, int(ef)))
            total_work[q] += int(ef)

    best_dist = [dict() for _ in range(n_q)]  # id -> distance (kept minimal)
    for shard, entries in per_shard.items():
        index = indices[shard]
        qs = np.array([q for q, _ in entries])
        efs = np.array([ef for _, ef in entries])
        for ef_val in np.unique(efs):
            sel = qs[efs == ef_val]
            index.set_ef(max(int(ef_val), top_k))
            labels, dists = index.knn_query(queries[sel], k=top_k, num_threads=threads)
            for local, q in enumerate(sel):
                bd = best_dist[q]
                for pid, d in zip(labels[local], dists[local]):
                    pid = int(pid)
                    if pid not in bd or d < bd[pid]:
                        bd[pid] = float(d)

    recalls = np.zeros(n_q)
    for q in range(n_q):
        if not best_dist[q]:
            continue
        ids = np.array(list(best_dist[q].keys()))
        ds = np.array(list(best_dist[q].values()))
        topk = ids[np.argsort(ds)[:top_k]]
        recalls[q] = len(gt_sets[q].intersection(topk.tolist())) / top_k
    return float(recalls.mean()), float(total_work.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--layout-file", required=True)
    parser.add_argument("--vector-distance", choices=("cosine", "euclid", "l2"), required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--upper-k", type=int, required=True)
    parser.add_argument("--base-ef", type=int, default=20)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--shard-m", type=int, default=16)
    parser.add_argument("--shard-ef-construction", type=int, default=100)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-graph-seed", type=int, default=100)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as handle:
        train = np.asarray(handle["train"], dtype=np.float32)
        queries = np.asarray(handle["test"], dtype=np.float32)
        ground_truth = np.asarray(handle["neighbors"], dtype=np.int64)
    if args.max_queries:
        queries = queries[: args.max_queries]
        ground_truth = ground_truth[: args.max_queries]

    layout = Layout.load(args.layout_file)
    distance_config = experiment.vector_distance_config(args.vector_distance)
    space = distance_config["hnsw_space"]
    if space == "cosine":
        for matrix in (train, queries):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.divide(matrix, np.maximum(norms, 1e-12), out=matrix)

    # Upper-graph routing: probed shards + hits per query at the tuned upper_k.
    upper_indices = experiment.global_upper_indices(
        len(train), int(args.sample_denominator), int(args.upper_sample_seed)
    )
    upper_index = experiment.build_upper_index(
        train[upper_indices], upper_indices.astype(np.int64, copy=False),
        int(train.shape[1]), int(args.upper_m), int(args.upper_ef_construction),
        int(args.upper_k), space, random_seed=int(args.upper_graph_seed),
        construction_threads=-1,
    )
    query_k = min(int(args.upper_k), upper_index.get_current_count())
    upper_index.set_ef(max(query_k, int(args.upper_ef_construction)))
    upper_labels, _ = upper_index.knn_query(queries, k=query_k)

    probed: list[np.ndarray] = []
    hits: list[np.ndarray] = []
    for q in range(len(queries)):
        counts: dict[int, int] = {}
        for label in upper_labels[q]:
            point = int(label)
            if 0 <= point < layout.points:
                for shard in layout.shards_of(point):
                    counts[shard] = counts.get(shard, 0) + 1
        probed.append(np.fromiter(counts.keys(), dtype=np.int64))
        hits.append(np.fromiter(counts.values(), dtype=np.int64))

    gt_sets = [set(ground_truth[q, : args.top_k].tolist()) for q in range(len(queries))]

    print(f"building {layout.shards} shard indices (m={args.shard_m}) ...", flush=True)
    indices = build_shard_indices(
        train, layout, space, args.shard_m, args.shard_ef_construction, args.upper_graph_seed
    )

    results = {"uniform": [], "adaptive": []}

    # Uniform: every probed shard at global ef E.
    for E in [10, 12, 16, 20, 28, 40, 60, 90, 130]:
        ef_table = [np.full(len(p), E, dtype=np.int64) for p in probed]
        recall, work = evaluate(indices, queries, probed, ef_table, gt_sets, args.top_k, args.threads)
        results["uniform"].append({"ef": E, "recall": recall, "work": work})
        print(f"  uniform  E={E:4d}  recall={recall:.4f}  work={work:.1f}", flush=True)

    # Adaptive: ef_s = round(alpha * (base_ef + factor * hits_s)).
    for alpha in [0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0, 4.5]:
        ef_table = [
            np.maximum(args.top_k, np.round(alpha * (args.base_ef + args.factor * h)).astype(np.int64))
            for h in hits
        ]
        recall, work = evaluate(indices, queries, probed, ef_table, gt_sets, args.top_k, args.threads)
        results["adaptive"].append({"alpha": alpha, "recall": recall, "work": work})
        print(f"  adaptive a={alpha:.1f}  recall={recall:.4f}  work={work:.1f}", flush=True)

    def work_at_recall(points, target):
        pts = sorted(points, key=lambda r: r["work"])
        for r in pts:
            if r["recall"] >= target:
                return r["work"]
        return None

    speedups = {}
    for target in (0.85, 0.90, 0.95):
        u = work_at_recall(results["uniform"], target)
        a = work_at_recall(results["adaptive"], target)
        speedups[f"{target}"] = {
            "uniform_work": u, "adaptive_work": a,
            "speedup": (u / a) if (u and a) else None,
        }

    report = {
        "dataset": Path(args.hdf5).name, "layout": Path(args.layout_file).stem,
        "shards": layout.shards, "upper_k": query_k, "queries": len(queries),
        "ef_model": {"base_ef": args.base_ef, "factor": args.factor},
        "curves": results, "speedup_at_matched_recall": speedups,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{report['dataset']} P={layout.shards}: measured adaptive-vs-uniform speedup")
    for target, s in speedups.items():
        if s["speedup"]:
            print(f"  @recall {target}: uniform work={s['uniform_work']:.1f} "
                  f"adaptive work={s['adaptive_work']:.1f} -> {s['speedup']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
