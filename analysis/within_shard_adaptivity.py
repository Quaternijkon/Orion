"""Does Orion's adaptive within-shard search strength actually buy anything?

Orion searches each probed shard at ef_s = base_ef + factor * (L1 hits on s)
(defaults 20 + 4 * hits). A shard that many of the query's nearest upper-tier
nodes fall on is searched harder; a shard grazed by a single L1 hit is searched
at the floor. This only helps if two conditions hold, both checkable offline
against ground truth without running a single HNSW search:

  (1) Concentration: the query's true neighbours are unevenly spread across the
      probed shards. If every probed shard held k/S of them, one global ef would
      be optimal and adaptive strength would be pure overhead.
  (2) Aimed correctly: the routing signal hits_s tracks t_s, the number of true
      neighbours the shard actually holds. Otherwise adaptive ef spends strength
      on the wrong shards.

Given both, the within-shard work saved versus one global ef is bounded by the
ef skew mean_s(ef_s) / max_s(ef_s): a flat router must raise every probed shard
to the strength the heaviest one needs, while Orion pays that only where needed.

Cost model: HNSW search cost is taken proportional to ef (candidate-list depth),
the standard first-order proxy. The exact speedup depends on the within-shard
recall(ef) curve, which needs the real per-shard index (Tier 2); this script
establishes the mechanism and a model-based magnitude.
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

from fanout_headroom import Layout, neighbor_masks, popcount_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--layout-file", required=True)
    parser.add_argument("--vector-distance", choices=("cosine", "euclid", "l2"), required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--upper-k", type=int, required=True, help="deployed tuned upper_k")
    parser.add_argument("--base-ef", type=int, default=20)
    parser.add_argument("--factor", type=int, default=4)
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
        int(args.upper_k),
        distance_config["hnsw_space"],
        random_seed=int(args.upper_graph_seed),
        construction_threads=-1,
    )
    query_k = min(int(args.upper_k), upper_index.get_current_count())
    upper_index.set_ef(max(query_k, int(args.upper_ef_construction)))
    upper_labels, _ = upper_index.knn_query(queries, k=query_k)

    masks = neighbor_masks(layout, ground_truth, args.top_k)
    table = popcount_table(args.top_k)

    n_q = queries.shape[0]
    top1_share = np.zeros(n_q)          # fraction of covered neighbours on the heaviest shard
    covered = np.zeros(n_q)             # true neighbours reachable within the probed set
    probed = np.zeros(n_q, dtype=np.int32)
    ef_work_ratio = np.zeros(n_q)       # adaptive total ef / uniform-at-max total ef
    marginal_frac = np.zeros(n_q)       # fraction of probed shards holding <=1 true neighbour
    hits_t_agree = np.zeros(n_q)        # does the max-hits shard equal the max-neighbour shard?
    # How fast neighbours are captured when shards are ordered by the routing signal
    # (hits) versus by the oracle (true neighbour count).
    recover_by_hits = []
    recover_by_oracle = []

    for q in range(n_q):
        hit_counts: dict[int, int] = {}
        for label in upper_labels[q][:query_k]:
            point = int(label)
            if 0 <= point < layout.points:
                for shard in layout.shards_of(point):
                    hit_counts[shard] = hit_counts.get(shard, 0) + 1
        if not hit_counts:
            continue
        shards = np.fromiter(hit_counts.keys(), dtype=np.int64)
        hits = np.fromiter(hit_counts.values(), dtype=np.int64)
        t = table[masks[q, shards]].astype(np.int64)  # true neighbours per probed shard

        S = shards.size
        probed[q] = S
        total_t = int(t.sum())
        covered[q] = total_t / args.top_k
        if total_t > 0:
            top1_share[q] = t.max() / total_t
        marginal_frac[q] = float(np.mean(t <= 1))

        ef = args.base_ef + args.factor * hits
        ef_work_ratio[q] = ef.mean() / ef.max()
        hits_t_agree[q] = float(shards[hits.argmax()] == shards[t.argmax()])

        # cumulative neighbour recovery, shards added by descending hits vs descending t
        order_hits = np.argsort(-hits, kind="stable")
        order_oracle = np.argsort(-t, kind="stable")
        cum_hits = np.cumsum(t[order_hits]) / max(total_t, 1)
        cum_oracle = np.cumsum(t[order_oracle]) / max(total_t, 1)
        recover_by_hits.append(cum_hits)
        recover_by_oracle.append(cum_oracle)

    def frac_at(curves, depth):
        vals = [c[min(depth - 1, len(c) - 1)] for c in curves if len(c)]
        return float(np.mean(vals))

    report = {
        "dataset": Path(args.hdf5).name,
        "layout": Path(args.layout_file).stem,
        "shards": layout.shards,
        "upper_k": query_k,
        "ef_model": {"base_ef": args.base_ef, "factor": args.factor},
        "probed_shards_mean": float(probed.mean()),
        "neighbour_coverage_mean": float(covered.mean()),
        "concentration_top1_share_mean": float(top1_share.mean()),
        "marginal_shard_fraction_mean": float(marginal_frac.mean()),
        "proxy_argmax_agreement": float(hits_t_agree.mean()),
        "recover_top1_by_hits": frac_at(recover_by_hits, 1),
        "recover_top1_by_oracle": frac_at(recover_by_oracle, 1),
        "recover_top2_by_hits": frac_at(recover_by_hits, 2),
        "recover_top2_by_oracle": frac_at(recover_by_oracle, 2),
        "adaptive_ef_work_ratio_mean": float(ef_work_ratio.mean()),
        "adaptive_speedup_vs_uniform_max_mean": float((1.0 / ef_work_ratio).mean()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{report['dataset']:24} P={layout.shards} upper_k={query_k}")
    print(f"  probed shards/query      : {report['probed_shards_mean']:.2f}")
    print(f"  neighbour coverage       : {report['neighbour_coverage_mean']:.3f}")
    print(f"  (1) top-1 shard's share  : {report['concentration_top1_share_mean']:.2f}  "
          f"marginal(<=1 nbr) shards : {report['marginal_shard_fraction_mean']:.2f}")
    print(f"  (2) argmax(hits)==argmax(t): {report['proxy_argmax_agreement']:.2f}  "
          f"| top-1-by-hits captures {report['recover_top1_by_hits']:.2f} vs oracle "
          f"{report['recover_top1_by_oracle']:.2f}")
    print(f"  adaptive-ef work ratio    : {report['adaptive_ef_work_ratio_mean']:.2f}  "
          f"(=> {report['adaptive_speedup_vs_uniform_max_mean']:.2f}x less within-shard work "
          f"than uniform-at-max)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
