"""Probe: can selective replication of hot regions plus load-aware routing lower
query load skew *without* inflating fan-out, i.e. actually raise throughput?

Throughput is capped by the busiest shard, W = fan-out x load_skew. Moving points
cannot beat this trade-off (a query's neighbors are clustered, so spreading them
raises fan-out; that was the negative result of the weighted-balance probe). The
one lever that can spread load at fixed fan-out is *replication*: put a hot
point's copies on several shards, and let the router cover it from whichever
replica is least loaded. With single assignment the cover of a query's neighbor
set is forced, so a load-aware router has no freedom; replication is what unlocks
it.

Four arms, all under full-coverage (required-fan-out) routing on TEST queries:
  A  single-assign            + load-oblivious cover   (baseline)
  C  hot-point replication    + load-oblivious cover
  D  hot-point replication    + load-aware cover        (proposed)
(single-assign + load-aware equals A, since the cover is forced, so it is omitted.)

Hot points are the top --replicate-frac by an offline load weight (hubness);
each is copied onto its --replicas nearest shard centroids. Load weights come
from TRAIN, routing is scored on TEST, so there is no leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis"))
from fanout_headroom import Layout, neighbor_masks, popcount_table  # noqa: E402


def shard_centroids_from_assignment(train, assignment, shards):
    cent = np.zeros((shards, train.shape[1]), dtype=np.float64)
    sizes = np.bincount(assignment, minlength=shards).astype(np.float64)
    np.add.at(cent, assignment, train)
    cent /= np.maximum(sizes, 1)[:, None]
    return cent.astype(np.float32)


def build_replicated_layout(assignment, centroids, weights, frac, replicas, shards):
    """Copy the top-`frac` hottest points onto their `replicas` nearest shards."""
    n = assignment.shape[0]
    lists = [[int(assignment[i])] for i in range(n)]
    if frac > 0.0 and replicas > 1:
        n_hot = int(frac * n)
        hot = np.argsort(weights)[::-1][:n_hot]
        # nearest shards for each hot point (by centroid distance)
        hp = np.asarray(hot, dtype=np.int64)
        d = (
            (train_ref[hp] ** 2).sum(1)[:, None]
            - 2.0 * train_ref[hp] @ centroids.T
            + (centroids**2).sum(1)[None, :]
        )
        nearest = np.argsort(d, axis=1)[:, :replicas]
        for row, point in enumerate(hp.tolist()):
            home = assignment[point]
            picks = [int(home)]
            for s in nearest[row].tolist():
                if s not in picks:
                    picks.append(int(s))
                if len(picks) >= replicas:
                    break
            lists[point] = picks
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(x) for x in lists])
    indices = np.fromiter((s for x in lists for s in x), dtype=np.int32,
                          count=int(indptr[-1]))
    return Layout(indptr, indices, shards)


def route(masks, top_k, target, load_aware, seed=0):
    """Full-coverage greedy cover per query; optionally break choices by load.

    Returns per-query fan-out (shards touched) and per-shard touch counts.
    Coverage progresses by max new-neighbor gain; when load_aware, ties (and the
    replica choices they expose) are broken toward the least loaded shard.
    """
    table = popcount_table(top_k)
    queries, shards = masks.shape
    need = int(np.ceil(target * top_k))
    load = np.zeros(shards, dtype=np.int64)
    fanout = np.zeros(queries, dtype=np.int32)
    order = np.arange(queries)
    if load_aware:
        np.random.default_rng(seed).shuffle(order)
    for q in order.tolist():
        row = masks[q]
        covered = np.uint32(0)
        touched = 0
        while table[int(covered)] < need and touched < shards:
            gain = table[(row & ~covered).astype(np.uint32)].astype(np.int64)
            gain[load < 0] = -1  # placeholder, no-op
            avail = gain > 0
            if not avail.any():
                break
            if load_aware:
                # among positive-gain shards, prefer high gain then low load
                score = np.where(avail, gain * 10_000 - load, -1)
            else:
                score = np.where(avail, gain, -1)
            s = int(np.argmax(score))
            covered |= row[s]
            load[s] += 1
            touched += 1
        fanout[q] = touched
    return fanout, load


def summarize(name, fanout, load):
    mean = load.mean()
    skew = float(load.max() / mean) if mean > 0 else 0.0
    fo = float(fanout.mean())
    W = fo * skew
    print(f"  {name:34s} fanout={fo:5.2f}  load_skew={skew:5.2f}  W={W:6.2f}")
    return {"fanout": fo, "load_skew": skew, "W": W}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--train-path", required=True)
    p.add_argument("--layout-file", required=True, help="single-assign base layout .npz")
    p.add_argument("--weights-path", required=True, help="per-point load weights .npy")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--target", type=float, default=0.9)
    p.add_argument("--replicate-frac", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--replicas", type=int, nargs="+", default=[2, 3])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    global train_ref
    train_ref = np.load(args.train_path).astype(np.float32)
    if args.normalize:
        train_ref /= np.maximum(np.linalg.norm(train_ref, axis=1, keepdims=True), 1e-12)
    with h5py.File(args.hdf5, "r") as h:
        ground_truth = np.asarray(h["neighbors"], dtype=np.int64)
    weights = np.load(args.weights_path).astype(np.float64)

    base = Layout.load(args.layout_file)
    assignment = base.indices.copy()
    shards = base.shards
    centroids = shard_centroids_from_assignment(train_ref, assignment, shards)

    results = {}
    print(f"target per-query recall={args.target} top_k={args.top_k} P={shards}")
    # Arm A: single assignment baseline
    masks_A = neighbor_masks(base, ground_truth, args.top_k)
    fo, ld = route(masks_A, args.top_k, args.target, load_aware=False)
    results["A_single_oblivious"] = summarize("A single-assign / oblivious", fo, ld)

    for frac in args.replicate_frac:
        for R in args.replicas:
            lay = build_replicated_layout(assignment, centroids, weights, frac, R, shards)
            exp = lay.copies / lay.points
            masks = neighbor_masks(lay, ground_truth, args.top_k)
            tag = f"frac{frac:g}_R{R}"
            print(f"[{tag}] expansion={exp:.3f}")
            foc, ldc = route(masks, args.top_k, args.target, load_aware=False)
            r_c = summarize(f"C repl({tag}) / oblivious", foc, ldc)
            fod, ldd = route(masks, args.top_k, args.target, load_aware=True)
            r_d = summarize(f"D repl({tag}) / load-aware", fod, ldd)
            r_c["expansion"] = r_d["expansion"] = exp
            results[f"C_{tag}"] = r_c
            results[f"D_{tag}"] = r_d

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
