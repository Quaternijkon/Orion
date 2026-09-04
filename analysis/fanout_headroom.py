"""Exactly computable fan-out quantities for a sharded vector index.

Everything here is derived from ground-truth neighbors and a shard assignment
alone. There is no index, no serving, no surrogate metric, and no fitted model,
so the numbers cannot be confounded by a load generator or by HNSW parameters.

Two quantities are computed.

Q1, the *required fan-out* of a query,

    F*(q) = |{ sigma(x) : x in NN_k(q) }|,

is the number of shards that hold at least one true neighbor. It is a
mechanism-free lower bound on the shards any correct router must touch.

Q2 decomposes the cost of serving at a target recall R along two independent
axes, because a router makes two separable decisions: in what order to consider
shards, and how many to actually search.

The *ordering* is the router's estimate of where the neighbors are:

  * ``oracle`` ranks a query's shards by how many true neighbors they hold. It
    is unavailable at serving time and exists to give the floor that any
    ordering mechanism is chasing.
  * ``centroid`` ranks by distance from the query to each shard's centroid,
    which is what an IVF-style or k-means router actually does.

The *budget* is either one number for the whole workload or one per query:

  * fixed: F_R = min { f : E_q[recall_f(q)] >= R }, the smallest budget that
    meets the mean-recall service level this project's benchmarks target.
  * adaptive: A_R, the minimum mean number of shards subject to
    E_q[recall] >= R. It is computed as a Lagrangian lower bound, so it is
    valid even when marginal gains are not monotone in rank (which they are
    not under centroid ordering), and any ratio built on it is an upper bound
    on the achievable gain.

Two ratios follow, and they answer different questions:

    H_adaptive = F_R / A_R          within one ordering, what per-query
                                    budgeting can win over a single budget
    H_ordering = F_R^centroid / F_R^oracle
                                    at a fixed budget policy, what a better
                                    shard-ranking mechanism can win

Reporting both matters because they are easy to confuse and can differ by an
order of magnitude. If H_adaptive is near 1 while H_ordering is large, then a
system's advantage cannot come from adaptivity and must come from ranking
shards better -- which is a statement about what mechanism to build.

A per-query view of the same skew is also reported: a_R(q) = min { j :
recall_j(q) >= R }. A fixed budget that must satisfy nearly every query pays a
high quantile of a_R while an adaptive one pays its mean, so quantile/mean
measures what the distribution's tail costs a fixed policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class Distribution:
    mean: float
    median: float
    std: float
    p90: float
    p95: float
    p99: float
    max: float

    @staticmethod
    def of(values: np.ndarray) -> "Distribution":
        return Distribution(
            mean=float(values.mean()),
            median=float(np.median(values)),
            std=float(values.std()),
            p90=float(np.percentile(values, 90)),
            p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99)),
            max=float(values.max()),
        )


def build_layout(
    train: np.ndarray, shards: int, method: str, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Assign every base vector to a shard.

    Returns the assignment, the per-shard centroids that a serving-time router
    would rank with, and balance statistics.
    """
    generator = np.random.default_rng(seed)
    if method == "random":
        assignment = generator.integers(0, shards, size=train.shape[0], dtype=np.int32)
    elif method in ("kmeans", "minibatch_kmeans"):
        from sklearn.cluster import KMeans, MiniBatchKMeans

        estimator = (
            MiniBatchKMeans(
                n_clusters=shards,
                random_state=seed,
                n_init=3,
                batch_size=4096,
                max_iter=200,
            )
            if method == "minibatch_kmeans"
            else KMeans(n_clusters=shards, random_state=seed, n_init=1, max_iter=50)
        )
        assignment = estimator.fit_predict(train).astype(np.int32)
    elif method == "balanced_kmeans":
        assignment = balanced_kmeans(train, shards, seed)
    else:
        raise ValueError(f"unknown layout method {method!r}")

    # Centroids are recomputed from the final assignment so that they describe
    # the shards that actually exist, including after balance repair.
    centroids = np.zeros((shards, train.shape[1]), dtype=np.float64)
    sizes = np.bincount(assignment, minlength=shards).astype(np.float64)
    np.add.at(centroids, assignment, train)
    centroids /= np.maximum(sizes, 1)[:, None]

    return (
        assignment,
        centroids.astype(np.float32),
        {
            "min_shard_size": float(sizes.min()),
            "max_shard_size": float(sizes.max()),
            "max_over_mean": float(sizes.max() / sizes.mean()),
            "size_cv": float(sizes.std() / sizes.mean()),
        },
    )


def balanced_kmeans(train: np.ndarray, shards: int, seed: int) -> np.ndarray:
    """K-means centroids followed by an exact equal-quota assignment.

    Points are assigned to their nearest centroid with a hard capacity of
    ceil(N/P) per shard, resolving contention in order of how much a point
    prefers its best centroid over its second best.
    """
    from sklearn.cluster import MiniBatchKMeans

    estimator = MiniBatchKMeans(
        n_clusters=shards, random_state=seed, n_init=3, batch_size=4096, max_iter=200
    ).fit(train)
    centroids = estimator.cluster_centers_.astype(np.float32)

    count = train.shape[0]
    capacity = -(-count // shards)
    distances = np.empty((count, shards), dtype=np.float32)
    for index in range(0, count, 100_000):
        window = train[index : index + 100_000]
        distances[index : index + 100_000] = (
            np.linalg.norm(window[:, None, :] - centroids[None, :, :], axis=2)
        )

    order_by_shard = np.argsort(distances, axis=1)
    best = distances[np.arange(count), order_by_shard[:, 0]]
    second = distances[np.arange(count), order_by_shard[:, 1]]
    priority = np.argsort(second - best)[::-1]

    assignment = np.full(count, -1, dtype=np.int32)
    remaining = np.full(shards, capacity, dtype=np.int64)
    for point in priority:
        for shard in order_by_shard[point]:
            if remaining[shard] > 0:
                assignment[point] = shard
                remaining[shard] -= 1
                break
    if (assignment < 0).any():
        raise RuntimeError("balanced assignment left points unplaced")
    return assignment


class Layout:
    """A point-to-shard mapping, allowing a point to occupy several shards.

    Stored in CSR form so that single-assignment and multi-assignment layouts
    are handled by the same code; ``indptr[i+1] - indptr[i] == 1`` for every
    point in the single-assignment case.
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, shards: int) -> None:
        self.indptr = indptr
        self.indices = indices
        self.shards = shards

    @staticmethod
    def from_assignment(assignment: np.ndarray, shards: int) -> "Layout":
        return Layout(
            np.arange(assignment.size + 1, dtype=np.int64),
            assignment.astype(np.int32, copy=False),
            shards,
        )

    @staticmethod
    def load(path: str | Path) -> "Layout":
        archive = np.load(path, allow_pickle=False)
        return Layout(
            archive["indptr"].astype(np.int64),
            archive["indices"].astype(np.int32),
            int(archive["shard_count"]),
        )

    @property
    def points(self) -> int:
        return self.indptr.size - 1

    @property
    def copies(self) -> int:
        return int(self.indices.size)

    def shards_of(self, point: int) -> np.ndarray:
        return self.indices[self.indptr[point] : self.indptr[point + 1]]

    def copies_per_shard(self) -> np.ndarray:
        return np.bincount(self.indices, minlength=self.shards).astype(np.int64)


def shard_centroids(layout: Layout, train: np.ndarray) -> np.ndarray:
    """Mean vector of each shard's members, counting a replicated point once per shard."""
    centroids = np.zeros((layout.shards, train.shape[1]), dtype=np.float64)
    owners = np.repeat(np.arange(layout.points), np.diff(layout.indptr))
    np.add.at(centroids, layout.indices, train[owners])
    counts = layout.copies_per_shard()
    centroids /= np.maximum(counts, 1)[:, None]
    return centroids.astype(np.float32)


def neighbor_masks(
    layout: Layout, ground_truth: np.ndarray, top_k: int
) -> np.ndarray:
    """masks[q, s] has bit j set when shard s holds query q's j-th true neighbor.

    A bitmask rather than a count, because with replication the same neighbor
    can sit on several shards and recall must not double-count it.
    """
    queries = ground_truth.shape[0]
    masks = np.zeros((queries, layout.shards), dtype=np.uint32)
    neighbors = ground_truth[:, :top_k]
    for rank in range(top_k):
        column = neighbors[:, rank]
        starts = layout.indptr[column]
        stops = layout.indptr[column + 1]
        bit = np.uint32(1 << rank)
        widths = stops - starts
        for width in np.unique(widths):
            selected = np.nonzero(widths == width)[0]
            for offset in range(int(width)):
                shard = layout.indices[starts[selected] + offset]
                masks[selected, shard] |= bit
    return masks


def popcount_table(top_k: int) -> np.ndarray:
    return np.array(
        [bin(value).count("1") for value in range(1 << top_k)], dtype=np.int16
    )


def greedy_coverage_order(masks: np.ndarray, top_k: int) -> np.ndarray:
    """Order shards by marginal coverage: at each step take the most new neighbors.

    This is the multi-assignment generalization of ranking shards by how many
    true neighbors they hold, and is the best ordering an oracle router could
    use. With disjoint shards it reduces exactly to that ranking.
    """
    table = popcount_table(top_k)
    queries, shards = masks.shape
    order = np.empty((queries, shards), dtype=np.int32)
    covered = np.zeros(queries, dtype=np.uint32)
    available = np.ones((queries, shards), dtype=bool)
    rows = np.arange(queries)

    for step in range(shards):
        marginal = table[masks & ~covered[:, None]].astype(np.int32)
        marginal[~available] = -1
        choice = np.argmax(marginal, axis=1)
        order[:, step] = choice
        covered |= masks[rows, choice]
        available[rows, choice] = False
    return order


def recall_curve(masks: np.ndarray, order: np.ndarray, top_k: int) -> np.ndarray:
    """cumulative_recall[q, j] = recall from query q's first j+1 shards in ``order``."""
    table = popcount_table(top_k)
    rows = np.arange(masks.shape[0])[:, None]
    unioned = np.bitwise_or.accumulate(masks[rows, order], axis=1)
    return table[unioned].astype(np.float64) / top_k


def greedy_adaptive_shards(marginal: np.ndarray, top_k: int, target: float) -> int:
    """Minimum total shards for mean recall >= target, assuming sorted gains.

    Exact only when each query's marginal gains are non-increasing in rank, as
    they are under oracle ordering. Kept as an independent check on the general
    Lagrangian bound.
    """
    queries = marginal.shape[0]
    needed_hits = target * queries * top_k
    first_hits = float(marginal[:, 0].sum())
    if first_hits >= needed_hits:
        return queries

    extra = np.sort(marginal[:, 1:].ravel())[::-1]
    cumulative = first_hits + np.cumsum(extra, dtype=np.float64)
    reached = np.searchsorted(cumulative, needed_hits, side="left")
    if reached >= extra.size:
        raise RuntimeError("target recall unreachable even with every shard")
    return queries + int(reached) + 1


def adaptive_lower_bound(cumulative_recall: np.ndarray, target: float) -> float:
    """Lagrangian lower bound on the mean shards needed for mean recall >= target.

    For a price ``lam`` per shard each query independently minimizes
    ``(j + 1) * lam - recall_j``, which is exact by enumeration since the shard
    count is small. Sweeping ``lam`` traces the lower convex envelope of the
    cost-recall trade-off; interpolating across the crossing point gives a valid
    lower bound on the optimal adaptive cost, and therefore an upper bound on
    any gain measured against it.
    """
    budgets = np.arange(1, cumulative_recall.shape[1] + 1, dtype=np.float64)

    def solve(price: float) -> tuple[float, float]:
        choice = np.argmin(budgets[None, :] * price - cumulative_recall, axis=1)
        rows = np.arange(cumulative_recall.shape[0])
        return (
            float(budgets[choice].mean()),
            float(cumulative_recall[rows, choice].mean()),
        )

    shards_at_zero, recall_at_zero = solve(0.0)
    if recall_at_zero < target - 1e-12:
        raise RuntimeError("target recall unreachable even with every shard")

    low, high = 0.0, 1.0
    while solve(high)[1] >= target - 1e-12:
        high *= 2.0
        if high > 1e6:
            return 1.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if solve(middle)[1] >= target - 1e-12:
            low = middle
        else:
            high = middle

    shards_lo, recall_lo = solve(low)
    shards_hi, recall_hi = solve(high)
    if recall_lo <= recall_hi + 1e-15:
        return shards_lo
    weight = (target - recall_hi) / (recall_lo - recall_hi)
    weight = min(max(weight, 0.0), 1.0)
    return weight * shards_lo + (1.0 - weight) * shards_hi


def analyze_ordering(
    masks: np.ndarray,
    order: np.ndarray,
    top_k: int,
    recall_targets: list[float],
    tail_quantile: float,
    monotone: bool,
) -> dict[str, object]:
    cumulative_recall = recall_curve(masks, order, top_k)
    mean_recall_by_budget = cumulative_recall.mean(axis=0)
    result: dict[str, object] = {
        "mean_recall_by_budget": [
            float(value)
            for value in mean_recall_by_budget[: min(16, cumulative_recall.shape[1])]
        ],
        "targets": {},
    }

    for target in recall_targets:
        reached = cumulative_recall >= target - 1e-12
        if not reached.any(axis=1).all():
            raise RuntimeError(f"some query cannot reach recall {target}")
        per_query_need = np.argmax(reached, axis=1) + 1

        eligible = np.nonzero(mean_recall_by_budget >= target - 1e-12)[0]
        fixed_budget = int(eligible[0]) + 1
        adaptive_mean = adaptive_lower_bound(cumulative_recall, target)

        entry: dict[str, object] = {
            "per_query_need": asdict(Distribution.of(per_query_need)),
            "per_query_need_at_tail": float(
                np.percentile(per_query_need, tail_quantile * 100)
            ),
            "headroom_tail": float(
                np.percentile(per_query_need, tail_quantile * 100)
                / per_query_need.mean()
            ),
            "fixed_budget": fixed_budget,
            "adaptive_mean_shards": adaptive_mean,
            "headroom_adaptive": fixed_budget / adaptive_mean,
        }
        if monotone:
            marginal = np.diff(cumulative_recall, axis=1, prepend=0.0) * top_k
            greedy = (
                greedy_adaptive_shards(marginal, top_k, target)
                / cumulative_recall.shape[0]
            )
            entry["adaptive_mean_shards_greedy_check"] = greedy
            entry["greedy_matches_lagrangian"] = bool(
                abs(greedy - adaptive_mean) <= 0.02 * max(greedy, 1.0)
            )
        result["targets"][f"{target:.2f}"] = entry
    return result


def analyze(
    masks: np.ndarray,
    centroid_order: np.ndarray,
    top_k: int,
    recall_targets: list[float],
    tail_quantile: float,
) -> dict[str, object]:
    oracle_order = greedy_coverage_order(masks, top_k)
    oracle_recall = recall_curve(masks, oracle_order, top_k)
    # With replication the exact minimum cover is a set-cover instance; the
    # greedy cover is reported instead because it is what a router can realize,
    # and it coincides with the exact minimum when shards are disjoint.
    required_fanout = np.argmax(oracle_recall >= 1.0 - 1e-12, axis=1) + 1

    report: dict[str, object] = {
        "queries": int(masks.shape[0]),
        "top_k": top_k,
        "tail_quantile": tail_quantile,
        "required_fanout": asdict(Distribution.of(required_fanout)),
        "orderings": {
            "oracle": analyze_ordering(
                masks, oracle_order, top_k, recall_targets, tail_quantile, True
            ),
            "centroid": analyze_ordering(
                masks, centroid_order, top_k, recall_targets, tail_quantile, False
            ),
        },
    }

    report["headroom_ordering"] = {
        target: (
            report["orderings"]["centroid"]["targets"][target]["fixed_budget"]
            / report["orderings"]["oracle"]["targets"][target]["fixed_budget"]
        )
        for target in report["orderings"]["oracle"]["targets"]
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, help="ann-benchmarks style dataset")
    parser.add_argument(
        "--shards", type=int, default=0, help="ignored when --layout-file is given"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=(
            "random",
            "kmeans",
            "minibatch_kmeans",
            "balanced_kmeans",
            "layout_file",
        ),
    )
    parser.add_argument(
        "--layout-file",
        help="npz produced by build_orion_layout.py; required for --method layout_file",
    )
    parser.add_argument("--label", help="name for this layout in the report")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--recall-targets", type=float, nargs="+", default=[0.90, 0.95, 0.99]
    )
    parser.add_argument("--tail-quantile", type=float, default=0.95)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="unit-normalize before clustering, for angular datasets",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as handle:
        train = np.asarray(handle["train"], dtype=np.float32)
        queries = np.asarray(handle["test"], dtype=np.float32)
        ground_truth = np.asarray(handle["neighbors"], dtype=np.int64)
    if args.normalize:
        for matrix in (train, queries):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.divide(matrix, np.maximum(norms, 1e-12), out=matrix)

    if args.method == "layout_file":
        if not args.layout_file:
            raise SystemExit("--method layout_file requires --layout-file")
        layout = Layout.load(args.layout_file)
        if layout.points != train.shape[0]:
            raise SystemExit(
                f"layout covers {layout.points} points but dataset has "
                f"{train.shape[0]}"
            )
        centroids = shard_centroids(layout, train)
        copies = layout.copies_per_shard().astype(np.float64)
        balance = {
            "min_shard_size": float(copies.min()),
            "max_shard_size": float(copies.max()),
            "max_over_mean": float(copies.max() / copies.mean()),
            "size_cv": float(copies.std() / copies.mean()),
        }
    else:
        if args.shards <= 0:
            raise SystemExit("--shards must be positive for generated layouts")
        assignment, centroids, balance = build_layout(
            train, args.shards, args.method, args.seed
        )
        layout = Layout.from_assignment(assignment, args.shards)

    masks = neighbor_masks(layout, ground_truth, args.top_k)
    centroid_distances = np.linalg.norm(
        queries[:, None, :] - centroids[None, :, :], axis=2
    )
    centroid_order = np.argsort(centroid_distances, axis=1, kind="stable")

    report = analyze(
        masks, centroid_order, args.top_k, args.recall_targets, args.tail_quantile
    )
    report |= {
        "dataset": Path(args.hdf5).name,
        "base_vectors": int(train.shape[0]),
        "shards": layout.shards,
        "layout_method": args.label or args.method,
        "expansion_ratio": layout.copies / layout.points,
        "normalized": args.normalize,
        "seed": args.seed,
        "layout_balance": balance,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fanout = report["required_fanout"]
    print(
        f"{report['dataset']} P={report['shards']} {args.method}: "
        f"required fan-out mean={fanout['mean']:.2f} median={fanout['median']:.0f} "
        f"p95={fanout['p95']:.0f} max={fanout['max']:.0f} "
        f"| balance max/mean={balance['max_over_mean']:.2f}"
    )
    for target in report["orderings"]["oracle"]["targets"]:
        oracle = report["orderings"]["oracle"]["targets"][target]
        centroid = report["orderings"]["centroid"]["targets"][target]
        print(
            f"  R={target}  oracle: fixed={oracle['fixed_budget']:2d} "
            f"adaptive={oracle['adaptive_mean_shards']:5.2f} "
            f"H_adapt={oracle['headroom_adaptive']:4.2f}x"
            f"   centroid: fixed={centroid['fixed_budget']:2d} "
            f"adaptive={centroid['adaptive_mean_shards']:5.2f} "
            f"H_adapt={centroid['headroom_adaptive']:4.2f}x"
            f"   H_order={report['headroom_ordering'][target]:5.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
