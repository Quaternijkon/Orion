"""Materialize an Orion navigation-derived layout for offline fan-out analysis.

This calls the offline pipeline in ``tools/qdrant_two_level_routing_experiment.py``
directly and keeps only the point-to-shard assignment, skipping the Rust
production-artifact build and the import bundle that the CLI would also produce.

The upper graph here is built with ``hnswlib``. Per ``agent.md`` that offline
graph is *not* the canonical Orion path, which builds the production upper graph
with Qdrant's Rust ``GraphLayersBuilder`` and then derives attachments from that
same graph. The layout produced here is therefore the legacy dual-graph variant.
It is used deliberately: the question this analysis asks is whether a
navigation-derived layout family has different fan-out structure than a
geometric one, which does not depend on which library built the sampled upper
graph. Any conclusion must still be labelled as coming from this variant.

Assignments are stored in CSR form because multi-assignment lets one point
occupy several shards.

Requires the system libstdc++ to be preloaded, since conda ships an older one
than ``hnswlib`` was built against:

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 build_orion_layout.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import orion_native_layout as native_layout  # noqa: E402
from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402

DETERMINISTIC_THREADS = native_layout.DETERMINISTIC_ATTACHMENT_BUILD_THREADS


def compute_initial_seed(
    method: str,
    train: np.ndarray,
    upper_indices: np.ndarray,
    up_tier_weights: np.ndarray,
    num_shards: int,
    kmeans_iters: int,
    kmeans_seed: int,
    max_load_ratio: float,
    init_seed: int,
    spectral_knn: int,
) -> list[int]:
    """Initial L1->shard assignment (length len(train), -1 for non-upper nodes).

    'kmeans' reproduces the shipped balanced-k-means seed exactly (same function,
    same args), so passing it back through build_original_routing_state is a
    no-op relative to the default path. 'random' and 'spectral' are the ablation
    alternatives, both balanced by construction where possible.
    """
    n = len(train)
    up = upper_indices.tolist()
    if method == "kmeans":
        return experiment.initial_l1_shards_by_balanced_kmeans(
            train, upper_indices, up_tier_weights, num_shards,
            kmeans_iters, kmeans_seed, max_load_ratio=max_load_ratio,
        )
    seed = [-1] * n
    if method == "random":
        rng = np.random.default_rng(init_seed)
        perm = rng.permutation(len(up))
        for rank, local in enumerate(perm.tolist()):
            seed[up[local]] = rank % num_shards  # exactly balanced by node count
        return seed
    if method == "spectral":
        from sklearn.neighbors import kneighbors_graph
        from sklearn.cluster import SpectralClustering

        features = np.asarray(train[upper_indices], dtype=np.float32)
        affinity = kneighbors_graph(
            features, n_neighbors=spectral_knn, mode="connectivity",
            include_self=False,
        )
        affinity = 0.5 * (affinity + affinity.T)  # symmetrize the kNN graph
        labels = SpectralClustering(
            n_clusters=num_shards, affinity="precomputed",
            assign_labels="kmeans", random_state=init_seed,
        ).fit_predict(affinity)
        for local, label in enumerate(labels.tolist()):
            seed[up[local]] = int(label)
        return seed
    raise ValueError(f"unknown init method {method!r}")


def attachment_cut_fraction(
    assign: list[int], upper_nodes: list[int], point_to_l1s: list[list[int]]
) -> float:
    """Fraction of attachment (L1, entry-point) pairs that cross a shard boundary.

    This is exactly the objective the label-propagation refinement minimizes, so
    it is the intrinsic quality of a partition independent of any downstream
    serving. Lower is better (neighbors kept together)."""
    cross = total = 0
    for node in upper_nodes:
        home = assign[node]
        for entry in point_to_l1s[node]:
            total += 1
            if assign[int(entry)] != home:
                cross += 1
    return cross / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--out", required=True, help="npz destination for the layout")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument(
        "--vector-distance", choices=("cosine", "euclid", "l2"), required=True
    )
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--attachment-search-ef", type=int, default=100)
    parser.add_argument("--upper-graph-seed", type=int, default=100)
    parser.add_argument("--k-overlap", type=int, default=10)
    parser.add_argument("--upper-build-batch-size", type=int, default=10_000)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--kmeans-seed", type=int, default=1)
    parser.add_argument("--topology-iters", type=int, default=50)
    parser.add_argument("--enable-fission", action="store_true")
    parser.add_argument("--disable-multi-assign", action="store_true")
    parser.add_argument("--disable-topology-refinement", action="store_true")
    parser.add_argument("--multi-assign-min-max-vote", type=int, default=2)
    parser.add_argument("--multi-assign-vote-delta", type=int, default=0)
    parser.add_argument("--multi-assign-max-shards", type=int, default=0)
    parser.add_argument(
        "--balance-mode",
        choices=("none", "capacity_constrained", "post_layout_capacity_constrained"),
        default="capacity_constrained",
    )
    parser.add_argument("--balance-min-load-ratio", type=float, default=0.99)
    parser.add_argument("--balance-max-load-ratio", type=float, default=1.01)
    parser.add_argument("--balance-max-passes", type=int, default=8)
    parser.add_argument(
        "--balance-max-vote-loss",
        type=int,
        default=3,
        help="Shared vote-loss bound for L1 and L0; matches the CLI default.",
    )
    parser.add_argument(
        "--init",
        choices=("kmeans", "random", "spectral"),
        default="kmeans",
        help=(
            "Initial L1->shard seed fed to topology refinement. 'kmeans' is the "
            "shipped balanced-k-means seed; 'random' is a balanced round-robin "
            "cold start; 'spectral' is a normalized-cut partition of the upper "
            "kNN graph (graph-native seed). Ablation only."
        ),
    )
    parser.add_argument("--init-seed", type=int, default=1)
    parser.add_argument("--spectral-knn", type=int, default=15)
    parser.add_argument(
        "--init-file",
        help=(
            "npy of a precomputed initial L1->shard seed (length N, -1 for "
            "non-upper nodes). Overrides --init; used for seeds that cannot be "
            "computed in-process under LD_PRELOAD (e.g. spectral via sklearn)."
        ),
    )
    args = parser.parse_args()

    started = time.perf_counter()
    train, dataset_record = native_layout.load_train_vectors(
        Path(args.hdf5_path).expanduser().resolve(), None, args.vector_distance
    )
    distance_config = experiment.vector_distance_config(args.vector_distance)
    upper_indices = experiment.global_upper_indices(
        len(train), int(args.sample_denominator), int(args.upper_sample_seed)
    )
    upper_built = time.perf_counter()
    upper_index = experiment.build_upper_index(
        train[upper_indices],
        upper_indices.astype(np.int64, copy=False),
        int(train.shape[1]),
        int(args.upper_m),
        int(args.upper_ef_construction),
        int(args.attachment_search_ef),
        distance_config["hnsw_space"],
        random_seed=int(args.upper_graph_seed),
        construction_threads=DETERMINISTIC_THREADS,
    )
    print(
        f"upper graph: {len(upper_indices)} nodes "
        f"in {time.perf_counter() - upper_built:.1f}s",
        flush=True,
    )

    attached = time.perf_counter()
    point_to_l1s = experiment.compute_point_to_l1s(
        upper_index,
        train,
        int(args.k_overlap),
        int(args.upper_build_batch_size),
    )
    print(
        f"attachments: {len(point_to_l1s)} points "
        f"in {time.perf_counter() - attached:.1f}s",
        flush=True,
    )

    # Ablation seed: build the initial L1->shard assignment for the chosen init
    # and pass it into the refinement so kmeans/random/spectral share every
    # downstream stage. up_tier_weights is recomputed exactly as the refiner does.
    nearest_l1 = np.asarray([l1s[0] for l1s in point_to_l1s], dtype=np.int64)
    l1_weights_map = np.bincount(nearest_l1, minlength=len(train))
    up_tier_weights = l1_weights_map[upper_indices].astype(np.int64, copy=False)
    init_max_load_ratio = (
        float(args.balance_max_load_ratio)
        if str(args.balance_mode) == "capacity_constrained"
        else 1.5
    )
    seeded = time.perf_counter()
    if args.init_file:
        loaded = np.load(args.init_file)
        if loaded.shape[0] != len(train):
            raise SystemExit(
                f"--init-file has {loaded.shape[0]} entries, expected {len(train)}"
            )
        initial_l1_to_shard = [int(x) for x in loaded.tolist()]
        init_label = str(args.init) if args.init != "kmeans" else "file"
    else:
        initial_l1_to_shard = compute_initial_seed(
            str(args.init), train, upper_indices, up_tier_weights, int(args.shards),
            int(args.kmeans_iters), int(args.kmeans_seed), init_max_load_ratio,
            int(args.init_seed), int(args.spectral_knn),
        )
        init_label = str(args.init)
    print(f"init seed: {init_label} in {time.perf_counter() - seeded:.1f}s", flush=True)

    refined = time.perf_counter()
    routing = experiment.build_original_routing_state(
        train,
        upper_indices,
        point_to_l1s,
        int(args.shards),
        int(args.kmeans_iters),
        int(args.kmeans_seed),
        int(args.topology_iters),
        initial_l1_to_shard=initial_l1_to_shard,
        use_multi_assign=not bool(args.disable_multi_assign),
        enable_fission=bool(args.enable_fission),
        multi_assign_min_max_vote=int(args.multi_assign_min_max_vote),
        multi_assign_vote_delta=int(args.multi_assign_vote_delta),
        multi_assign_max_shards=int(args.multi_assign_max_shards),
        enable_topology_refinement=not bool(args.disable_topology_refinement),
        balance_mode=str(args.balance_mode),
        balance_min_load_ratio=float(args.balance_min_load_ratio),
        balance_max_load_ratio=float(args.balance_max_load_ratio),
        balance_max_passes=int(args.balance_max_passes),
        balance_max_vote_loss=int(args.balance_max_vote_loss),
        balance_l1_max_vote_loss=None,
        balance_l0_max_vote_loss=None,
    )
    print(f"layout: refined in {time.perf_counter() - refined:.1f}s", flush=True)

    lengths = np.fromiter(
        (len(shards) for shards in routing.point_to_shards),
        dtype=np.int64,
        count=len(routing.point_to_shards),
    )
    indptr = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = np.fromiter(
        (shard for shards in routing.point_to_shards for shard in shards),
        dtype=np.int32,
        count=int(lengths.sum()),
    )

    # Ablation diagnostics: how far the refinement moved the seed, and the
    # intrinsic partition quality (attachment cut) before vs after refinement.
    final_l1 = list(routing.l1_to_shard)
    upper_nodes = upper_indices.tolist()
    init_churn = sum(
        1 for node in upper_nodes if initial_l1_to_shard[node] != final_l1[node]
    ) / max(len(upper_nodes), 1)
    cut_initial = attachment_cut_fraction(
        initial_l1_to_shard, upper_nodes, point_to_l1s
    )
    cut_final = attachment_cut_fraction(final_l1, upper_nodes, point_to_l1s)

    # Refinement convergence: how many passes ran and the per-pass move/cut curve
    # (cut present only when ORION_LOG_REFINE_CUT=1 was set).
    l1_topo = (routing.balance_diagnostics or {}).get("l1_topology") or {}
    topo_iterations = int(getattr(routing, "topology_iterations", 0))
    per_pass = l1_topo.get("per_pass") or []
    l1_move_counts = l1_topo.get("move_counts") or {}
    if per_pass:
        print("refinement per pass (pass: changed [cut]):", flush=True)
        for row in per_pass:
            cut_str = f" cut={row['cut']:.4f}" if "cut" in row else ""
            print(f"  pass {int(row['pass']):>3}: changed={int(row['changed']):>7}{cut_str}", flush=True)
    print(
        f"refinement: {topo_iterations} passes; move_counts={l1_move_counts}",
        flush=True,
    )

    shard_count = int(routing.num_shards)
    copies_per_shard = np.bincount(indices, minlength=shard_count).astype(np.int64)
    metadata = {
        "dataset": Path(args.hdf5_path).name,
        "dataset_sha256": dataset_record.get("sha256"),
        "init_method": init_label,
        "init_churn_fraction": float(init_churn),
        "attachment_cut_initial": float(cut_initial),
        "attachment_cut_final": float(cut_final),
        "topology_iterations": topo_iterations,
        "refinement_per_pass": per_pass,
        "l1_move_counts": {str(k): int(v) for k, v in l1_move_counts.items()},
        "requested_shards": int(args.shards),
        "effective_shards": shard_count,
        "logical_points": int(lengths.size),
        "physical_copies": int(lengths.sum()),
        "expansion_ratio": float(lengths.sum() / lengths.size),
        "fission_enabled": bool(args.enable_fission),
        "multi_assign_enabled": not bool(args.disable_multi_assign),
        "topology_refinement_enabled": not bool(args.disable_topology_refinement),
        "balance_mode": args.balance_mode,
        "upper_graph": "hnswlib (legacy dual-graph variant, see module docstring)",
        "copies_per_shard_max_over_mean": float(
            copies_per_shard.max() / copies_per_shard.mean()
        ),
        "build_seconds": float(time.perf_counter() - started),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        indptr=indptr,
        indices=indices,
        shard_count=np.int32(shard_count),
        metadata=json.dumps(metadata),
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
