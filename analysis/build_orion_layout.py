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

    refined = time.perf_counter()
    routing = experiment.build_original_routing_state(
        train,
        upper_indices,
        point_to_l1s,
        int(args.shards),
        int(args.kmeans_iters),
        int(args.kmeans_seed),
        int(args.topology_iters),
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

    shard_count = int(routing.num_shards)
    copies_per_shard = np.bincount(indices, minlength=shard_count).astype(np.int64)
    metadata = {
        "dataset": Path(args.hdf5_path).name,
        "dataset_sha256": dataset_record.get("sha256"),
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
