"""Pre-build Orion's routed search requests, one per query, outside any timer.

The measurement harness (measure.py --mode routed) must not do routing inside the
timed path, or it re-creates the client-bound artifact STAGE0.md documents. So
all routing happens here, offline, and the output is a JSONL file of fully
serialized request bodies -- exactly what the timed path replays as opaque bytes.

Each body is Orion's real navigation-routed search: the query's nearest upper-tier
(L1) nodes are looked up in the upper graph, the shards owning those nodes become
the probed set, each probed shard carries the L1 points that landed on it as HNSW
entry points, and its ef scales with how many landed (ef = base_ef + factor*hits).
This is the same rule as tools/qdrant_two_level_routing_experiment.py's
`route_upper_labels_to_shard_eps` + `shard_efs_from_routed_eps` + `search_request`,
reused directly rather than reimplemented so the encoding (including replicated
copy ids) matches the deployed collection bit for bit.

Two routers are available so all comparison arms come from one code path:
  --router navigation  Orion's upper-graph router (entry points + scaled ef). Knob
                       is --upper-k (top L1 hits to route on). This is Orion proper.
  --router centroid    a k-means baseline: rank shards by query-to-centroid
                       distance and probe the top --nprobe (uniform ef, no entry
                       points). Feed it a k-means layout to get the honest baseline.

The knob (--upper-k or --nprobe) is the deployed routing budget, like nprobe.
Sweep it across runs to trace a recall-vs-QPS frontier; a smaller budget probes
fewer shards (lower fan-out, higher throughput) at some recall cost. Compare arms
at matched recall.

Shard membership of the L1 points comes from either:
  --layout-file       an offline layout .npz (analysis/build_orion_layout.py); fast,
                      but YOU must have built the collection from this same layout.
  --from-collection   scroll the live collection to recover it; authoritative, zero
                      drift from what is actually stored. Preferred before a real run.

Needs the system libstdc++ ahead of conda's for hnswlib:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 build_routed_requests.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import qdrant_two_level_routing_experiment as experiment  # noqa: E402

ANALYSIS = REPO_ROOT / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from fanout_headroom import Layout, shard_centroids  # noqa: E402
from load_skew import centroid_order  # noqa: E402


def layout_point_to_shards(layout: Layout) -> list[list[int]]:
    """Materialize the CSR layout as point -> [shard ids] for the router."""
    indptr = layout.indptr
    indices = layout.indices.astype(int, copy=False)
    return [indices[indptr[p] : indptr[p + 1]].tolist() for p in range(layout.points)]


def build_upper_labels(
    train: np.ndarray,
    queries: np.ndarray,
    args: argparse.Namespace,
    distance_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (upper_labels, upper_indices) reproducing the deployed upper graph."""
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
    return upper_labels.astype(np.int64, copy=False), upper_indices


def percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, help="dataset with train/test")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--vector-distance", choices=("cosine", "euclid", "l2"), required=True)
    parser.add_argument("--out", required=True, help="requests.jsonl to write")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--router",
        choices=("navigation", "centroid", "broadcast"),
        default="navigation",
        help="navigation = Orion's upper-graph router (entry points + scaled ef); "
        "centroid = rank shards by query-to-centroid distance and probe the top "
        "--nprobe (a k-means router baseline, uniform ef, no entry points); "
        "broadcast = probe all P shards at uniform ef (fan-out = P), the reference. "
        "broadcast carries the source_id payload so recall is correct on replicated "
        "collections, which plain --mode broadcast does not.",
    )
    parser.add_argument(
        "--upper-k",
        type=int,
        default=0,
        help="navigation router: top L1 hits to route on (deployed knob; sweep this)",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=0,
        help="centroid router: number of nearest shards to probe (sweep this)",
    )
    parser.add_argument("--base-ef", type=int, default=20)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument(
        "--hnsw-ef",
        type=int,
        default=0,
        help="centroid router: uniform per-shard ef (0 = use --base-ef)",
    )
    parser.add_argument("--queries-limit", type=int, default=0, help="0 = all queries")

    # Upper-graph parameters. These MUST match the collection's build.
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-graph-seed", type=int, default=100)
    parser.add_argument("--upper-query-k", type=int, default=200)

    # Shard membership source (exactly one).
    parser.add_argument("--layout-file", help="offline layout .npz")
    parser.add_argument("--from-collection", action="store_true", help="recover from live collection")
    parser.add_argument("--base-url", default="http://127.0.0.1:6333")
    parser.add_argument("--num-shards", type=int, default=0, help="required with --from-collection")

    # Copy-id encoding. MUST match the value used when the collection was uploaded.
    parser.add_argument(
        "--source-id-dedup-block-size",
        type=int,
        default=-1,
        help="-1 = num_points+1 (tool default); 0 = disable encoding (points stored at raw ids)",
    )
    args = parser.parse_args()

    if args.router == "navigation":
        if bool(args.layout_file) == bool(args.from_collection):
            raise SystemExit(
                "navigation router needs exactly one of --layout-file / --from-collection"
            )
    elif args.router == "centroid":
        if not args.layout_file:
            raise SystemExit("centroid router requires --layout-file (needs centroids)")
    else:  # broadcast
        if not args.layout_file and args.num_shards <= 0:
            raise SystemExit("broadcast router requires --num-shards or --layout-file")

    with h5py.File(args.hdf5, "r") as handle:
        train = np.asarray(handle["train"], dtype=np.float32)
        queries = np.asarray(handle["test"], dtype=np.float32)
    if args.queries_limit and args.queries_limit < len(queries):
        queries = queries[: args.queries_limit]

    num_points = int(train.shape[0])
    if args.source_id_dedup_block_size == -1:
        block_size: int | None = num_points + 1
    elif args.source_id_dedup_block_size == 0:
        block_size = None
    else:
        block_size = int(args.source_id_dedup_block_size)
    use_payload_source_id = block_size is not None

    distance_config = experiment.vector_distance_config(args.vector_distance)

    # Shard membership / geometry source, shared by both routers.
    if args.layout_file:
        layout: Layout | None = Layout.load(args.layout_file)
        if layout.points != num_points:
            raise SystemExit(
                f"layout covers {layout.points} points, dataset has {num_points}"
            )
        num_shards = layout.shards
    else:
        if args.num_shards <= 0:
            raise SystemExit("--from-collection requires --num-shards")
        num_shards = int(args.num_shards)
        layout = None

    base_ef = int(args.base_ef)
    shard_key_for_id = experiment.shard_key_for_id

    if args.router == "navigation":
        if int(args.upper_k) <= 0:
            raise SystemExit("navigation router requires --upper-k > 0")
        upper_labels, upper_indices = build_upper_labels(train, queries, args, distance_config)
        if layout is not None:
            point_to_shards = layout_point_to_shards(layout)
        else:
            point_to_shards = experiment.recover_upper_point_to_shards_from_collection(
                args.base_url, args.collection, upper_indices, num_points, num_shards
            )
        knob_value = min(int(args.upper_k), upper_labels.shape[1])

        def build_body(q: int) -> tuple[bytes, int, int, int]:
            shard_to_eps = experiment.route_upper_labels_to_shard_eps(
                upper_labels[q][:knob_value], point_to_shards
            )
            shard_keys, ef_values = experiment.shard_efs_from_routed_eps(
                shard_to_eps, num_shards, base_ef, int(args.factor)
            )
            if not shard_keys:
                # No L1 hit landed anywhere (degenerate); probe shard 0 so the query
                # still returns something and stays aligned by offset.
                shard_keys = [shard_key_for_id(0)]
                ef_values = [base_ef]
                shard_to_eps = {0: []}
            body = experiment.search_request(
                queries[q].tolist(),
                int(args.top_k),
                int(max(ef_values)),
                shard_keys,
                use_payload_source_id,
                hnsw_entry_points_by_shard={
                    shard_key_for_id(sid): eps for sid, eps in shard_to_eps.items()
                },
                hnsw_ef_by_shard=dict(zip(shard_keys, ef_values)),
                source_id_dedup_block_size=block_size,
            )
            ep_count = sum(len(eps) for eps in shard_to_eps.values())
            return orjson.dumps(body), len(shard_keys), int(sum(ef_values)), ep_count

    elif args.router == "centroid":  # k-means baseline
        if layout is None:
            raise SystemExit("centroid router requires --layout-file (needs centroids)")
        if int(args.nprobe) <= 0:
            raise SystemExit("centroid router requires --nprobe > 0")
        knob_value = min(int(args.nprobe), num_shards)
        centroids = shard_centroids(layout, train)
        order = centroid_order(queries, centroids)  # ascending query-centroid distance
        uniform_ef = int(args.hnsw_ef) if int(args.hnsw_ef) > 0 else base_ef

        def build_body(q: int) -> tuple[bytes, int, int, int]:
            shard_keys = [shard_key_for_id(int(s)) for s in order[q][:knob_value]]
            body = experiment.search_request(
                queries[q].tolist(),
                int(args.top_k),
                uniform_ef,
                shard_keys,
                use_payload_source_id,
                hnsw_ef_by_shard={key: uniform_ef for key in shard_keys},
                source_id_dedup_block_size=block_size,
            )
            return orjson.dumps(body), len(shard_keys), uniform_ef * len(shard_keys), 0

    else:  # broadcast (fan-out = P reference)
        all_keys = [shard_key_for_id(s) for s in range(num_shards)]
        uniform_ef = int(args.hnsw_ef) if int(args.hnsw_ef) > 0 else base_ef
        knob_value = uniform_ef  # the swept variable for broadcast is ef, not fan-out
        ef_by_all = {key: uniform_ef for key in all_keys}

        def build_body(q: int) -> tuple[bytes, int, int, int]:
            body = experiment.search_request(
                queries[q].tolist(),
                int(args.top_k),
                uniform_ef,
                all_keys,
                use_payload_source_id,
                hnsw_ef_by_shard=ef_by_all,
                source_id_dedup_block_size=block_size,
            )
            return orjson.dumps(body), num_shards, uniform_ef * num_shards, 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fanout = np.zeros(len(queries), dtype=np.int32)
    ef_sum = np.zeros(len(queries), dtype=np.int64)
    ep_sum = np.zeros(len(queries), dtype=np.int64)

    with out_path.open("wb") as sink:
        for q in range(len(queries)):
            body_bytes, fo, efs, eps = build_body(q)
            sink.write(body_bytes)
            sink.write(b"\n")
            fanout[q] = fo
            ef_sum[q] = efs
            ep_sum[q] = eps

    knob_name = {"navigation": "upper_k", "centroid": "nprobe", "broadcast": "hnsw_ef"}[
        args.router
    ]
    stats = {
        "dataset": Path(args.hdf5).name,
        "collection": args.collection,
        "router": args.router,
        "queries": int(len(queries)),
        "num_shards": num_shards,
        knob_name: knob_value,
        "base_ef": args.base_ef,
        "factor": args.factor,
        "source_id_dedup_block_size": block_size,
        "membership_source": args.layout_file or f"collection:{args.collection}",
        "fanout": percentiles(fanout),
        "ef_sum": percentiles(ef_sum),
        "entry_points": percentiles(ep_sum),
        "requests_file": str(out_path.resolve()),
    }
    Path(str(out_path) + ".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        f"{stats['dataset']} collection={args.collection} router={args.router} "
        f"P={num_shards} {knob_name}={knob_value}: wrote {len(queries)} bodies -> {out_path}"
    )
    print(
        f"  fan-out mean={stats['fanout']['mean']:.2f} p95={stats['fanout']['p95']:.1f} "
        f"max={stats['fanout']['max']:.0f} of P={num_shards}; "
        f"total ef mean={stats['ef_sum']['mean']:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
