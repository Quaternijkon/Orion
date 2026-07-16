#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class PartitionState:
    name: str
    family: str
    description: str
    initial_num_shards: int
    effective_num_shards: int
    l1_to_shard: list[int] | None
    point_to_shards: list[list[int]]
    total_assigned: int
    index_expansion_ratio: float
    topology_iterations: int | None
    fission_event_count: int
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Claim A offline partition oracle analysis. Compare random, "
            "balanced-KMeans, topology-converged, load-recalibrated, and full "
            "Method4/fission partitions without executing lower HNSW search."
        )
    )
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--control-num-shards", type=int, default=46)
    parser.add_argument("--full-initial-num-shards", type=int, default=31)
    parser.add_argument("--upper-ks", type=int, nargs="+", default=[80, 120, 160])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-query-count", type=int, default=3000)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=400)
    parser.add_argument("--k-overlap", type=int, default=10)
    parser.add_argument("--upper-build-batch-size", type=int, default=10000)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--kmeans-rand-seed", type=int, default=1)
    parser.add_argument("--topology-iters", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--disable-multi-assign", action="store_true")
    parser.add_argument("--orion-multi-assign-min-max-vote", type=int, default=2)
    parser.add_argument("--orion-multi-assign-vote-delta", type=int, default=0)
    parser.add_argument("--orion-multi-assign-max-shards", type=int, default=0)
    parser.add_argument("--output-root", default="results/method4_claim_a_partition_oracle_20260704")
    parser.add_argument("--qdrant-tool", default="tools/qdrant_two_level_routing_experiment.py")
    return parser.parse_args()


def load_qdrant_tool(path: str | Path) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location("qdrant_two_level_routing_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def min_shards_for_gt(gt_shard_sets: list[set[int]]) -> int:
    if not gt_shard_sets:
        return 0
    full_mask = (1 << len(gt_shard_sets)) - 1
    shard_masks: dict[int, int] = {}
    for item_idx, shard_set in enumerate(gt_shard_sets):
        for shard_id in shard_set:
            shard_masks[int(shard_id)] = shard_masks.get(int(shard_id), 0) | (1 << item_idx)
    if not shard_masks:
        return 0
    inf = len(gt_shard_sets) + 1
    dp = [inf] * (1 << len(gt_shard_sets))
    dp[0] = 0
    for shard_mask in shard_masks.values():
        for mask in range(full_mask + 1):
            next_mask = mask | shard_mask
            if dp[mask] + 1 < dp[next_mask]:
                dp[next_mask] = dp[mask] + 1
    return int(dp[full_mask] if dp[full_mask] < inf else 0)


def shannon_entropy_from_counts(counts: dict[int, int]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = float(count) / float(total)
        entropy -= probability * math.log(probability)
    return entropy


def route_shards_for_labels(labels: np.ndarray, point_to_shards: list[list[int]], upper_k: int) -> set[int]:
    routed: set[int] = set()
    for label in labels[:upper_k].tolist():
        point_id = int(label)
        if 0 <= point_id < len(point_to_shards):
            routed.update(int(shard_id) for shard_id in point_to_shards[point_id])
    return routed


def analyze_partition_oracle(
    partition: str,
    labels: np.ndarray,
    neighbors: np.ndarray,
    point_to_shards: list[list[int]],
    upper_ks: list[int],
    index_expansion_ratio: float,
    topology_edge_cut: float | None = None,
    route_all_shards: bool = False,
    all_shard_count: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_count = int(len(neighbors))
    top_k = int(neighbors.shape[1]) if neighbors.ndim == 2 else 0
    all_shards = set(range(int(all_shard_count or 0)))

    for upper_k in upper_ks:
        covered_items = 0
        total_items = 0
        covered_copies = 0
        total_copies = 0
        routed_shard_counts: list[float] = []
        min_cover_counts: list[float] = []
        entropy_values: list[float] = []
        waste_shards = 0
        routed_shard_observations = 0
        any_covered_queries = 0
        all_covered_queries = 0

        for query_idx in range(query_count):
            if route_all_shards:
                routed_shards = set(all_shards)
            else:
                routed_shards = route_shards_for_labels(labels[query_idx], point_to_shards, int(upper_k))

            routed_shard_counts.append(float(len(routed_shards)))
            routed_shard_observations += len(routed_shards)

            gt_shard_sets: list[set[int]] = []
            gt_copy_counts: dict[int, int] = {}
            query_covered_items = 0
            for gt_id in neighbors[query_idx].tolist():
                gt_shards = {
                    int(shard_id)
                    for shard_id in point_to_shards[int(gt_id)]
                    if int(shard_id) >= 0
                }
                gt_shard_sets.append(gt_shards)
                total_items += 1
                total_copies += len(gt_shards)
                copy_hits = len(gt_shards & routed_shards)
                covered_copies += copy_hits
                if copy_hits > 0:
                    covered_items += 1
                    query_covered_items += 1
                for shard_id in gt_shards:
                    gt_copy_counts[shard_id] = gt_copy_counts.get(shard_id, 0) + 1

            gt_union = set().union(*gt_shard_sets) if gt_shard_sets else set()
            waste_shards += len(routed_shards - gt_union)
            min_cover_counts.append(float(min_shards_for_gt(gt_shard_sets)))
            entropy_values.append(shannon_entropy_from_counts(gt_copy_counts))
            if query_covered_items > 0:
                any_covered_queries += 1
            if query_covered_items == top_k:
                all_covered_queries += 1

        coverage = covered_items / total_items if total_items else 0.0
        copy_coverage = covered_copies / total_copies if total_copies else 0.0
        rows.append(
            {
                "partition": partition,
                "upper_k": int(upper_k),
                "query_count": query_count,
                "top_k": top_k,
                "avg_routed_shards": sum(routed_shard_counts) / query_count if query_count else 0.0,
                "p95_routed_shards": percentile(routed_shard_counts, 95),
                f"oracle_gt_coverage_at_{top_k}": coverage,
                f"oracle_gt_miss_at_{top_k}": 1.0 - coverage,
                f"oracle_gt_copy_coverage_at_{top_k}": copy_coverage,
                "query_any_gt_covered_rate": any_covered_queries / query_count if query_count else 0.0,
                "query_all_gt_covered_rate": all_covered_queries / query_count if query_count else 0.0,
                f"avg_min_shards_for_gt_at_{top_k}": sum(min_cover_counts) / query_count if query_count else 0.0,
                f"p95_min_shards_for_gt_at_{top_k}": percentile(min_cover_counts, 95),
                f"avg_gt_shard_entropy_at_{top_k}": sum(entropy_values) / query_count if query_count else 0.0,
                f"p95_gt_shard_entropy_at_{top_k}": percentile(entropy_values, 95),
                "routed_waste_ratio": waste_shards / routed_shard_observations if routed_shard_observations else 0.0,
                "topology_edge_cut": topology_edge_cut if topology_edge_cut is not None else "",
                "index_expansion_ratio": float(index_expansion_ratio),
            }
        )
    return rows


def topology_edge_cut(
    point_to_l1s: list[list[int]],
    upper_indices: np.ndarray,
    l1_to_shard: list[int] | None,
) -> float | None:
    if l1_to_shard is None:
        return None
    cut_edges = 0
    total_edges = 0
    for l1_idx in upper_indices.tolist():
        source = int(l1_idx)
        if source >= len(l1_to_shard):
            continue
        source_shard = int(l1_to_shard[source])
        if source_shard < 0:
            continue
        for target in point_to_l1s[source]:
            target = int(target)
            if target >= len(l1_to_shard):
                continue
            target_shard = int(l1_to_shard[target])
            if target_shard < 0:
                continue
            total_edges += 1
            cut_edges += int(source_shard != target_shard)
    return cut_edges / total_edges if total_edges else None


def weighted_random_l1_shards(
    num_points: int,
    upper_indices: np.ndarray,
    up_tier_weights: np.ndarray,
    num_shards: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    l1_to_shard = [-1] * int(num_points)
    shard_loads = np.zeros(int(num_shards), dtype=np.int64)
    order = rng.permutation(len(upper_indices))
    for local_idx in order.tolist():
        l1_idx = int(upper_indices[int(local_idx)])
        shard_id = int(np.argmin(shard_loads))
        l1_to_shard[l1_idx] = shard_id
        shard_loads[shard_id] += int(up_tier_weights[int(local_idx)])
    return l1_to_shard


def build_partition_state(
    q2l: Any,
    name: str,
    family: str,
    description: str,
    l1_to_shard: list[int] | None,
    point_to_l1s: list[list[int]],
    num_points: int,
    initial_num_shards: int,
    effective_num_shards: int,
    use_multi_assign: bool,
    min_max_vote: int,
    vote_delta: int,
    max_shards: int,
    topology_iterations: int | None,
    fission_event_count: int,
    note: str,
    point_to_shards_override: list[list[int]] | None = None,
) -> PartitionState:
    if point_to_shards_override is None:
        if l1_to_shard is None:
            raise ValueError("l1_to_shard is required without point_to_shards_override")
        _primary, point_to_shards = q2l.assign_points_by_l1_vote(
            point_to_l1s,
            l1_to_shard,
            effective_num_shards,
            use_multi_assign,
            min_max_vote,
            vote_delta,
            max_shards,
        )
    else:
        point_to_shards = point_to_shards_override
    total_assigned = sum(len(shards) for shards in point_to_shards)
    return PartitionState(
        name=name,
        family=family,
        description=description,
        initial_num_shards=initial_num_shards,
        effective_num_shards=effective_num_shards,
        l1_to_shard=l1_to_shard,
        point_to_shards=point_to_shards,
        total_assigned=total_assigned,
        index_expansion_ratio=total_assigned / num_points if num_points else 0.0,
        topology_iterations=topology_iterations,
        fission_event_count=fission_event_count,
        note=note,
    )


def load_data(args: argparse.Namespace, q2l: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    with h5py.File(args.hdf5_path, "r") as handle:
        train = handle["train"][:].astype(np.float32, copy=True)
        queries = handle["test"][: args.eval_query_count].astype(np.float32, copy=True)
        neighbors = handle["neighbors"][: args.eval_query_count, : args.top_k].astype(np.int64, copy=True)
    train = q2l.normalize_rows(train)
    queries = q2l.normalize_rows(queries)
    upper_indices = q2l.global_upper_indices(len(train), args.sample_denominator, args.upper_sample_seed)
    return train, queries, neighbors, upper_indices, int(train.shape[1])


def build_partition_states(
    args: argparse.Namespace,
    q2l: Any,
    train: np.ndarray,
    upper_indices: np.ndarray,
    point_to_l1s: list[list[int]],
) -> tuple[list[PartitionState], dict[str, Any]]:
    nearest_l1 = np.asarray([l1s[0] for l1s in point_to_l1s], dtype=np.int64)
    l1_weights_map = np.bincount(nearest_l1, minlength=len(train))
    up_tier_weights = l1_weights_map[upper_indices].astype(np.int64, copy=False)
    use_multi_assign = not args.disable_multi_assign
    min_max_vote = args.orion_multi_assign_min_max_vote
    vote_delta = args.orion_multi_assign_vote_delta
    max_shards = args.orion_multi_assign_max_shards

    random_l1 = weighted_random_l1_shards(
        len(train),
        upper_indices,
        up_tier_weights,
        args.control_num_shards,
        args.random_seed,
    )
    kmeans_l1 = q2l.initial_l1_shards_by_balanced_kmeans(
        train,
        upper_indices,
        up_tier_weights,
        args.control_num_shards,
        args.kmeans_iters,
        args.kmeans_rand_seed,
    )
    topology_l1, topology_iters = q2l.converge_l1_topology(
        point_to_l1s,
        upper_indices,
        up_tier_weights,
        kmeans_l1,
        len(train),
        args.control_num_shards,
        args.topology_iters,
    )
    _recalibrated_weights_46 = q2l.recalibrate_l1_weights_by_voting(
        point_to_l1s,
        upper_indices,
        topology_l1,
        args.control_num_shards,
        use_multi_assign,
        min_max_vote,
        vote_delta,
        max_shards,
    )

    full_kmeans_l1 = q2l.initial_l1_shards_by_balanced_kmeans(
        train,
        upper_indices,
        up_tier_weights,
        args.full_initial_num_shards,
        args.kmeans_iters,
        args.kmeans_rand_seed,
    )
    full_topology_l1, full_topology_iters = q2l.converge_l1_topology(
        point_to_l1s,
        upper_indices,
        up_tier_weights,
        full_kmeans_l1,
        len(train),
        args.full_initial_num_shards,
        args.topology_iters,
    )
    full_recalibrated_weights = q2l.recalibrate_l1_weights_by_voting(
        point_to_l1s,
        upper_indices,
        full_topology_l1,
        args.full_initial_num_shards,
        use_multi_assign,
        min_max_vote,
        vote_delta,
        max_shards,
    )
    full_fission_l1, full_effective_shards, fission_events = q2l.apply_fission_simulator(
        train,
        upper_indices,
        full_recalibrated_weights,
        full_topology_l1,
        args.full_initial_num_shards,
        args.kmeans_iters,
        args.kmeans_rand_seed,
    )

    states = [
        build_partition_state(
            q2l,
            "random_balanced_46",
            "random_balanced",
            "Weighted random assignment of upper L1 nodes to 46 balanced shards.",
            random_l1,
            point_to_l1s,
            len(train),
            args.control_num_shards,
            args.control_num_shards,
            use_multi_assign,
            min_max_vote,
            vote_delta,
            max_shards,
            None,
            0,
            "Random balanced control; keeps the same final voting assignment policy as other controls.",
        ),
        build_partition_state(
            q2l,
            "balanced_kmeans_only_46",
            "balanced_kmeans_only",
            "Balanced KMeans placement of upper L1 nodes into 46 shards.",
            kmeans_l1,
            point_to_l1s,
            len(train),
            args.control_num_shards,
            args.control_num_shards,
            use_multi_assign,
            min_max_vote,
            vote_delta,
            max_shards,
            None,
            0,
            "Pure vector-clustering control.",
        ),
        build_partition_state(
            q2l,
            "kmeans_topology_46",
            "kmeans_topology_convergence",
            "Balanced KMeans followed by L1 topology convergence into 46 shards.",
            topology_l1,
            point_to_l1s,
            len(train),
            args.control_num_shards,
            args.control_num_shards,
            use_multi_assign,
            min_max_vote,
            vote_delta,
            max_shards,
            topology_iters,
            0,
            "Isolates topology convergence without fission.",
        ),
        build_partition_state(
            q2l,
            "kmeans_topology_load_recalibrated_46",
            "kmeans_topology_load_recalibration",
            "Topology-converged 46-shard map with voting load recalibration computed but no fission applied.",
            topology_l1,
            point_to_l1s,
            len(train),
            args.control_num_shards,
            args.control_num_shards,
            use_multi_assign,
            min_max_vote,
            vote_delta,
            max_shards,
            topology_iters,
            0,
            "In the current implementation, load recalibration affects routing only through fission, so this route map matches kmeans_topology_46.",
        ),
        build_partition_state(
            q2l,
            "full_method4_fission_31_to_46",
            "full_method4_with_fission",
            "Current Method4 path: 31 initial shards, topology convergence, voting load recalibration, and fission.",
            full_fission_l1,
            point_to_l1s,
            len(train),
            args.full_initial_num_shards,
            full_effective_shards,
            use_multi_assign,
            min_max_vote,
            vote_delta,
            max_shards,
            full_topology_iters,
            len(fission_events),
            "Uses current full Method4 reconstruction; expected effective shard count is 46 for default voting.",
        ),
    ]

    naive_point_to_shards = [[int(point_id % args.control_num_shards)] for point_id in range(len(train))]
    states.append(
        build_partition_state(
            q2l,
            "naive_all_shards_46",
            "naive_all_shards_reference",
            "Naive all-shards oracle reference with hash-style one-copy GT placement.",
            None,
            point_to_l1s,
            len(train),
            args.control_num_shards,
            args.control_num_shards,
            False,
            min_max_vote,
            vote_delta,
            max_shards,
            None,
            0,
            "Route set is forced to all 46 shards; topology_edge_cut is not applicable.",
            point_to_shards_override=naive_point_to_shards,
        )
    )

    metadata = {
        "use_multi_assign": use_multi_assign,
        "multi_assign_min_max_vote": min_max_vote,
        "multi_assign_vote_delta": vote_delta,
        "multi_assign_max_shards": max_shards,
        "full_fission_events": fission_events,
    }
    return states, metadata


def main() -> int:
    args = parse_args()
    q2l = load_qdrant_tool(args.qdrant_tool)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    train, queries, neighbors, upper_indices, dim = load_data(args, q2l)
    upper_index = q2l.build_upper_index(
        train[upper_indices],
        upper_indices.astype(np.int64, copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        args.upper_search_ef,
    )
    max_upper_k = max(int(value) for value in args.upper_ks)
    labels = q2l.compute_upper_labels(upper_index, queries, max_upper_k).astype(np.int64, copy=False)
    print("computing point_to_l1s", flush=True)
    point_to_l1s = q2l.compute_point_to_l1s(
        upper_index,
        train,
        args.k_overlap,
        args.upper_build_batch_size,
    )

    states, build_metadata = build_partition_states(args, q2l, train, upper_indices, point_to_l1s)
    oracle_rows: list[dict[str, Any]] = []
    build_rows: list[dict[str, Any]] = []
    for state in states:
        print(f"analyzing {state.name}", flush=True)
        edge_cut = topology_edge_cut(point_to_l1s, upper_indices, state.l1_to_shard)
        oracle_rows.extend(
            analyze_partition_oracle(
                state.name,
                labels,
                neighbors,
                state.point_to_shards,
                [int(value) for value in args.upper_ks],
                state.index_expansion_ratio,
                edge_cut,
                route_all_shards=(state.family == "naive_all_shards_reference"),
                all_shard_count=state.effective_num_shards,
            )
        )
        build_rows.append(
            {
                "partition": state.name,
                "family": state.family,
                "description": state.description,
                "initial_num_shards": state.initial_num_shards,
                "effective_num_shards": state.effective_num_shards,
                "total_assigned": state.total_assigned,
                "index_expansion_ratio": state.index_expansion_ratio,
                "topology_iterations": state.topology_iterations if state.topology_iterations is not None else "",
                "fission_event_count": state.fission_event_count,
                "topology_edge_cut": edge_cut if edge_cut is not None else "",
                "note": state.note,
            }
        )

    write_csv(output_dir / "partition_oracle_summary.csv", oracle_rows)
    write_csv(output_dir / "partition_build_summary.csv", build_rows)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_kind": "method4_claim_a_partition_oracle",
        "hdf5_path": args.hdf5_path,
        "num_points": int(len(train)),
        "control_num_shards": args.control_num_shards,
        "full_initial_num_shards": args.full_initial_num_shards,
        "upper_ks": [int(value) for value in args.upper_ks],
        "top_k": args.top_k,
        "eval_query_count": args.eval_query_count,
        "sample_denominator": args.sample_denominator,
        "upper_sample_seed": args.upper_sample_seed,
        "upper_count": int(len(upper_indices)),
        "upper_search_ef": args.upper_search_ef,
        "k_overlap": args.k_overlap,
        "topology_iters": args.topology_iters,
        "build_metadata": build_metadata,
        "oracle_summary_csv": str(output_dir / "partition_oracle_summary.csv"),
        "build_summary_csv": str(output_dir / "partition_build_summary.csv"),
        "notes": [
            "Offline oracle only; no lower HNSW search is executed.",
            "The load-recalibration row is intentionally identical to the topology row in routing metrics because current load recalibration only changes routing through fission.",
            "The full Method4 row uses the current 31-initial-shard fission path, which reconstructs the 46-shard operating point for default voting.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
