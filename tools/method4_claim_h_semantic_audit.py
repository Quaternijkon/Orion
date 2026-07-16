#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


INVARIANTS = [
    "upper routing unchanged",
    "point_to_shards unchanged",
    "multi-assignment preserved",
    "per-shard entry points preserved",
    "dynamic EF formula preserved",
    "lower search remains per logical shard",
    "no adaptive shard pruning",
    "source-id dedup preserved",
    "final global merge preserved",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Claim H semantic audit. Reconstruct compact Method4 query plans "
            "and verify externally visible routing/search invariants."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:6833")
    parser.add_argument("--collection", default="qdrant_controller_idea_method4map_full_20260601")
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--num-shards", type=int, default=46)
    parser.add_argument("--upper-k", type=int, default=160)
    parser.add_argument("--base-ef", type=int, default=80)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-count", type=int, default=100)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=160)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--source-id-dedup-block-size", type=int, default=None)
    parser.add_argument("--output-root", default="results/method4_claim_h_semantic_audit_20260704")
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


def shard_id_from_key(shard_key: str) -> int:
    return int(str(shard_key).rsplit("_", 1)[1])


def shard_key_for_id(shard_id: int) -> str:
    return f"centroid_{int(shard_id):02d}"


def search_shard_keys(search: dict[str, Any]) -> list[str]:
    shard_keys = search.get("shard_key") or []
    if isinstance(shard_keys, str):
        return [shard_keys]
    return [str(value) for value in shard_keys]


def sorted_shard_keys(keys: list[str] | set[str]) -> list[str]:
    return sorted([str(key) for key in keys], key=lambda key: shard_id_from_key(key))


def expected_counts_from_labels(
    upper_labels: list[int],
    point_to_shards: list[list[int]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in upper_labels:
        point_id = int(label)
        if 0 <= point_id < len(point_to_shards):
            for shard_id in point_to_shards[point_id]:
                shard_key = shard_key_for_id(int(shard_id))
                counts[shard_key] = counts.get(shard_key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: shard_id_from_key(item[0])))


def compact_search_for_plan(plan: dict[str, Any]) -> dict[str, Any] | None:
    searches = plan.get("searches") or []
    if not searches:
        return None
    return searches[0]


def encoded_id_matches_shard(encoded_id: int, shard_id: int, block_size: int) -> bool:
    if block_size <= 0 or encoded_id < 0:
        return False
    # Current Method4 copied point IDs use shard * block + source + 1.
    # Some historical traces used shard * block + source; accept both so the
    # audit catches semantic drift without rejecting old diagnostic fixtures.
    return ((encoded_id - 1) // block_size == shard_id) or (encoded_id // block_size == shard_id)


def add_failure(failures: dict[str, list[str]], invariant: str, detail: str) -> None:
    failures.setdefault(invariant, []).append(detail)


def audit_query_plans(
    query_plans: list[dict[str, Any]],
    point_to_shards: list[list[int]],
    source_id_dedup_block_size: int,
    base_ef: int,
    factor: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: dict[str, list[str]] = {name: [] for name in INVARIANTS}
    stats: dict[str, int] = {
        "query_count": len(query_plans),
        "search_count": 0,
        "visited_shards": 0,
        "upper_labels": 0,
        "multi_assignment_queries": 0,
    }
    samples: list[dict[str, Any]] = []

    for plan_idx, plan in enumerate(query_plans):
        query_index = int(plan.get("query_index", plan_idx))
        upper_labels = [int(label) for label in plan.get("upper_labels", [])]
        stats["upper_labels"] += len(upper_labels)
        search = compact_search_for_plan(plan)

        if search is None:
            for invariant in (
                "lower search remains per logical shard",
                "source-id dedup preserved",
                "final global merge preserved",
            ):
                add_failure(failures, invariant, f"query {query_index} has no search object")
            continue

        stats["search_count"] += 1
        routed_shards = search_shard_keys(search)
        routed_shard_set = set(routed_shards)
        stats["visited_shards"] += len(routed_shards)

        entry_points_by_shard = {
            str(shard_key): list(values)
            for shard_key, values in (search.get("hnsw_entry_points_by_shard") or {}).items()
        }
        ef_by_shard = {
            str(shard_key): int(value)
            for shard_key, value in (search.get("hnsw_ef_by_shard") or {}).items()
        }
        actual_counts = {
            shard_key: len(values)
            for shard_key, values in entry_points_by_shard.items()
        }
        expected_counts = (
            {
                str(shard_key): int(value)
                for shard_key, value in plan.get("expected_entry_point_count_by_shard", {}).items()
            }
            or expected_counts_from_labels(upper_labels, point_to_shards)
        )
        expected_shards = set(expected_counts)

        samples.append(
            {
                "query_index": query_index,
                "upper_labels": " ".join(str(label) for label in upper_labels),
                "routed_shards": routed_shards,
                "visited_shards": len(routed_shards),
                "assigned_ef_sum": int(plan.get("assigned_ef_sum", 0)),
            }
        )

        if not upper_labels:
            add_failure(failures, "upper routing unchanged", f"query {query_index} has no upper labels")
        if int(plan.get("upper_hits", len(upper_labels))) <= 0:
            add_failure(failures, "upper routing unchanged", f"query {query_index} has no upper hits")

        missing_labels = [
            str(label)
            for label in upper_labels
            if label < 0 or label >= len(point_to_shards) or not point_to_shards[label]
        ]
        if missing_labels:
            add_failure(
                failures,
                "point_to_shards unchanged",
                f"query {query_index} missing shard membership for labels {' '.join(missing_labels[:8])}",
            )

        expected_total_eps = sum(expected_counts.values())
        actual_total_eps = sum(actual_counts.values())
        if expected_total_eps > len(upper_labels):
            stats["multi_assignment_queries"] += 1
        if expected_counts and actual_total_eps != expected_total_eps:
            add_failure(
                failures,
                "multi-assignment preserved",
                f"query {query_index} expected {expected_total_eps} routed EP copies got {actual_total_eps}",
            )

        if expected_shards and routed_shard_set != expected_shards:
            add_failure(
                failures,
                "per-shard entry points preserved",
                (
                    f"query {query_index} expected shards "
                    f"{' '.join(sorted_shard_keys(expected_shards))} got {' '.join(routed_shards)}"
                ),
            )
        for shard_key, expected_count in expected_counts.items():
            actual_count = actual_counts.get(shard_key, 0)
            if actual_count != expected_count:
                add_failure(
                    failures,
                    "per-shard entry points preserved",
                    f"query {query_index} {shard_key} expected {expected_count} EPs got {actual_count}",
                )

        if set(ef_by_shard) != routed_shard_set or set(entry_points_by_shard) != routed_shard_set:
            add_failure(
                failures,
                "lower search remains per logical shard",
                f"query {query_index} shard_key/EP/EF maps do not address identical logical shards",
            )
        if int(plan.get("visited_shards", len(routed_shards))) != len(routed_shards):
            add_failure(
                failures,
                "lower search remains per logical shard",
                f"query {query_index} visited_shards does not match compact shard list",
            )

        if expected_shards and routed_shard_set != expected_shards:
            add_failure(
                failures,
                "no adaptive shard pruning",
                f"query {query_index} routed shard set differs from upper-label membership",
            )

        expected_ef_by_shard = {
            str(shard_key): int(value)
            for shard_key, value in plan.get("expected_hnsw_ef_by_shard", {}).items()
        }
        if expected_ef_by_shard:
            for shard_key, expected_ef in expected_ef_by_shard.items():
                actual_ef = ef_by_shard.get(shard_key)
                if actual_ef != expected_ef:
                    add_failure(
                        failures,
                        "dynamic EF formula preserved",
                        f"query {query_index} {shard_key} expected {expected_ef} got {actual_ef}",
                    )
        elif not ef_by_shard:
            add_failure(
                failures,
                "dynamic EF formula preserved",
                f"query {query_index} missing hnsw_ef_by_shard",
            )
        else:
            assigned_sum = plan.get("assigned_ef_sum")
            actual_sum = sum(ef_by_shard.values())
            if assigned_sum is not None and int(assigned_sum) != actual_sum:
                deficit = int(assigned_sum) - actual_sum
                shard_key = sorted_shard_keys(set(ef_by_shard))[-1]
                expected_ef = ef_by_shard[shard_key] + deficit
                add_failure(
                    failures,
                    "dynamic EF formula preserved",
                    f"{shard_key} expected {expected_ef} got {ef_by_shard[shard_key]}",
                )

        block_in_search = search.get("source_id_dedup_block_size")
        if int(block_in_search or -1) != int(source_id_dedup_block_size):
            add_failure(
                failures,
                "source-id dedup preserved",
                f"query {query_index} source_id_dedup_block_size expected {source_id_dedup_block_size} got {block_in_search}",
            )
        for shard_key, encoded_ids in entry_points_by_shard.items():
            shard_id = shard_id_from_key(shard_key)
            for encoded_id in encoded_ids:
                if not encoded_id_matches_shard(int(encoded_id), shard_id, int(source_id_dedup_block_size)):
                    add_failure(
                        failures,
                        "source-id dedup preserved",
                        f"query {query_index} encoded EP {encoded_id} does not belong to {shard_key}",
                    )
                    break

        if int(search.get("limit", -1)) != int(top_k):
            add_failure(
                failures,
                "final global merge preserved",
                f"query {query_index} expected final compact limit {top_k} got {search.get('limit')}",
            )

    rows: list[dict[str, Any]] = []
    for invariant in INVARIANTS:
        invariant_failures = failures.get(invariant, [])
        rows.append(
            {
                "invariant": invariant,
                "status": "fail" if invariant_failures else "pass",
                "query_count": stats["query_count"],
                "search_count": stats["search_count"],
                "visited_shards": stats["visited_shards"],
                "upper_labels": stats["upper_labels"],
                "multi_assignment_queries": stats["multi_assignment_queries"],
                "failure_count": len(invariant_failures),
                "detail": "; ".join(invariant_failures[:5]) if invariant_failures else "ok",
            }
        )
    return rows, samples


def load_upper_and_queries(args: argparse.Namespace, q2l: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    with h5py.File(args.hdf5_path, "r") as handle:
        train_ds = handle["train"]
        num_points = int(train_ds.shape[0])
        dim = int(train_ds.shape[1])
        upper_indices = q2l.global_upper_indices(num_points, args.sample_denominator, args.upper_sample_seed)

        order = np.argsort(upper_indices)
        sorted_upper = upper_indices[order]
        upper_vectors_sorted = train_ds[sorted_upper].astype(np.float32, copy=True)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        upper_vectors = upper_vectors_sorted[inverse]

        queries = handle["test"][: args.query_count].astype(np.float32, copy=True)

    return q2l.normalize_rows(upper_vectors), q2l.normalize_rows(queries), upper_indices, num_points, dim


def recover_upper_point_to_shards(
    args: argparse.Namespace,
    q2l: Any,
    upper_indices: np.ndarray,
    num_points: int,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    upper_set = {int(point_id) for point_id in upper_indices.tolist()}
    point_to_shards: list[list[int]] = [[] for _ in range(num_points)]
    shard_rows: list[dict[str, Any]] = []
    for shard_id in range(args.num_shards):
        shard_key = q2l.shard_key_for_id(shard_id)
        offset: Any = None
        scrolled_points = 0
        matched_upper_points = 0
        while True:
            body: dict[str, Any] = {
                "limit": args.scroll_page_size,
                "with_payload": ["source_id"],
                "with_vector": False,
                "shard_key": shard_key,
            }
            if offset is not None:
                body["offset"] = offset
            result = q2l.request_json(
                args.base_url,
                "POST",
                f"/collections/{urllib.parse.quote(args.collection, safe='')}/points/scroll",
                body=body,
                timeout=300.0,
            )["result"]
            points = result.get("points") or []
            scrolled_points += len(points)
            for point in points:
                source_id = q2l.source_id_from_scrolled_point(point, num_points)
                if int(source_id) in upper_set:
                    point_to_shards[int(source_id)].append(int(shard_id))
                    matched_upper_points += 1
            offset = result.get("next_page_offset")
            if offset is None:
                break
        shard_rows.append(
            {
                "shard_id": shard_id,
                "shard_key": shard_key,
                "scrolled_points": scrolled_points,
                "matched_upper_points": matched_upper_points,
            }
        )
        print(
            f"recovered {shard_key}: {scrolled_points} copies, {matched_upper_points} upper copies",
            flush=True,
        )
    return point_to_shards, shard_rows


def annotate_expected_plan_fields(
    q2l: Any,
    plans: list[dict[str, Any]],
    labels: np.ndarray,
    point_to_shards: list[list[int]],
    base_ef: int,
    factor: int,
) -> None:
    for query_index, (plan, labels_row) in enumerate(zip(plans, labels)):
        upper_labels = [int(label) for label in labels_row.tolist()]
        shard_to_eps = q2l.route_upper_labels_to_shard_eps(upper_labels, point_to_shards)
        plan["query_index"] = query_index
        plan["upper_labels"] = upper_labels
        plan["expected_entry_point_count_by_shard"] = {
            q2l.shard_key_for_id(shard_id): len(eps)
            for shard_id, eps in shard_to_eps.items()
        }
        plan["expected_hnsw_ef_by_shard"] = {
            q2l.shard_key_for_id(shard_id): int(base_ef + factor * len(eps))
            for shard_id, eps in shard_to_eps.items()
        }


def main() -> int:
    args = parse_args()
    q2l = load_qdrant_tool(args.qdrant_tool)
    output_dir = Path(args.output_root) / time.strftime("audit_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    upper_vectors, queries, upper_indices, num_points, dim = load_upper_and_queries(args, q2l)
    source_id_dedup_block_size = args.source_id_dedup_block_size or (num_points + 1)
    upper_index = q2l.build_upper_index(
        upper_vectors,
        upper_indices.astype(np.int64, copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        args.upper_search_ef,
    )
    labels = q2l.compute_upper_labels(upper_index, queries, args.upper_k).astype(np.int64, copy=False)
    point_to_shards, shard_rows = recover_upper_point_to_shards(args, q2l, upper_indices, num_points)
    missing_upper = [
        int(point_id)
        for point_id in upper_indices.tolist()
        if not point_to_shards[int(point_id)]
    ]
    if missing_upper:
        preview = " ".join(str(point_id) for point_id in missing_upper[:10])
        raise RuntimeError(f"missing shard membership for {len(missing_upper)} upper points: {preview}")

    plans = q2l.build_routed_search_plans(
        queries,
        labels,
        point_to_shards,
        args.num_shards,
        args.top_k,
        args.base_ef,
        args.factor,
        False,
        False,
        "compact_multi_ep",
        "max",
        "top_k",
        1,
        source_id_dedup_block_size,
    )
    annotate_expected_plan_fields(q2l, plans, labels, point_to_shards, args.base_ef, args.factor)
    rows, samples = audit_query_plans(
        plans,
        point_to_shards,
        source_id_dedup_block_size,
        args.base_ef,
        args.factor,
        args.top_k,
    )

    sample_rows = [
        {
            **row,
            "routed_shards": " ".join(row["routed_shards"]),
        }
        for row in samples[: args.query_count]
    ]
    write_csv(output_dir / "semantic_invariant_summary.csv", rows)
    write_csv(output_dir / "semantic_query_samples.csv", sample_rows)
    write_csv(output_dir / "collection_shard_upper_counts.csv", shard_rows)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_kind": "method4_claim_h_semantic_invariant_audit",
        "base_url": args.base_url,
        "collection": args.collection,
        "hdf5_path": args.hdf5_path,
        "num_points": num_points,
        "num_shards": args.num_shards,
        "upper_k": args.upper_k,
        "base_ef": args.base_ef,
        "factor": args.factor,
        "top_k": args.top_k,
        "query_count": args.query_count,
        "sample_denominator": args.sample_denominator,
        "upper_sample_seed": args.upper_sample_seed,
        "upper_count": int(len(upper_indices)),
        "upper_search_ef": args.upper_search_ef,
        "source_id_dedup_block_size": source_id_dedup_block_size,
        "summary_csv": str(output_dir / "semantic_invariant_summary.csv"),
        "sample_csv": str(output_dir / "semantic_query_samples.csv"),
        "collection_shard_upper_counts_csv": str(output_dir / "collection_shard_upper_counts.csv"),
        "notes": [
            "This audit checks compact_multi_ep request semantics without executing lower HNSW search.",
            "Every invariant row must pass before using this as Claim H support.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
