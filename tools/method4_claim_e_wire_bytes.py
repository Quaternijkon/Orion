#!/usr/bin/env python3
"""Collect client-observed request and response body bytes for Claim E.

This supplements the Claim E execution-mode evidence with byte accounting for
actual `/points/search/batch` calls. It measures the UTF-8 JSON request body
sent by this client and the raw response body returned by Qdrant.

It does not measure HTTP framing, TLS, compression, NIC-level counters,
controller CPU, or route-planning time.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple


DEFAULT_VARIANTS = [
    "grouped_by_ef_materialized",
    "compact_current",
    "client_shard_major_expanded",
]


CAVEAT = "client_observed_json_request_and_response_bodies_only_not_network_bytes"


BATCH_FIELDNAMES = [
    "variant",
    "batch_size",
    "batch_index",
    "query_start",
    "query_end",
    "query_count",
    "top_k",
    "status",
    "error_type",
    "error_message",
    "lower_execution_order",
    "search_batch_calls",
    "search_request_count",
    "avg_search_requests_per_query",
    "request_body_bytes",
    "request_body_bytes_per_query",
    "response_body_bytes",
    "response_body_bytes_per_query",
    "total_body_bytes",
    "total_body_bytes_per_query",
    "hits",
    "recall_at_10",
    "wall_s",
    "qps",
    "candidate_group_count",
    "returned_candidate_count",
    "avg_candidate_groups_per_query",
    "avg_returned_candidates_per_query",
]


SUMMARY_FIELDNAMES = [
    "variant",
    "batch_size",
    "status",
    "ok_batch_count",
    "error_batch_count",
    "query_count",
    "top_k",
    "recall_at_10",
    "wall_s",
    "qps",
    "search_batch_calls",
    "search_request_count",
    "avg_search_requests_per_query",
    "request_body_bytes",
    "request_body_bytes_per_query",
    "response_body_bytes",
    "response_body_bytes_per_query",
    "total_body_bytes",
    "total_body_bytes_per_query",
    "candidate_group_count",
    "returned_candidate_count",
    "avg_candidate_groups_per_query",
    "avg_returned_candidates_per_query",
    "request_body_bytes_reduction_vs_baseline_pct",
    "response_body_bytes_reduction_vs_baseline_pct",
    "total_body_bytes_reduction_vs_baseline_pct",
    "baseline_variant",
    "caveat",
]


class ParsedSearchBatchResponse(NamedTuple):
    response_body_bytes: int
    results: list[list[tuple[float, int]]]


class SearchBatchBytes(NamedTuple):
    request_body_bytes: int
    response_body_bytes: int
    results: list[list[tuple[float, int]]]


def load_module(path: str | Path, module_name: str) -> Any:
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_body_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def shard_major_searches_for_query(search: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    shard_keys = [str(shard_key) for shard_key in search.get("shard_key") or []]
    if not shard_keys:
        return [("", dict(search))]

    entry_points_by_shard = search.get("hnsw_entry_points_by_shard")
    ef_by_shard = search.get("hnsw_ef_by_shard")
    expanded: list[tuple[str, dict[str, Any]]] = []
    for shard_key in shard_keys:
        single = dict(search)
        single["shard_key"] = [shard_key]

        if entry_points_by_shard is not None:
            single.pop("hnsw_entry_points_by_shard", None)
            entry_points = entry_points_by_shard.get(shard_key)
            if entry_points:
                single["hnsw_entry_points"] = list(entry_points)
            else:
                single.pop("hnsw_entry_points", None)

        if ef_by_shard is not None:
            single.pop("hnsw_ef_by_shard", None)
            if shard_key in ef_by_shard:
                params = dict(single.get("params") or {})
                params["hnsw_ef"] = int(ef_by_shard[shard_key])
                single["params"] = params

        expanded.append((shard_key, single))
    return expanded


def flatten_searches_with_positions(
    plans: list[dict[str, Any]],
    *,
    lower_execution_order: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    if lower_execution_order not in {"query_major", "shard_major"}:
        raise ValueError(f"unsupported lower execution order: {lower_execution_order}")

    if lower_execution_order == "query_major":
        positions: list[int] = []
        searches: list[dict[str, Any]] = []
        for query_idx, plan in enumerate(plans):
            for search in plan["searches"]:
                positions.append(query_idx)
                searches.append(search)
        return positions, searches

    entries: list[tuple[str, int, int, dict[str, Any]]] = []
    original_order = 0
    for query_idx, plan in enumerate(plans):
        for search in plan["searches"]:
            for shard_key, single_search in shard_major_searches_for_query(search):
                entries.append((shard_key, query_idx, original_order, single_search))
                original_order += 1
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return [query_idx for _shard_key, query_idx, _order, _search in entries], [
        search for _shard_key, _query_idx, _order, search in entries
    ]


def parse_search_batch_response(raw_body: bytes) -> ParsedSearchBatchResponse:
    payload = json.loads(raw_body.decode("utf-8"))
    rows: list[list[tuple[float, int]]] = []
    for per_query in payload["result"]:
        row: list[tuple[float, int]] = []
        for item in per_query:
            payload_value = item.get("payload") or {}
            if "source_id" in payload_value:
                point_id = int(payload_value["source_id"])
            else:
                point_id = int(item["id"]) - 1
            row.append((float(item["score"]), point_id))
        rows.append(row)
    return ParsedSearchBatchResponse(response_body_bytes=len(raw_body), results=rows)


def post_search_batch_with_bytes(
    *,
    base_url: str,
    collection: str,
    searches: list[dict[str, Any]],
    timeout: float,
) -> SearchBatchBytes:
    request_body = json_body_bytes({"searches": searches})
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/collections/{urllib.parse.quote(collection, safe='')}/points/search/batch",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw_body = response.read()
    parsed = parse_search_batch_response(raw_body)
    return SearchBatchBytes(
        request_body_bytes=len(request_body),
        response_body_bytes=parsed.response_body_bytes,
        results=parsed.results,
    )


def execute_wire_batch(
    *,
    q2l: Any,
    base_url: str,
    collection: str,
    plans: list[dict[str, Any]],
    neighbors: Any,
    top_k: int,
    lower_execution_order: str,
    timeout: float,
    score_higher_is_better: bool,
) -> dict[str, Any]:
    positions, searches = flatten_searches_with_positions(
        plans,
        lower_execution_order=lower_execution_order,
    )
    started = time.perf_counter()
    batch = post_search_batch_with_bytes(
        base_url=base_url,
        collection=collection,
        searches=searches,
        timeout=timeout,
    )
    wall_s = time.perf_counter() - started

    per_query_candidates: list[list[list[tuple[float, int]]]] = [
        [] for _ in range(len(plans))
    ]
    for local_idx, result in zip(positions, batch.results):
        per_query_candidates[local_idx].append(result)

    candidate_group_count = sum(len(candidate_groups) for candidate_groups in per_query_candidates)
    returned_candidate_count = sum(
        len(group)
        for candidate_groups in per_query_candidates
        for group in candidate_groups
    )
    hits = 0
    for offset, candidate_groups in enumerate(per_query_candidates):
        top_ids = q2l.merge_topk_candidates(candidate_groups, top_k, score_higher_is_better)
        gt = set(map(int, neighbors[offset]))
        hits += len(set(top_ids) & gt)

    query_count = len(plans)
    return {
        "query_count": query_count,
        "hits": hits,
        "recall_at_10": hits / (query_count * top_k) if query_count else 0.0,
        "wall_s": wall_s,
        "qps": query_count / wall_s if wall_s > 0.0 else 0.0,
        "search_batch_calls": 1 if searches else 0,
        "search_request_count": len(searches),
        "avg_search_requests_per_query": len(searches) / query_count if query_count else 0.0,
        "request_body_bytes": batch.request_body_bytes,
        "request_body_bytes_per_query": batch.request_body_bytes / query_count if query_count else 0.0,
        "response_body_bytes": batch.response_body_bytes,
        "response_body_bytes_per_query": batch.response_body_bytes / query_count if query_count else 0.0,
        "total_body_bytes": batch.request_body_bytes + batch.response_body_bytes,
        "total_body_bytes_per_query": (
            (batch.request_body_bytes + batch.response_body_bytes) / query_count if query_count else 0.0
        ),
        "candidate_group_count": candidate_group_count,
        "returned_candidate_count": returned_candidate_count,
        "avg_candidate_groups_per_query": candidate_group_count / query_count if query_count else 0.0,
        "avg_returned_candidates_per_query": returned_candidate_count / query_count if query_count else 0.0,
    }


def error_batch_row(
    *,
    variant: str,
    batch_size: int,
    batch_index: int,
    query_start: int,
    query_end: int,
    top_k: int,
    lower_execution_order: str,
    error: BaseException,
) -> dict[str, Any]:
    query_count = max(0, query_end - query_start)
    return {
        "variant": variant,
        "batch_size": batch_size,
        "batch_index": batch_index,
        "query_start": query_start,
        "query_end": query_end,
        "query_count": query_count,
        "top_k": top_k,
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "lower_execution_order": lower_execution_order,
        "search_batch_calls": "",
        "search_request_count": "",
        "avg_search_requests_per_query": "",
        "request_body_bytes": "",
        "request_body_bytes_per_query": "",
        "response_body_bytes": "",
        "response_body_bytes_per_query": "",
        "total_body_bytes": "",
        "total_body_bytes_per_query": "",
        "hits": "",
        "recall_at_10": "",
        "wall_s": "",
        "qps": "",
        "candidate_group_count": "",
        "returned_candidate_count": "",
        "avg_candidate_groups_per_query": "",
        "avg_returned_candidates_per_query": "",
    }


def run_variant_batches(
    *,
    q2l: Any,
    base_url: str,
    collection: str,
    variant: str,
    lower_execution_order: str,
    plans: list[dict[str, Any]],
    neighbors: Any,
    top_k: int,
    batch_size: int,
    timeout: float,
    score_higher_is_better: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_index, start_idx in enumerate(range(0, len(plans), batch_size)):
        end_idx = min(start_idx + batch_size, len(plans))
        try:
            result = execute_wire_batch(
                q2l=q2l,
                base_url=base_url,
                collection=collection,
                plans=plans[start_idx:end_idx],
                neighbors=neighbors[start_idx:end_idx],
                top_k=top_k,
                lower_execution_order=lower_execution_order,
                timeout=timeout,
                score_higher_is_better=score_higher_is_better,
            )
            row = {
                "variant": variant,
                "batch_size": batch_size,
                "batch_index": batch_index,
                "query_start": start_idx,
                "query_end": end_idx,
                "top_k": top_k,
                "status": "ok",
                "error_type": "",
                "error_message": "",
                "lower_execution_order": lower_execution_order,
                **result,
            }
        except Exception as exc:
            row = error_batch_row(
                variant=variant,
                batch_size=batch_size,
                batch_index=batch_index,
                query_start=start_idx,
                query_end=end_idx,
                top_k=top_k,
                lower_execution_order=lower_execution_order,
                error=exc,
            )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
    return rows


def reduction_pct(value: float, baseline: float | None) -> float | str:
    if baseline in (None, 0):
        return ""
    return (float(baseline) - float(value)) / float(baseline) * 100.0


def summarize_wire_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_variant: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["batch_size"]), str(row["variant"]))].append(row)

    summaries: list[dict[str, Any]] = []
    baseline_by_batch: dict[int, dict[str, Any]] = {}
    for (batch_size, variant), group_rows in sorted(grouped.items()):
        ok_rows = [row for row in group_rows if row.get("status") == "ok"]
        query_count = sum(int(row["query_count"]) for row in ok_rows)
        top_k = int(ok_rows[0].get("top_k") or group_rows[0].get("top_k") or 10)
        hits = sum(int(row.get("hits") or 0) for row in ok_rows)
        wall_s = sum(float(row.get("wall_s") or 0.0) for row in ok_rows)
        search_request_count = sum(int(row.get("search_request_count") or 0) for row in ok_rows)
        request_body_bytes = sum(int(row.get("request_body_bytes") or 0) for row in ok_rows)
        response_body_bytes = sum(int(row.get("response_body_bytes") or 0) for row in ok_rows)
        total_body_bytes = sum(int(row.get("total_body_bytes") or 0) for row in ok_rows)
        candidate_group_count = sum(int(row.get("candidate_group_count") or 0) for row in ok_rows)
        returned_candidate_count = sum(int(row.get("returned_candidate_count") or 0) for row in ok_rows)
        summary = {
            "variant": variant,
            "batch_size": batch_size,
            "status": "ok" if len(ok_rows) == len(group_rows) else ("partial" if ok_rows else "error"),
            "ok_batch_count": len(ok_rows),
            "error_batch_count": len(group_rows) - len(ok_rows),
            "query_count": query_count,
            "top_k": top_k,
            "recall_at_10": hits / (query_count * top_k) if query_count else "",
            "wall_s": wall_s if ok_rows else "",
            "qps": query_count / wall_s if wall_s > 0.0 else "",
            "search_batch_calls": sum(int(row.get("search_batch_calls") or 0) for row in ok_rows),
            "search_request_count": search_request_count,
            "avg_search_requests_per_query": search_request_count / query_count if query_count else "",
            "request_body_bytes": request_body_bytes,
            "request_body_bytes_per_query": request_body_bytes / query_count if query_count else "",
            "response_body_bytes": response_body_bytes,
            "response_body_bytes_per_query": response_body_bytes / query_count if query_count else "",
            "total_body_bytes": total_body_bytes,
            "total_body_bytes_per_query": total_body_bytes / query_count if query_count else "",
            "candidate_group_count": candidate_group_count,
            "returned_candidate_count": returned_candidate_count,
            "avg_candidate_groups_per_query": candidate_group_count / query_count if query_count else "",
            "avg_returned_candidates_per_query": returned_candidate_count / query_count if query_count else "",
            "request_body_bytes_reduction_vs_baseline_pct": "",
            "response_body_bytes_reduction_vs_baseline_pct": "",
            "total_body_bytes_reduction_vs_baseline_pct": "",
            "baseline_variant": baseline_variant,
            "caveat": CAVEAT,
        }
        if variant == baseline_variant and ok_rows:
            baseline_by_batch[batch_size] = summary
        summaries.append(summary)

    for summary in summaries:
        baseline = baseline_by_batch.get(int(summary["batch_size"]))
        summary["request_body_bytes_reduction_vs_baseline_pct"] = reduction_pct(
            float(summary["request_body_bytes"] or 0.0),
            float(baseline["request_body_bytes"]) if baseline else None,
        )
        summary["response_body_bytes_reduction_vs_baseline_pct"] = reduction_pct(
            float(summary["response_body_bytes"] or 0.0),
            float(baseline["response_body_bytes"]) if baseline else None,
        )
        summary["total_body_bytes_reduction_vs_baseline_pct"] = reduction_pct(
            float(summary["total_body_bytes"] or 0.0),
            float(baseline["total_body_bytes"]) if baseline else None,
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:6833")
    parser.add_argument("--collection", default="qdrant_controller_idea_method4map_full_20260601")
    parser.add_argument("--routing-source-collection", default=None)
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--num-shards", type=int, default=46)
    parser.add_argument("--upper-k", type=int, default=160)
    parser.add_argument("--base-ef", type=int, default=80)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-query-count", type=int, default=3000)
    parser.add_argument("--warmup-query-count", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[200])
    parser.add_argument("--variant", action="append", default=None, choices=DEFAULT_VARIANTS)
    parser.add_argument("--sample-denominator", type=int, default=32)
    parser.add_argument("--upper-sample-seed", type=int, default=100)
    parser.add_argument("--upper-m", type=int, default=32)
    parser.add_argument("--upper-ef-construction", type=int, default=100)
    parser.add_argument("--upper-search-ef", type=int, default=200)
    parser.add_argument("--scroll-page-size", type=int, default=20000)
    parser.add_argument("--baseline-variant", default="client_shard_major_expanded")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--score-higher-is-better", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", default="results/method4_claim_e_wire_bytes_20260705")
    parser.add_argument("--qdrant-tool", default="tools/qdrant_two_level_routing_experiment.py")
    parser.add_argument("--claim-e-tool", default="tools/method4_claim_e_execution_mode_latency.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    q2l = load_module(args.qdrant_tool, "qdrant_two_level_routing_experiment")
    claim_e = load_module(args.claim_e_tool, "method4_claim_e_execution_mode_latency")
    specs = claim_e.variant_specs(args.variant)
    output_dir = Path(args.output_root) / time.strftime("analysis_%Y%m%d_%H%M%S", time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir {output_dir}", flush=True)

    queries, upper_vectors, neighbors, train_count, dim = claim_e.load_queries_and_neighbors(args, q2l)
    upper_indices = q2l.global_upper_indices(train_count, args.sample_denominator, args.upper_sample_seed)
    upper_index = q2l.build_upper_index(
        upper_vectors,
        upper_indices.astype("int64", copy=False),
        dim,
        args.upper_m,
        args.upper_ef_construction,
        max(args.upper_search_ef, args.upper_k),
    )
    routing_source = args.routing_source_collection or args.collection
    point_to_shards = claim_e.recover_upper_membership(args, q2l, routing_source, upper_indices, train_count)

    plans_by_mode: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        mode = spec["routed_execution_mode"]
        if mode in plans_by_mode:
            continue
        plans_by_mode[mode] = claim_e.build_plans_for_execution_mode(
            q2l,
            queries,
            upper_index,
            point_to_shards,
            args.num_shards,
            args.top_k,
            args.upper_k,
            args.base_ef,
            args.factor,
            mode,
            train_count + 1,
        )[args.warmup_query_count :]

    measured_neighbors = neighbors[args.warmup_query_count :]
    batch_rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        for spec in specs:
            batch_rows.extend(
                run_variant_batches(
                    q2l=q2l,
                    base_url=args.base_url,
                    collection=args.collection,
                    variant=spec["variant"],
                    lower_execution_order=spec["lower_execution_order"],
                    plans=plans_by_mode[spec["routed_execution_mode"]],
                    neighbors=measured_neighbors,
                    top_k=args.top_k,
                    batch_size=batch_size,
                    timeout=args.timeout,
                    score_higher_is_better=args.score_higher_is_better,
                )
            )

    summary_rows = summarize_wire_rows(batch_rows, baseline_variant=args.baseline_variant)
    write_csv(output_dir / "claim_e_wire_bytes_batches.csv", batch_rows, BATCH_FIELDNAMES)
    write_csv(output_dir / "claim_e_wire_bytes_summary.csv", summary_rows, SUMMARY_FIELDNAMES)
    metadata = {
        "analysis_kind": "claim_e_client_observed_request_response_body_bytes",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "collection": args.collection,
        "routing_source_collection": routing_source,
        "hdf5_path": args.hdf5_path,
        "num_points": train_count,
        "num_shards": args.num_shards,
        "upper_k": args.upper_k,
        "base_ef": args.base_ef,
        "factor": args.factor,
        "top_k": args.top_k,
        "eval_query_count": args.eval_query_count,
        "warmup_query_count_excluded_from_byte_summary": args.warmup_query_count,
        "batch_sizes": args.batch_sizes,
        "variants": specs,
        "baseline_variant": args.baseline_variant,
        "notes": [
            CAVEAT,
            "The script sends actual /points/search/batch requests for the selected variants and records JSON request and response body byte counts.",
            "HTTP framing, compression, NIC-level network counters, route-planning time, controller CPU, and RSS are not measured.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
