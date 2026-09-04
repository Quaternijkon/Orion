#!/usr/bin/env python3
"""Run Stage 15 with one non-empty HNSW segment per logical shard."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

import c1_stage14_physical_retest as stage14
from c1_benchmark import ensure_collection, evaluate_queries, write_per_query_csv
from c1_e1_scaleout import execute as execute_e1
from c1_protocol import TARGET_RECALL, sha256_path, utc_timestamp


DATASET_ORDER = stage14.DATASET_ORDER
PHYSICAL_COUNTS = stage14.PHYSICAL_COUNTS
FIXED_EF = {"sift1m": 24, "glove-200-angular": 320}
GLOVE_SINGLE_GRAPH_EF_GRID = (192, 224, 256, 320, 384)
GLOVE_SINGLE_GRAPH_EF_HARD_CAP = max(GLOVE_SINGLE_GRAPH_EF_GRID)
MAX_SEGMENT_SIZE_KB = 2_000_000
EXPECTED_TOTAL_SEGMENTS_PER_LOGICAL_SHARD = 2
FANOUT_TUNING_QUERY_COUNT = 1_000
FANOUT_HOLDOUT_QUERY_COUNT = 9_000
FANOUT_QUERY_CONCURRENCY = 16
FANOUT_REQUEST_WORKERS = 128


def collection_name(dataset: str, method: str, shards: int, suffix: str = "") -> str:
    tail = f"_{suffix}" if suffix else ""
    return f"c1s15_{stage14.dataset_token(dataset)}_{method}_m{shards}{tail}"


def e1_output(
    args: Namespace, dataset: str, method: str, shards: int, label: str
) -> Path:
    return args.output_root / "e1" / f"stage15-e1-{dataset}-{method}-m{shards}-{label}.json"


def valid_e1(
    path: Path, *, dataset: str, method: str, shards: int, fanout: int, ef: int
) -> bool:
    if not path.is_file():
        return False
    payload = stage14.load_json(path)
    return (
        payload.get("status") == "VALID_E1"
        and payload.get("dataset") == dataset
        and payload.get("partition_method") == method
        and int(payload.get("logical_shard_count") or 0) == shards
        and int(payload.get("physical_machine_count") or 0) == shards
        and int(payload.get("fanout") or 0) == fanout
        and int(payload.get("ef_search") or 0) == ef
    )


def prepare_log(args: Namespace, collection: str) -> Path:
    return args.output_root / "logs" / f"{collection}-prepare.json"


def fanout_selection_path(args: Namespace, dataset: str, shards: int) -> Path:
    ef = FIXED_EF[dataset]
    return (
        args.output_root
        / "fanout"
        / "holdout"
        / f"{dataset}-kmeans-m{shards}-ef{ef}-selected.json"
    )


def valid_stage15_fanout_selection(
    path: Path, *, dataset: str, shards: int
) -> bool:
    if not path.is_file():
        return False
    payload = stage14.load_json(path)
    fanout = int(payload.get("fanout") or 0)
    return (
        payload.get("record_type")
        == "stage15_single_graph_controlled_ef_holdout_selection"
        and payload.get("selected_after_holdout") is True
        and payload.get("result_status") == "VALID"
        and payload.get("dataset") == dataset
        and payload.get("partition_method") == "kmeans"
        and int(payload.get("logical_shards") or 0) == shards
        and int(payload.get("ef_search") or 0) == FIXED_EF[dataset]
        and 1 <= fanout <= shards
        and float(payload.get("achieved_recall") or 0.0) >= TARGET_RECALL
    )


def measure_stage15_kmeans_fanout(
    args: Namespace, *, dataset: str, shards: int, collection: str
) -> tuple[int, Path]:
    selection = fanout_selection_path(args, dataset, shards)
    if valid_stage15_fanout_selection(
        selection, dataset=dataset, shards=shards
    ):
        return int(stage14.load_json(selection)["fanout"]), selection

    ef = FIXED_EF[dataset]
    root = args.output_root / "fanout"
    stem = f"{dataset}-kmeans-m{shards}-ef{ef}"
    tuning: list[dict[str, Any]] = []
    for fanout in range(1, shards + 1):
        print(f"[stage15] fanout tuning {stem}: P={fanout}", flush=True)
        rows, summary = evaluate_queries(
            args.topology,
            stage14.dataset_path(args, dataset),
            stage14.partition_path(args, dataset, "kmeans", shards),
            dataset,
            "kmeans",
            shards,
            collection,
            fanout,
            ef,
            0,
            FANOUT_TUNING_QUERY_COUNT,
            query_concurrency=FANOUT_QUERY_CONCURRENCY,
            request_workers=FANOUT_REQUEST_WORKERS,
            require_worker_cpu_time=True,
            require_benchmark_affinity=True,
        )
        per_query = root / "tuning" / "per_query" / f"{stem}-p{fanout}.csv"
        write_per_query_csv(per_query, rows)
        summary.update(
            {
                "record_type": "stage15_single_graph_controlled_ef_fanout_tuning",
                "per_query": str(per_query),
                "per_query_sha256": sha256_path(per_query),
                "single_nonempty_hnsw_segment_per_shard": True,
                "selection_rule": "minimum_fanout_at_fixed_bounded_ef",
            }
        )
        summary_path = root / "tuning" / f"{stem}-p{fanout}.json"
        stage14.write_json(summary_path, summary)
        tuning.append(
            {
                "fanout": fanout,
                "achieved_recall": float(summary["achieved_recall"]),
                "result_status": summary["result_status"],
                "summary": str(summary_path),
                "summary_sha256": sha256_path(summary_path),
            }
        )

    feasible = [
        row for row in tuning if float(row["achieved_recall"]) >= TARGET_RECALL
    ]
    initial_fanout = int(feasible[0]["fanout"]) if feasible else shards
    attempts: list[dict[str, Any]] = []
    for fanout in range(initial_fanout, shards + 1):
        print(f"[stage15] fanout holdout {stem}: P={fanout}", flush=True)
        rows, summary = evaluate_queries(
            args.topology,
            stage14.dataset_path(args, dataset),
            stage14.partition_path(args, dataset, "kmeans", shards),
            dataset,
            "kmeans",
            shards,
            collection,
            fanout,
            ef,
            FANOUT_TUNING_QUERY_COUNT,
            FANOUT_HOLDOUT_QUERY_COUNT,
            query_concurrency=FANOUT_QUERY_CONCURRENCY,
            request_workers=FANOUT_REQUEST_WORKERS,
            require_worker_cpu_time=True,
            require_benchmark_affinity=True,
        )
        per_query = root / "holdout" / "per_query" / f"{stem}-p{fanout}.csv"
        write_per_query_csv(per_query, rows)
        summary.update(
            {
                "record_type": "stage15_single_graph_controlled_ef_holdout",
                "per_query": str(per_query),
                "per_query_sha256": sha256_path(per_query),
                "single_nonempty_hnsw_segment_per_shard": True,
                "tuning_candidates": tuning,
                "tuning_selected_fanout": initial_fanout,
                "selection_rule": (
                    "increase_fanout_only_on_disjoint_holdout_while_ef_is_fixed"
                ),
            }
        )
        attempt_path = root / "holdout" / f"{stem}-p{fanout}.json"
        stage14.write_json(attempt_path, summary)
        attempts.append(
            {
                "fanout": fanout,
                "achieved_recall": float(summary["achieved_recall"]),
                "result_status": summary["result_status"],
                "summary": str(attempt_path),
                "summary_sha256": sha256_path(attempt_path),
            }
        )
        if float(summary["achieved_recall"]) >= TARGET_RECALL:
            summary.update(
                {
                    "record_type": (
                        "stage15_single_graph_controlled_ef_holdout_selection"
                    ),
                    "holdout_attempts": attempts,
                    "selected_after_holdout": True,
                    "bounded_ef_grid": list(GLOVE_SINGLE_GRAPH_EF_GRID)
                    if dataset == "glove-200-angular"
                    else None,
                    "ef_hard_cap": GLOVE_SINGLE_GRAPH_EF_HARD_CAP
                    if dataset == "glove-200-angular"
                    else ef,
                }
            )
            stage14.write_json(selection, summary)
            return fanout, selection

    failure = {
        "timestamp": utc_timestamp(),
        "record_type": "stage15_single_graph_controlled_ef_holdout_selection",
        "dataset": dataset,
        "partition_method": "kmeans",
        "logical_shards": shards,
        "ef_search": ef,
        "tuning_candidates": tuning,
        "holdout_attempts": attempts,
        "selected_after_holdout": False,
        "result_status": "INFEASIBLE_AT_FIXED_BOUNDED_EF",
    }
    stage14.write_json(selection, failure)
    raise RuntimeError(f"full fan-out misses recall target: {selection}")


def prepare_valid(args: Namespace, collection: str, shards: int) -> bool:
    path = prepare_log(args, collection)
    if not path.is_file():
        return False
    payload = stage14.load_json(path)
    gate = payload.get("single_indexed_segment_per_shard_gate") or {}
    return (
        gate.get("status") == "PASS"
        and int(gate.get("logical_shards") or 0) == shards
        and int(gate.get("expected_total_segments") or 0)
        == EXPECTED_TOTAL_SEGMENTS_PER_LOGICAL_SHARD * shards
        and int(gate.get("observed_total_segments") or 0)
        == EXPECTED_TOTAL_SEGMENTS_PER_LOGICAL_SHARD * shards
        and int(payload.get("max_segment_size_kb") or 0)
        == args.max_segment_size_kb
    )


def prepare_collection(
    args: Namespace, dataset: str, method: str, shards: int, collection: str
) -> None:
    print(
        f"[stage15] prepare {collection}: dataset={dataset} method={method} M={shards}",
        flush=True,
    )
    payload = ensure_collection(
        args.topology,
        stage14.dataset_path(args, dataset),
        dataset,
        method,
        shards,
        stage14.partition_path(args, dataset, method, shards),
        collection,
        batch_size=args.upload_batch_size,
        replace=True,
        max_segment_size_kb=args.max_segment_size_kb,
    )
    observed = int(payload["collection_info"]["segments_count"])
    expected = EXPECTED_TOTAL_SEGMENTS_PER_LOGICAL_SHARD * shards
    gate = {
        "status": "PASS" if observed == expected else "FAIL",
        "logical_shards": shards,
        "expected_total_segments": expected,
        "observed_total_segments": observed,
        "expected_nonempty_indexed_hnsw_segments": shards,
        "expected_empty_appendable_segments": shards,
        "max_segment_size_kb": args.max_segment_size_kb,
    }
    payload["single_indexed_segment_per_shard_gate"] = gate
    output = prepare_log(args, collection)
    stage14.write_json(output, payload)
    if gate["status"] != "PASS":
        raise RuntimeError(f"single-HNSW-segment gate failed: {output}")
    if (
        payload.get("deterministic_graph_construction") is not True
        or int(payload.get("hnsw_max_indexing_threads") or 0) != 1
        or int(payload.get("max_optimization_threads") or 0) != 1
    ):
        raise RuntimeError(f"deterministic prepare gate failed: {output}")


def run_e1(
    args: Namespace,
    *,
    dataset: str,
    method: str,
    shards: int,
    fanout: int,
    collection: str,
    label: str,
) -> dict[str, Any]:
    experiment_id = f"stage15-e1-{dataset}-{method}-m{shards}-{label}"
    output = e1_output(args, dataset, method, shards, label)
    namespace = Namespace(
        experiment_id=experiment_id,
        topology=str(args.topology),
        dataset=dataset,
        hdf5_path=str(stage14.dataset_path(args, dataset)),
        method=method,
        logical_shards=shards,
        partition_artifact=str(stage14.partition_path(args, dataset, method, shards)),
        collection=collection,
        fanout=fanout,
        ef_search=FIXED_EF[dataset],
        network_interface=args.network_interface,
        warmup_query_count=1_000,
        warmup_concurrency=16,
        sweep_query_count=2_000,
        measurement_query_count=18_000,
        concurrency_grid=[1, 2, 4, 8, 16, 32, 64, 128],
        formal_concurrency_override=None,
        formal_concurrency_override_reason=None,
        client_processes=128,
        request_workers=4,
        minimum_repetitions=3,
        maximum_repetitions=5,
        cv_threshold=0.05,
        progress_every=1_000,
        output_dir=str(args.output_root / "e1" / "raw" / experiment_id),
        output=str(output),
    )
    return execute_e1(namespace)


def run_point(
    args: Namespace,
    *,
    dataset: str,
    method: str,
    shards: int,
    fanout: int,
    label: str,
    collection: str,
) -> None:
    output = e1_output(args, dataset, method, shards, label)
    if (
        valid_e1(
            output,
            dataset=dataset,
            method=method,
            shards=shards,
            fanout=fanout,
            ef=FIXED_EF[dataset],
        )
        and prepare_valid(args, collection, shards)
        and stage14.cleanup_valid(args, collection)
    ):
        print(f"[stage15] resume complete: {output.name}", flush=True)
        return
    succeeded = False
    prepare_collection(args, dataset, method, shards, collection)
    try:
        run_e1(
            args,
            dataset=dataset,
            method=method,
            shards=shards,
            fanout=fanout,
            collection=collection,
            label=label,
        )
        if not valid_e1(
            output,
            dataset=dataset,
            method=method,
            shards=shards,
            fanout=fanout,
            ef=FIXED_EF[dataset],
        ):
            raise RuntimeError(f"Stage 15 E1 point is invalid: {output}")
        succeeded = True
    finally:
        if succeeded:
            stage14.cleanup_collection(args, collection)
        else:
            print(f"[stage15] retaining failed collection {collection}", file=sys.stderr)


def run_stage15_kmeans_point(
    args: Namespace,
    *,
    dataset: str,
    shards: int,
    label: str,
    collection: str,
) -> tuple[int, Path]:
    selection = fanout_selection_path(args, dataset, shards)
    if valid_stage15_fanout_selection(
        selection, dataset=dataset, shards=shards
    ):
        fanout = int(stage14.load_json(selection)["fanout"])
        output = e1_output(args, dataset, "kmeans", shards, label)
        if (
            valid_e1(
                output,
                dataset=dataset,
                method="kmeans",
                shards=shards,
                fanout=fanout,
                ef=FIXED_EF[dataset],
            )
            and prepare_valid(args, collection, shards)
            and stage14.cleanup_valid(args, collection)
        ):
            print(f"[stage15] resume complete: {output.name}", flush=True)
            return fanout, selection

    succeeded = False
    prepare_collection(args, dataset, "kmeans", shards, collection)
    try:
        fanout, selection = measure_stage15_kmeans_fanout(
            args, dataset=dataset, shards=shards, collection=collection
        )
        run_e1(
            args,
            dataset=dataset,
            method="kmeans",
            shards=shards,
            fanout=fanout,
            collection=collection,
            label=label,
        )
        output = e1_output(args, dataset, "kmeans", shards, label)
        if not valid_e1(
            output,
            dataset=dataset,
            method="kmeans",
            shards=shards,
            fanout=fanout,
            ef=FIXED_EF[dataset],
        ):
            raise RuntimeError(f"Stage 15 E1 point is invalid: {output}")
        succeeded = True
        return fanout, selection
    finally:
        if succeeded:
            stage14.cleanup_collection(args, collection)
        else:
            print(f"[stage15] retaining failed collection {collection}", file=sys.stderr)


def rewrite_record_type(path: Path, record_type: str) -> None:
    payload = stage14.load_json(path)
    payload["record_type"] = record_type
    stage14.write_json(path, payload)


def execute(args: Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    affinity_records: list[dict[str, Any]] = []
    body_error: BaseException | None = None
    try:
        affinity_records = stage14.apply_worker_affinity(args)
        rewrite_record_type(
            args.output_root / "worker-affinity-applied.json",
            "stage15_worker_affinity_apply",
        )
        stage14.preflight(args, affinity_records)
        rewrite_record_type(
            args.output_root / "preflight.json", "stage15_single_graph_preflight"
        )
        if args.preflight_only:
            print("[stage15] preflight-only PASS", flush=True)
            return 0
        fanout_sources: list[dict[str, Any]] = []
        for dataset in DATASET_ORDER:
            print(f"[stage15] dataset start: {dataset}", flush=True)
            run_point(
                args,
                dataset=dataset,
                method="random",
                shards=1,
                fanout=1,
                label="baseline-a",
                collection=collection_name(dataset, "common", 1, "baseline-a"),
            )
            for shards in PHYSICAL_COUNTS[1:]:
                if dataset == "glove-200-angular":
                    fanout, source = run_stage15_kmeans_point(
                        args,
                        dataset=dataset,
                        shards=shards,
                        label="primary",
                        collection=collection_name(dataset, "kmeans", shards),
                    )
                else:
                    fanout, source = stage14.selected_kmeans_fanout(dataset, shards)
                    run_point(
                        args,
                        dataset=dataset,
                        method="kmeans",
                        shards=shards,
                        fanout=fanout,
                        label="primary",
                        collection=collection_name(dataset, "kmeans", shards),
                    )
                fanout_sources.append(
                    {
                        "dataset": dataset,
                        "logical_shards": shards,
                        "fanout": fanout,
                        "fixed_ef": FIXED_EF[dataset],
                        "source": str(source.resolve()),
                        "source_sha256": sha256_path(source),
                    }
                )
                run_point(
                    args,
                    dataset=dataset,
                    method="random",
                    shards=shards,
                    fanout=shards,
                    label="primary",
                    collection=collection_name(dataset, "random", shards),
                )
            run_point(
                args,
                dataset=dataset,
                method="random",
                shards=1,
                fanout=1,
                label="baseline-b",
                collection=collection_name(dataset, "common", 1, "baseline-b"),
            )
            print(f"[stage15] dataset complete: {dataset}", flush=True)
        completion = {
            "timestamp": utc_timestamp(),
            "record_type": "stage15_single_hnsw_per_shard_retest",
            "status": "MEASUREMENTS_COMPLETE",
            "datasets": list(DATASET_ORDER),
            "physical_counts": list(PHYSICAL_COUNTS),
            "fixed_ef": FIXED_EF,
            "glove_single_graph_bounded_ef_grid": list(
                GLOVE_SINGLE_GRAPH_EF_GRID
            ),
            "glove_single_graph_ef_hard_cap": GLOVE_SINGLE_GRAPH_EF_HARD_CAP,
            "fanout_sources": fanout_sources,
            "max_segment_size_kb": args.max_segment_size_kb,
            "expected_total_segments_per_logical_shard": (
                EXPECTED_TOTAL_SEGMENTS_PER_LOGICAL_SHARD
            ),
            "expected_nonempty_hnsw_segments_per_logical_shard": 1,
            "topology": str(args.topology),
            "topology_sha256": sha256_path(args.topology),
        }
        stage14.write_json(args.output_root / "execution-complete.json", completion)
        print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        body_error = error
        raise
    finally:
        if affinity_records:
            try:
                stage14.restore_worker_affinity(args, affinity_records)
                rewrite_record_type(
                    args.output_root / "worker-affinity-restored.json",
                    "stage15_worker_affinity_restore",
                )
            except Exception:
                if body_error is None:
                    raise
                print("[stage15] worker affinity restore also failed", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-root", default="/users/dry/Orion/experiments/c1")
    parser.add_argument(
        "--output-root",
        default=(
            "/users/dry/Orion/experiments/c1/retests/"
            "stage15-single-hnsw-per-shard"
        ),
    )
    parser.add_argument(
        "--topology",
        default=(
            "/users/dry/Orion/experiments/c1/"
            "topology-amd-4node-physical-core-isolated.json"
        ),
    )
    parser.add_argument(
        "--base-partition-root",
        default="/users/dry/orion-distributed/c1-20260820-v1/partitions",
    )
    parser.add_argument(
        "--stage13-partition-root",
        default=(
            "/users/dry/orion-distributed/"
            "c1-stage13-physical1to4-ef-controlled/partitions"
        ),
    )
    parser.add_argument(
        "--controller-storage-root",
        default="/users/dry/orion-distributed/c1-20260821-v2/controller/storage",
    )
    parser.add_argument("--network-interface", default="enp65s0f0np0")
    parser.add_argument("--upload-batch-size", type=int, default=256)
    parser.add_argument(
        "--max-segment-size-kb", type=int, default=MAX_SEGMENT_SIZE_KB
    )
    parser.add_argument("--preflight-only", action="store_true")
    parsed = parser.parse_args(argv)
    for name in (
        "c1_root",
        "output_root",
        "topology",
        "base_partition_root",
        "stage13_partition_root",
        "controller_storage_root",
    ):
        setattr(parsed, name, Path(getattr(parsed, name)).expanduser().resolve())
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
