#!/usr/bin/env python3
"""Run the Stage 13 physical 1-4 node and controlled-ef fan-out retest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

from c1_benchmark import (
    HTTP,
    delete_collection as delete_collection_http,
    discover_nodes,
    ensure_collection,
    evaluate_kmeans_prefix_tuning_grid,
    evaluate_queries,
    load_topology,
    write_json_atomic,
    write_per_query_csv,
)
from c1_deterministic_rerun import verify_deleted
from c1_e1_scaleout import execute as execute_e1
from c1_protocol import DATASETS, TARGET_RECALL, partition_artifact, sha256_path, utc_timestamp


DATASET_ORDER = ("sift1m", "glove-200-angular")
PHYSICAL_COUNTS = (1, 2, 3, 4)
FANOUT_COUNTS = (1, 2, 3, 4, 8, 16, 32)
EF_GRIDS = {
    "sift1m": (16, 24, 32, 48, 64),
    "glove-200-angular": (64, 96, 128, 192),
}
FIXED_EF = {"sift1m": 24, "glove-200-angular": 192}
BENCHMARK_CPUS = tuple(range(20, 32))


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    write_json_atomic(path, payload)


def archive_partial(path: Path) -> None:
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archived = path.with_name(f"{path.name}.partial-{stamp}")
    path.replace(archived)


def dataset_token(dataset: str) -> str:
    return "glove" if dataset == "glove-200-angular" else dataset


def collection_name(dataset: str, method: str, shards: int, suffix: str = "") -> str:
    tail = f"_{suffix}" if suffix else ""
    return f"c1s13_{dataset_token(dataset)}_{method}_m{shards}{tail}"


def partition_path(args: Namespace, dataset: str, method: str, shards: int) -> Path:
    root = args.stage13_partition_root if shards == 3 else args.base_partition_root
    return root / dataset / f"{method}-m{shards}.npz"


def dataset_path(args: Namespace, dataset: str) -> Path:
    return Path(load_json(args.c1_root / "runs" / f"{dataset}.dataset.json")["path"])


def ensure_stage13_partitions(args: Namespace) -> None:
    for dataset in DATASET_ORDER:
        hdf5 = dataset_path(args, dataset)
        for method in ("random", "kmeans"):
            output = partition_path(args, dataset, method, 3)
            if output.is_file():
                print(f"[stage13] partition resume: {output}", flush=True)
                continue
            print(f"[stage13] build partition: {dataset} {method} M=3", flush=True)
            metadata = partition_artifact(
                hdf5,
                DATASETS[dataset],
                method,
                3,
                output,
            )
            write_json(
                args.output_root / "partitions" / dataset / f"{method}-m3.json",
                metadata,
            )


def prepare_collection(
    args: Namespace,
    dataset: str,
    method: str,
    shards: int,
    collection: str,
) -> dict[str, Any]:
    output = args.output_root / "logs" / f"{collection}-prepare.json"
    print(
        f"[stage13] prepare {collection}: dataset={dataset} method={method} M={shards}",
        flush=True,
    )
    payload = ensure_collection(
        args.topology,
        dataset_path(args, dataset),
        dataset,
        method,
        shards,
        partition_path(args, dataset, method, shards),
        collection,
        batch_size=args.upload_batch_size,
        replace=True,
    )
    write_json(output, payload)
    if (
        payload.get("deterministic_graph_construction") is not True
        or int(payload.get("hnsw_max_indexing_threads") or 0) != 1
        or int(payload.get("max_optimization_threads") or 0) != 1
    ):
        raise RuntimeError(f"deterministic prepare gate failed: {output}")
    return payload


def cleanup_collection(args: Namespace, collection: str) -> None:
    nodes = discover_nodes(load_topology(args.topology))
    delete_collection_http(nodes[0].base_url, collection)
    proof = verify_deleted(
        collection,
        controller_storage_root=args.controller_storage_root,
    )
    write_json(args.output_root / "cleanup" / f"{collection}.json", proof)
    print(f"[stage13] cleanup verified: {collection}", flush=True)


def tuning_summary(candidates: Sequence[dict[str, Any]], fixed_ef: int) -> dict[str, Any]:
    ef_values = sorted({int(row["ef_search"]) for row in candidates})
    by_ef: dict[str, Any] = {}
    for ef_search in ef_values:
        rows = [row for row in candidates if int(row["ef_search"]) == ef_search]
        feasible = [row for row in rows if float(row["recall_at_10"]) >= TARGET_RECALL]
        selected = min(feasible, key=lambda row: int(row["fanout"])) if feasible else None
        best_recall = max(rows, key=lambda row: float(row["recall_at_10"]))
        by_ef[str(ef_search)] = {
            "selected_minimum_fanout": int(selected["fanout"]) if selected else None,
            "selected_recall_at_10": float(selected["recall_at_10"]) if selected else None,
            "full_fanout_recall_at_10": float(
                max(rows, key=lambda row: int(row["fanout"]))["recall_at_10"]
            ),
            "maximum_recall_at_10": float(best_recall["recall_at_10"]),
            "status": "FEASIBLE" if selected else "INFEASIBLE_AT_RECALL_TARGET",
        }
    fixed = by_ef[str(fixed_ef)]
    return {
        "ef_sensitivity": by_ef,
        "fixed_ef": fixed_ef,
        "fixed_ef_selected_fanout": fixed["selected_minimum_fanout"],
        "fixed_ef_tuning_recall": fixed["selected_recall_at_10"],
        "fixed_ef_status": fixed["status"],
    }


def run_kmeans_tuning(
    args: Namespace,
    dataset: str,
    shards: int,
    collection: str,
) -> dict[str, Any]:
    stem = f"{dataset}-kmeans-m{shards}"
    output = args.output_root / "fanout" / "tuning" / f"{stem}.json"
    if output.is_file():
        payload = load_json(output)
        if (
            payload.get("fixed_ef") == FIXED_EF[dataset]
            and payload.get("partition_sha256")
            == sha256_path(partition_path(args, dataset, "kmeans", shards))
        ):
            print(f"[stage13] tuning resume: {stem}", flush=True)
            return payload
    manifest = output.with_suffix(".manifest.jsonl")
    archive_partial(manifest)
    candidates = evaluate_kmeans_prefix_tuning_grid(
        args.topology,
        dataset_path(args, dataset),
        partition_path(args, dataset, "kmeans", shards),
        dataset,
        shards,
        collection,
        EF_GRIDS[dataset],
        1_000,
        query_concurrency=args.query_concurrency,
        request_workers=args.request_workers,
        require_worker_cpu_time=True,
        require_benchmark_affinity=True,
        experiment_id=f"stage13-controlled-ef-{stem}",
        manifest_jsonl=manifest,
    )
    summary = tuning_summary(candidates, FIXED_EF[dataset])
    payload = {
        "timestamp": utc_timestamp(),
        "record_type": "stage13_controlled_ef_kmeans_tuning",
        "dataset": dataset,
        "partition_method": "kmeans",
        "logical_shard_count": shards,
        "physical_machine_count": min(shards, 4),
        "physical_scale_point": shards <= 4,
        "ef_grid": list(EF_GRIDS[dataset]),
        "ef_hard_cap": max(EF_GRIDS[dataset]),
        "selection_rule": "minimum_fanout_at_dataset_fixed_ef",
        "partition_artifact": str(partition_path(args, dataset, "kmeans", shards)),
        "partition_sha256": sha256_path(
            partition_path(args, dataset, "kmeans", shards)
        ),
        "candidate_manifest": str(manifest),
        "candidate_manifest_sha256": sha256_path(manifest),
        "candidates": list(candidates),
        **summary,
    }
    write_json(output, payload)
    return payload


def valid_holdout(path: Path, *, dataset: str, method: str, shards: int, ef: int) -> bool:
    if not path.is_file():
        return False
    payload = load_json(path)
    return (
        payload.get("dataset") == dataset
        and payload.get("partition_method") == method
        and int(payload.get("logical_shards") or 0) == shards
        and int(payload.get("ef_search") or 0) == ef
        and float(payload.get("achieved_recall") or 0.0) >= TARGET_RECALL
        and payload.get("result_status") == "VALID"
    )


def measure_holdout(
    args: Namespace,
    *,
    dataset: str,
    method: str,
    shards: int,
    collection: str,
    initial_fanout: int,
    ef_search: int,
) -> dict[str, Any]:
    stem = f"{dataset}-{method}-m{shards}-ef{ef_search}"
    selection_path = args.output_root / "fanout" / "holdout" / f"{stem}-selected.json"
    if valid_holdout(
        selection_path,
        dataset=dataset,
        method=method,
        shards=shards,
        ef=ef_search,
    ):
        print(f"[stage13] holdout resume: {stem}", flush=True)
        return load_json(selection_path)

    attempts: list[dict[str, Any]] = []
    fanouts = [shards] if method == "random" else list(range(initial_fanout, shards + 1))
    for fanout in fanouts:
        print(
            f"[stage13] holdout {stem}: P={fanout} ef={ef_search}",
            flush=True,
        )
        rows, summary = evaluate_queries(
            args.topology,
            dataset_path(args, dataset),
            partition_path(args, dataset, method, shards),
            dataset,
            method,
            shards,
            collection,
            fanout,
            ef_search,
            1_000,
            9_000,
            query_concurrency=args.query_concurrency,
            request_workers=args.request_workers,
            require_worker_cpu_time=True,
            require_benchmark_affinity=True,
        )
        per_query = (
            args.output_root
            / "fanout"
            / "holdout"
            / "per_query"
            / f"{stem}-p{fanout}.csv"
        )
        write_per_query_csv(per_query, rows)
        summary.update(
            {
                "record_type": "stage13_controlled_ef_holdout",
                "per_query": str(per_query),
                "per_query_sha256": sha256_path(per_query),
                "selection_rule": "increase_fanout_only_while_ef_is_fixed",
            }
        )
        attempt_path = (
            args.output_root / "fanout" / "holdout" / f"{stem}-p{fanout}.json"
        )
        write_json(attempt_path, summary)
        attempts.append(
            {
                "fanout": fanout,
                "achieved_recall": float(summary["achieved_recall"]),
                "result_status": summary["result_status"],
                "summary": str(attempt_path),
            }
        )
        if float(summary["achieved_recall"]) >= TARGET_RECALL:
            summary["holdout_attempts"] = attempts
            summary["selected_after_holdout"] = True
            write_json(selection_path, summary)
            return summary

    failure = {
        "timestamp": utc_timestamp(),
        "record_type": "stage13_controlled_ef_holdout_selection",
        "dataset": dataset,
        "partition_method": method,
        "logical_shards": shards,
        "ef_search": ef_search,
        "holdout_attempts": attempts,
        "result_status": "INFEASIBLE_AT_FIXED_EF",
        "selected_after_holdout": False,
    }
    write_json(selection_path, failure)
    raise RuntimeError(f"full fan-out misses recall target at fixed ef: {stem}")


def valid_e1(path: Path, *, dataset: str, method: str, shards: int, fanout: int, ef: int) -> bool:
    if not path.is_file():
        return False
    payload = load_json(path)
    return (
        payload.get("status") == "VALID_E1"
        and payload.get("dataset") == dataset
        and payload.get("partition_method") == method
        and int(payload.get("logical_shard_count") or 0) == shards
        and int(payload.get("physical_machine_count") or 0) == shards
        and int(payload.get("fanout") or 0) == fanout
        and int(payload.get("ef_search") or 0) == ef
    )


def run_e1(
    args: Namespace,
    *,
    dataset: str,
    method: str,
    shards: int,
    fanout: int,
    ef_search: int,
    collection: str,
    label: str,
) -> dict[str, Any]:
    experiment_id = f"stage13-e1-{dataset}-{method}-m{shards}-{label}"
    output = args.output_root / "e1" / f"{experiment_id}.json"
    if valid_e1(
        output,
        dataset=dataset,
        method=method,
        shards=shards,
        fanout=fanout,
        ef=ef_search,
    ):
        print(f"[stage13] E1 resume: {experiment_id}", flush=True)
        return load_json(output)
    print(
        f"[stage13] E1 {experiment_id}: physical M={shards} P={fanout} ef={ef_search}",
        flush=True,
    )
    namespace = Namespace(
        experiment_id=experiment_id,
        topology=str(args.topology),
        dataset=dataset,
        hdf5_path=str(dataset_path(args, dataset)),
        method=method,
        logical_shards=shards,
        partition_artifact=str(partition_path(args, dataset, method, shards)),
        collection=collection,
        fanout=fanout,
        ef_search=ef_search,
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


def run_m1_group(args: Namespace, dataset: str, *, baseline_label: str) -> None:
    collection = collection_name(dataset, "common", 1, baseline_label)
    succeeded = False
    prepare_collection(args, dataset, "random", 1, collection)
    try:
        if baseline_label == "baseline-a":
            tuning = run_kmeans_tuning(args, dataset, 1, collection)
            initial = int(tuning["fixed_ef_selected_fanout"] or 1)
            measure_holdout(
                args,
                dataset=dataset,
                method="kmeans",
                shards=1,
                collection=collection,
                initial_fanout=initial,
                ef_search=FIXED_EF[dataset],
            )
        run_e1(
            args,
            dataset=dataset,
            method="random",
            shards=1,
            fanout=1,
            ef_search=FIXED_EF[dataset],
            collection=collection,
            label=baseline_label,
        )
        succeeded = True
    finally:
        if succeeded:
            cleanup_collection(args, collection)
        else:
            print(f"[stage13] retaining failed collection {collection}", file=sys.stderr)


def run_kmeans_group(args: Namespace, dataset: str, shards: int, *, with_e1: bool) -> None:
    collection = collection_name(dataset, "kmeans", shards)
    succeeded = False
    prepare_collection(args, dataset, "kmeans", shards, collection)
    try:
        tuning = run_kmeans_tuning(args, dataset, shards, collection)
        initial = tuning.get("fixed_ef_selected_fanout")
        if initial is None:
            initial = shards
        holdout = measure_holdout(
            args,
            dataset=dataset,
            method="kmeans",
            shards=shards,
            collection=collection,
            initial_fanout=int(initial),
            ef_search=FIXED_EF[dataset],
        )
        if with_e1:
            run_e1(
                args,
                dataset=dataset,
                method="kmeans",
                shards=shards,
                fanout=int(holdout["fanout"]),
                ef_search=FIXED_EF[dataset],
                collection=collection,
                label="primary",
            )
        succeeded = True
    finally:
        if succeeded:
            cleanup_collection(args, collection)
        else:
            print(f"[stage13] retaining failed collection {collection}", file=sys.stderr)


def run_random_group(args: Namespace, dataset: str, shards: int) -> None:
    collection = collection_name(dataset, "random", shards)
    succeeded = False
    prepare_collection(args, dataset, "random", shards, collection)
    try:
        measure_holdout(
            args,
            dataset=dataset,
            method="random",
            shards=shards,
            collection=collection,
            initial_fanout=shards,
            ef_search=FIXED_EF[dataset],
        )
        run_e1(
            args,
            dataset=dataset,
            method="random",
            shards=shards,
            fanout=shards,
            ef_search=FIXED_EF[dataset],
            collection=collection,
            label="primary",
        )
        succeeded = True
    finally:
        if succeeded:
            cleanup_collection(args, collection)
        else:
            print(f"[stage13] retaining failed collection {collection}", file=sys.stderr)


def preflight(args: Namespace) -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != list(BENCHMARK_CPUS):
        raise RuntimeError(
            f"run Stage 13 with taskset -c 20-31; current affinity is {affinity}"
        )
    topology = load_topology(args.topology)
    nodes = discover_nodes(topology)
    if len(nodes) != 4:
        raise RuntimeError(f"expected four live nodes, got {len(nodes)}")
    visible_collections: dict[str, list[str]] = {}
    collections = []
    for node in nodes:
        collections.append({"host": node.private_ip, "peer_id": node.peer_id})
        payload, _ = HTTP.request(node.base_url, "GET", "/collections", timeout=30.0)
        names = sorted(
            str(row["name"])
            for row in ((payload.get("result") or {}).get("collections") or [])
        )
        visible_collections[node.private_ip] = names
    if any(visible_collections.values()):
        raise RuntimeError(
            f"preflight requires an empty cluster, found {visible_collections}"
        )
    payload = {
        "timestamp": utc_timestamp(),
        "record_type": "stage13_preflight",
        "benchmark_cpu_affinity": affinity,
        "fixed_ef": FIXED_EF,
        "ef_grids": {key: list(value) for key, value in EF_GRIDS.items()},
        "physical_counts": list(PHYSICAL_COUNTS),
        "fanout_counts": list(FANOUT_COUNTS),
        "nodes": collections,
        "visible_collections": visible_collections,
        "topology": str(args.topology),
        "topology_sha256": sha256_path(args.topology),
        "status": "PASS",
    }
    write_json(args.output_root / "preflight.json", payload)
    return payload


def execute(args: Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    preflight(args)
    if args.preflight_only:
        print("[stage13] preflight-only PASS", flush=True)
        return 0
    ensure_stage13_partitions(args)
    if args.partitions_only:
        print("[stage13] partitions-only complete", flush=True)
        return 0
    for dataset in DATASET_ORDER:
        print(f"[stage13] dataset start: {dataset}", flush=True)
        run_m1_group(args, dataset, baseline_label="baseline-a")
        for shards in PHYSICAL_COUNTS[1:]:
            run_kmeans_group(args, dataset, shards, with_e1=True)
            run_random_group(args, dataset, shards)
        for shards in (8, 16, 32):
            run_kmeans_group(args, dataset, shards, with_e1=False)
        run_m1_group(args, dataset, baseline_label="baseline-b")
        print(f"[stage13] dataset complete: {dataset}", flush=True)
    completion = {
        "timestamp": utc_timestamp(),
        "record_type": "stage13_retest_execution",
        "datasets": list(DATASET_ORDER),
        "physical_counts": list(PHYSICAL_COUNTS),
        "fanout_counts": list(FANOUT_COUNTS),
        "fixed_ef": FIXED_EF,
        "status": "MEASUREMENTS_COMPLETE",
    }
    write_json(args.output_root / "execution-complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/users/dry/Orion")
    parser.add_argument("--c1-root", default="/users/dry/Orion/experiments/c1")
    parser.add_argument(
        "--output-root",
        default=(
            "/users/dry/Orion/experiments/c1/retests/"
            "stage13-physical1to4-ef-controlled"
        ),
    )
    parser.add_argument(
        "--topology", default="/users/dry/Orion/experiments/c1/topology-amd-4node.json"
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
    parser.add_argument("--query-concurrency", type=int, default=16)
    parser.add_argument("--request-workers", type=int, default=128)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--partitions-only", action="store_true")
    parsed = parser.parse_args(argv)
    for name in (
        "repo",
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
