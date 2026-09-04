#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Sequence


DATASETS = ("sift1m", "glove-200-angular")
METHODS = ("random", "kmeans")
LOGICAL_SHARDS = (1, 2, 4, 8, 16, 32)
SOURCE_STEM = "stage12-e3-deterministic"
BENCHMARK_CPUS = "20-31"
PEERS = ("10.10.1.1", "10.10.1.2", "10.10.1.3", "10.10.1.4")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_command(command: list[str], *, cwd: Path) -> None:
    print("[c1-rerun]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def collection_name(dataset: str, method: str, shards: int) -> str:
    token = dataset.replace("glove-200-angular", "glove")
    return f"c1v2_e3_{token}_{method}_m{shards}"


def artifact_stem(dataset: str, method: str, shards: int) -> str:
    return f"{SOURCE_STEM}-{dataset}-{method}-m{shards}"


def http_status(peer: str, collection: str) -> int:
    encoded = urllib.parse.quote(collection, safe="")
    request = urllib.request.Request(
        f"http://{peer}:6333/collections/{encoded}", method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=15.0) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def delete_collection(collection: str) -> None:
    encoded = urllib.parse.quote(collection, safe="")
    request = urllib.request.Request(
        f"http://{PEERS[0]}:6333/collections/{encoded}", method="DELETE"
    )
    try:
        with urllib.request.urlopen(request, timeout=300.0) as response:
            if int(response.status) not in {200, 202}:
                raise RuntimeError(
                    f"unexpected delete status for {collection}: {response.status}"
                )
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise


def verify_deleted(
    collection: str,
    *,
    controller_storage_root: Path,
    timeout: float = 180.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, int] = {}
    collection_storage = controller_storage_root / "collections" / collection
    while time.monotonic() < deadline:
        last = {peer: http_status(peer, collection) for peer in PEERS}
        if all(status == 404 for status in last.values()) and not collection_storage.exists():
            return {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "collection": collection,
                "peers": {
                    peer: {"http_status": status} for peer, status in last.items()
                },
                "controller_storage_path": str(collection_storage),
                "controller_storage_absent": True,
                "status": "VERIFIED_DELETED",
            }
        time.sleep(1.0)
    raise TimeoutError(
        f"collection cleanup did not converge: collection={collection}, "
        f"peer_status={last}, storage_exists={collection_storage.exists()}"
    )


def valid_completed_summary(path: Path, graph_build_seed: int) -> bool:
    if not path.is_file():
        return False
    summary = load_json(path)
    return (
        summary.get("result_status") == "VALID_E3"
        and float(summary.get("measurement_recall") or 0.0) >= 0.90
        and int(summary.get("graph_build_seed") or -1) == graph_build_seed
        and int(summary.get("hnsw_max_indexing_threads") or 0) == 1
        and int(summary.get("max_optimization_threads") or 0) == 1
        and summary.get("deterministic_graph_construction") is True
    )


def selected_tuning(path: Path, graph_build_seed: int) -> dict[str, Any]:
    tuning = load_json(path)
    selected = dict(tuning["selected"])
    candidates = list(tuning.get("candidates") or [])
    if not candidates:
        raise ValueError(f"tuning artifact has no candidates: {path}")
    if any(int(row.get("graph_build_seed") or -1) != graph_build_seed for row in candidates):
        raise ValueError(f"tuning artifact has the wrong graph-build seed: {path}")
    if float(selected["recall_at_10"]) < 0.90:
        raise ValueError(f"selected tuning point misses Recall@10 target: {path}")
    return selected


def execute(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    runs = root / "runs"
    per_query = runs / "per_query"
    logs = root / "logs"
    topology = load_json(Path(args.topology).expanduser().resolve())
    graph_build_seed = int(topology["hnsw_graph_build_seed"])
    # Preserve the virtual-environment launcher instead of resolving its
    # symlink to the system interpreter, which would drop the venv packages.
    python = str(Path(args.python).expanduser().absolute())
    benchmark = str(root / "scripts" / "c1_benchmark.py")
    manifest = str(runs / "manifest.jsonl")
    partition_root = Path(args.partition_root).expanduser().resolve()
    controller_storage_root = (
        Path(topology["local_storage_root"])
        / args.run_id
        / "controller"
        / "storage"
    )
    dataset_paths = {
        dataset: Path(load_json(runs / f"{dataset}.dataset.json")["path"])
        for dataset in DATASETS
    }
    cleanup_records: list[dict[str, Any]] = []

    groups: list[tuple[str, int, tuple[str, ...]]] = []
    for dataset in DATASETS:
        groups.append((dataset, 1, METHODS))
        for shards in LOGICAL_SHARDS[1:]:
            for method in METHODS:
                groups.append((dataset, shards, (method,)))

    for dataset, shards, methods in groups:
        pending = []
        for method in methods:
            stem = artifact_stem(dataset, method, shards)
            summary_path = runs / f"{stem}-summary.json"
            if valid_completed_summary(summary_path, graph_build_seed):
                print(f"[c1-rerun] resume skip {stem}: verified VALID_E3", flush=True)
            else:
                pending.append(method)
        if not pending:
            continue

        build_method = "random" if shards == 1 else pending[0]
        collection = collection_name(dataset, build_method, shards)
        build_partition = partition_root / dataset / f"{build_method}-m{shards}.npz"
        prepare_output = logs / f"{artifact_stem(dataset, build_method, shards)}-prepare.json"
        run_command(
            [
                "taskset", "-c", BENCHMARK_CPUS, python, benchmark, "prepare",
                "--topology", args.topology,
                "--dataset", dataset,
                "--hdf5-path", str(dataset_paths[dataset]),
                "--method", build_method,
                "--logical-shards", str(shards),
                "--partition-artifact", str(build_partition),
                "--collection", collection,
                "--upload-batch-size", str(args.upload_batch_size),
                "--replace",
                "--output", str(prepare_output),
            ],
            cwd=repo,
        )
        prepare = load_json(prepare_output)
        if (
            int(prepare["graph_build_seed"]) != graph_build_seed
            or int(prepare["hnsw_max_indexing_threads"]) != 1
            or int(prepare["max_optimization_threads"]) != 1
            or prepare["deterministic_graph_construction"] is not True
        ):
            raise RuntimeError(f"deterministic prepare gate failed: {prepare_output}")

        group_succeeded = False
        try:
            for method in pending:
                stem = artifact_stem(dataset, method, shards)
                partition = partition_root / dataset / f"{method}-m{shards}.npz"
                tuning_path = runs / f"{stem}-tuning-pinned.json"
                if not tuning_path.is_file():
                    tune_command = [
                        "taskset", "-c", BENCHMARK_CPUS, python, benchmark, "tune",
                        "--topology", args.topology,
                        "--dataset", dataset,
                        "--hdf5-path", str(dataset_paths[dataset]),
                        "--method", method,
                        "--logical-shards", str(shards),
                        "--partition-artifact", str(partition),
                        "--collection", collection,
                        "--experiment-id", f"{stem}-tune-q1000",
                        "--tuning-query-count", "1000",
                        "--query-concurrency", str(args.query_concurrency),
                        "--request-workers", str(args.request_workers),
                        "--require-worker-cpu-time",
                        "--require-benchmark-affinity",
                        "--manifest-jsonl", manifest,
                        "--output", str(tuning_path),
                    ]
                    if method == "kmeans":
                        tune_command.append("--kmeans-prefix-reuse")
                    run_command(tune_command, cwd=repo)
                selected = selected_tuning(tuning_path, graph_build_seed)
                summary_path = runs / f"{stem}-summary.json"
                per_search_path = per_query / f"{stem}.csv"
                run_command(
                    [
                        "taskset", "-c", BENCHMARK_CPUS, python, benchmark, "e3-measure",
                        "--topology", args.topology,
                        "--dataset", dataset,
                        "--hdf5-path", str(dataset_paths[dataset]),
                        "--method", method,
                        "--logical-shards", str(shards),
                        "--partition-artifact", str(partition),
                        "--collection", collection,
                        "--fanout", str(int(selected["fanout"])),
                        "--ef-search", str(int(selected["ef_search"])),
                        "--selected-tuning-recall", str(float(selected["recall_at_10"])),
                        "--query-start", "1000",
                        "--query-count", "9000",
                        "--require-worker-cpu-time",
                        "--require-benchmark-affinity",
                        "--run-id", args.run_id,
                        "--experiment-id", f"{stem}-measure-q9000",
                        "--manifest-jsonl", manifest,
                        "--per-search-output", str(per_search_path),
                        "--summary-output", str(summary_path),
                    ],
                    cwd=repo,
                )
                if not valid_completed_summary(summary_path, graph_build_seed):
                    raise RuntimeError(f"formal E3 validation failed: {summary_path}")
            group_succeeded = True
        finally:
            if group_succeeded:
                delete_collection(collection)
                cleanup = verify_deleted(
                    collection,
                    controller_storage_root=controller_storage_root,
                )
                cleanup_records.append(cleanup)
                write_json_atomic(
                    runs / f"{artifact_stem(dataset, build_method, shards)}-cleanup.json",
                    cleanup,
                )
            else:
                print(
                    f"[c1-rerun] retaining {collection} because the group did not complete",
                    file=sys.stderr,
                    flush=True,
                )

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "record_type": "c1_deterministic_e3_matrix",
        "source_stem": SOURCE_STEM,
        "run_id": args.run_id,
        "graph_build_seed": graph_build_seed,
        "max_indexing_threads": 1,
        "max_optimization_threads": 1,
        "configuration_count": len(DATASETS) * len(METHODS) * len(LOGICAL_SHARDS),
        "cleanup_records_created_this_invocation": len(cleanup_records),
        "status": "COMPLETE",
    }
    write_json_atomic(runs / f"{SOURCE_STEM}-matrix-record.json", output)
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic C1 E3 matrix")
    parser.add_argument("--repo", default="/users/dry/Orion")
    parser.add_argument("--root", default="experiments/c1")
    parser.add_argument("--topology", default="experiments/c1/topology-amd-4node.json")
    parser.add_argument("--run-id", default="c1-20260821-v2")
    parser.add_argument(
        "--partition-root",
        default="/users/dry/orion-distributed/c1-20260820-v1/partitions",
    )
    parser.add_argument("--python", default="/users/dry/orion-distributed/venv/bin/python")
    parser.add_argument("--upload-batch-size", type=int, default=256)
    parser.add_argument("--query-concurrency", type=int, default=16)
    parser.add_argument("--request-workers", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
