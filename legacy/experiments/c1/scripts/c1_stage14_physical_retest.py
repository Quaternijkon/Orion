#!/usr/bin/env python3
"""Run the physical-core-isolated Stage 14 physical 1-4 host QPS retest."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

from c1_benchmark import (
    HTTP,
    delete_collection as delete_collection_http,
    discover_nodes,
    ensure_collection,
    load_topology,
    parse_cpu_set,
    write_json_atomic,
)
from c1_deterministic_rerun import verify_deleted
from c1_e1_scaleout import execute as execute_e1
from c1_protocol import DATASETS, sha256_path, utc_timestamp


DATASET_ORDER = ("sift1m", "glove-200-angular")
PHYSICAL_COUNTS = (1, 2, 3, 4)
FIXED_EF = {"sift1m": 24, "glove-200-angular": 192}
STAGE13_DIR = Path(
    "/users/dry/Orion/experiments/c1/retests/stage13-physical1to4-ef-controlled"
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    write_json_atomic(path, payload)


def dataset_token(dataset: str) -> str:
    return "glove" if dataset == "glove-200-angular" else dataset


def collection_name(dataset: str, method: str, shards: int, suffix: str = "") -> str:
    tail = f"_{suffix}" if suffix else ""
    return f"c1s14_{dataset_token(dataset)}_{method}_m{shards}{tail}"


def dataset_path(args: Namespace, dataset: str) -> Path:
    return Path(load_json(args.c1_root / "runs" / f"{dataset}.dataset.json")["path"])


def partition_path(args: Namespace, dataset: str, method: str, shards: int) -> Path:
    root = args.stage13_partition_root if shards == 3 else args.base_partition_root
    return root / dataset / f"{method}-m{shards}.npz"


def selected_kmeans_fanout(dataset: str, shards: int) -> tuple[int, Path]:
    ef = FIXED_EF[dataset]
    path = (
        STAGE13_DIR
        / "fanout"
        / "holdout"
        / f"{dataset}-kmeans-m{shards}-ef{ef}-selected.json"
    )
    payload = load_json(path)
    if (
        payload.get("selected_after_holdout") is not True
        or payload.get("result_status") != "VALID"
        or float(payload.get("achieved_recall") or 0.0) < 0.90
        or int(payload.get("ef_search") or 0) != ef
        or int(payload.get("logical_shards") or 0) != shards
    ):
        raise ValueError(f"invalid Stage 13 held-out fan-out selection: {path}")
    return int(payload["fanout"]), path


def remote_command(ssh_host: str, command: str) -> str:
    if ssh_host in {"localhost", "127.0.0.1", "::1"}:
        argv = ["bash", "-lc", command]
    else:
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            ssh_host,
            "bash -lc " + shlex.quote(command),
        ]
    return subprocess.check_output(argv, text=True, timeout=60).strip()


def container_state(ssh_host: str) -> dict[str, str]:
    command = r'''
set -euo pipefail
ids=$(sudo -n docker ps -q)
test "$(printf '%s\n' "$ids" | sed '/^$/d' | wc -l)" -eq 1
id=$(printf '%s\n' "$ids" | sed '/^$/d')
sudo -n docker inspect -f '{{.Id}}|{{.Name}}|{{.Config.Image}}|{{.HostConfig.CpusetCpus}}' "$id"
'''.strip()
    raw = remote_command(ssh_host, command)
    container_id, name, image, cpuset = raw.split("|", 3)
    return {
        "container_id": container_id,
        "container_name": name.removeprefix("/"),
        "image": image,
        "cpuset": cpuset,
    }


def runtime_qdrant_affinity(ssh_host: str) -> str:
    command = r'''
set -euo pipefail
pid=$(pgrep -x qdrant)
test "$(printf '%s\n' "$pid" | sed '/^$/d' | wc -l)" -eq 1
awk '/Cpus_allowed_list:/ {print $2}' "/proc/$pid/status"
'''.strip()
    return remote_command(ssh_host, command)


def update_container_cpuset(ssh_host: str, container_id: str, cpuset: str) -> None:
    command = (
        "set -euo pipefail; sudo -n docker update --cpuset-cpus "
        + shlex.quote(cpuset)
        + " "
        + shlex.quote(container_id)
        + " >/dev/null"
    )
    remote_command(ssh_host, command)


def apply_worker_affinity(args: Namespace) -> list[dict[str, Any]]:
    topology = load_topology(args.topology)
    records: list[dict[str, Any]] = []
    try:
        for node in [topology["controller"], *topology["workers"]]:
            ssh_host = str(node["ssh_host"])
            expected = str(node["cpuset"])
            before = container_state(ssh_host)
            record: dict[str, Any] = {
                "ssh_host": ssh_host,
                "private_ip": str(node["private_ip"]),
                "before": before,
            }
            records.append(record)
            update_container_cpuset(ssh_host, before["container_id"], expected)
            after = container_state(ssh_host)
            runtime = runtime_qdrant_affinity(ssh_host)
            if after["cpuset"] != expected or runtime != expected:
                raise RuntimeError(
                    f"failed to apply worker affinity on {ssh_host}: "
                    f"docker={after['cpuset']} runtime={runtime} expected={expected}"
                )
            record.update(
                {
                    "after": after,
                    "runtime_qdrant_affinity": runtime,
                }
            )
    except Exception:
        restore_worker_affinity(args, records)
        raise
    payload = {
        "timestamp": utc_timestamp(),
        "record_type": "stage14_worker_affinity_apply",
        "status": "PASS",
        "nodes": records,
    }
    write_json(args.output_root / "worker-affinity-applied.json", payload)
    return records


def restore_worker_affinity(args: Namespace, records: Sequence[dict[str, Any]]) -> None:
    restored: list[dict[str, Any]] = []
    failures: list[str] = []
    for record in records:
        ssh_host = str(record["ssh_host"])
        before = dict(record["before"])
        try:
            update_container_cpuset(
                ssh_host, str(before["container_id"]), str(before["cpuset"])
            )
            after = container_state(ssh_host)
            runtime = runtime_qdrant_affinity(ssh_host)
            ok = after["cpuset"] == before["cpuset"] and runtime == before["cpuset"]
            if not ok:
                failures.append(ssh_host)
            restored.append(
                {
                    "ssh_host": ssh_host,
                    "private_ip": record["private_ip"],
                    "restored": after,
                    "runtime_qdrant_affinity": runtime,
                    "expected_cpuset": before["cpuset"],
                    "valid": ok,
                }
            )
        except Exception as error:  # preserve evidence even if one host fails
            failures.append(ssh_host)
            restored.append(
                {
                    "ssh_host": ssh_host,
                    "private_ip": record["private_ip"],
                    "expected_cpuset": before["cpuset"],
                    "valid": False,
                    "error": repr(error),
                }
            )
    write_json(
        args.output_root / "worker-affinity-restored.json",
        {
            "timestamp": utc_timestamp(),
            "record_type": "stage14_worker_affinity_restore",
            "status": "PASS" if not failures else "FAIL",
            "nodes": restored,
        },
    )
    if failures:
        raise RuntimeError(f"failed to restore worker affinity on: {failures}")


def local_cpu_topology() -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for cpu_path in sorted(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*"),
        key=lambda path: int(path.name[3:]),
    ):
        cpu = int(cpu_path.name[3:])
        core = int((cpu_path / "topology" / "core_id").read_text().strip())
        socket = int((cpu_path / "topology" / "physical_package_id").read_text().strip())
        rows.append({"cpu": cpu, "core": core, "socket": socket})
    return rows


def preflight(args: Namespace, affinity_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    topology = load_topology(args.topology)
    expected_benchmark = parse_cpu_set(str(topology["benchmark_client_cpuset"]))
    actual_benchmark = sorted(os.sched_getaffinity(0))
    if actual_benchmark != expected_benchmark:
        raise RuntimeError(
            f"benchmark affinity mismatch: actual={actual_benchmark}, "
            f"expected={expected_benchmark}"
        )
    cpu_rows = local_cpu_topology()
    core_for_cpu = {row["cpu"]: (row["socket"], row["core"]) for row in cpu_rows}
    worker_cpus = parse_cpu_set(str(topology["controller"]["cpuset"]))
    worker_cores = {core_for_cpu[cpu] for cpu in worker_cpus}
    benchmark_cores = {core_for_cpu[cpu] for cpu in expected_benchmark}
    overlap = sorted(worker_cores & benchmark_cores)
    if overlap:
        raise RuntimeError(f"worker and benchmark share physical cores: {overlap}")
    nodes = discover_nodes(topology)
    visible_collections: dict[str, list[str]] = {}
    for node in nodes:
        payload, _ = HTTP.request(node.base_url, "GET", "/collections", timeout=30.0)
        visible_collections[node.private_ip] = sorted(
            str(row["name"])
            for row in ((payload.get("result") or {}).get("collections") or [])
        )
    if any(visible_collections.values()):
        raise RuntimeError(f"preflight requires an empty cluster: {visible_collections}")
    payload = {
        "timestamp": utc_timestamp(),
        "record_type": "stage14_preflight",
        "status": "PASS",
        "benchmark_cpu_affinity": actual_benchmark,
        "benchmark_physical_cores": sorted(benchmark_cores),
        "worker_cpu_affinity": worker_cpus,
        "worker_physical_cores": sorted(worker_cores),
        "physical_core_overlap": overlap,
        "cpu_topology": cpu_rows,
        "worker_affinity_records": list(affinity_records),
        "fixed_ef": FIXED_EF,
        "physical_counts": list(PHYSICAL_COUNTS),
        "visible_collections": visible_collections,
        "topology": str(args.topology),
        "topology_sha256": sha256_path(args.topology),
    }
    write_json(args.output_root / "preflight.json", payload)
    return payload


def prepare_collection(
    args: Namespace, dataset: str, method: str, shards: int, collection: str
) -> None:
    print(
        f"[stage14] prepare {collection}: dataset={dataset} method={method} M={shards}",
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
    output = args.output_root / "logs" / f"{collection}-prepare.json"
    write_json(output, payload)
    if (
        payload.get("deterministic_graph_construction") is not True
        or int(payload.get("hnsw_max_indexing_threads") or 0) != 1
        or int(payload.get("max_optimization_threads") or 0) != 1
    ):
        raise RuntimeError(f"deterministic prepare gate failed: {output}")


def cleanup_collection(args: Namespace, collection: str) -> None:
    nodes = discover_nodes(load_topology(args.topology))
    delete_collection_http(nodes[0].base_url, collection)
    proof = verify_deleted(collection, controller_storage_root=args.controller_storage_root)
    write_json(args.output_root / "cleanup" / f"{collection}.json", proof)
    print(f"[stage14] cleanup verified: {collection}", flush=True)


def e1_output(args: Namespace, dataset: str, method: str, shards: int, label: str) -> Path:
    return args.output_root / "e1" / f"stage14-e1-{dataset}-{method}-m{shards}-{label}.json"


def valid_e1(
    path: Path, *, dataset: str, method: str, shards: int, fanout: int, ef: int
) -> bool:
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


def cleanup_valid(args: Namespace, collection: str) -> bool:
    path = args.output_root / "cleanup" / f"{collection}.json"
    if not path.is_file():
        return False
    payload = load_json(path)
    return (
        payload.get("status") == "VERIFIED_DELETED"
        and payload.get("controller_storage_absent") is True
        and len(payload.get("peers") or {}) == 4
        and {int(row["http_status"]) for row in payload["peers"].values()} == {404}
        and not Path(payload["controller_storage_path"]).exists()
    )


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
    experiment_id = f"stage14-e1-{dataset}-{method}-m{shards}-{label}"
    output = e1_output(args, dataset, method, shards, label)
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
    if valid_e1(
        output,
        dataset=dataset,
        method=method,
        shards=shards,
        fanout=fanout,
        ef=FIXED_EF[dataset],
    ) and cleanup_valid(args, collection):
        print(f"[stage14] resume complete: {output.name}", flush=True)
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
            raise RuntimeError(f"Stage 14 E1 point is invalid: {output}")
        succeeded = True
    finally:
        if succeeded:
            cleanup_collection(args, collection)
        else:
            print(f"[stage14] retaining failed collection {collection}", file=sys.stderr)


def execute(args: Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    affinity_records: list[dict[str, Any]] = []
    body_error: BaseException | None = None
    try:
        affinity_records = apply_worker_affinity(args)
        preflight(args, affinity_records)
        if args.preflight_only:
            print("[stage14] preflight-only PASS", flush=True)
            return 0
        fanout_sources: list[dict[str, Any]] = []
        for dataset in DATASET_ORDER:
            print(f"[stage14] dataset start: {dataset}", flush=True)
            for baseline in ("baseline-a",):
                run_point(
                    args,
                    dataset=dataset,
                    method="random",
                    shards=1,
                    fanout=1,
                    label=baseline,
                    collection=collection_name(dataset, "common", 1, baseline),
                )
            for shards in PHYSICAL_COUNTS[1:]:
                fanout, source = selected_kmeans_fanout(dataset, shards)
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
                    method="kmeans",
                    shards=shards,
                    fanout=fanout,
                    label="primary",
                    collection=collection_name(dataset, "kmeans", shards),
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
            print(f"[stage14] dataset complete: {dataset}", flush=True)
        completion = {
            "timestamp": utc_timestamp(),
            "record_type": "stage14_physical_core_isolated_retest",
            "status": "MEASUREMENTS_COMPLETE",
            "datasets": list(DATASET_ORDER),
            "physical_counts": list(PHYSICAL_COUNTS),
            "fixed_ef": FIXED_EF,
            "fanout_sources": fanout_sources,
            "topology": str(args.topology),
            "topology_sha256": sha256_path(args.topology),
        }
        write_json(args.output_root / "execution-complete.json", completion)
        print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        body_error = error
        raise
    finally:
        if affinity_records:
            try:
                restore_worker_affinity(args, affinity_records)
            except Exception:
                if body_error is None:
                    raise
                print("[stage14] worker affinity restore also failed", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-root", default="/users/dry/Orion/experiments/c1")
    parser.add_argument(
        "--output-root",
        default=(
            "/users/dry/Orion/experiments/c1/retests/"
            "stage14-physical-core-isolated"
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
