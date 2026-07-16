#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_METHOD4_CONFIGS = [
    "target_080=16,16,10",
    "target_085=20,64,10",
    "target_090=60,40,8",
    "target_095=160,80,8",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a basic online Method4 worker-count scaling experiment on GloVe. "
            "Each worker-count deployment uses a temporary compose cluster with "
            "Qdrant storage bind-mounted under the result directory."
        )
    )
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--image", default="qdrant/qdrant:method4-peer-premerge")
    parser.add_argument("--output-root", default="results/method4_worker_count_online_scaling_20260708")
    parser.add_argument("--hdf5-path", default="/home/taig/dry/faiss/datasets/glove-200-angular.hdf5")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--qdrant-tool", default="tools/qdrant_two_level_routing_experiment.py")
    parser.add_argument("--latency-tool", default="tools/method4_claim_d_high_recall_latency.py")
    parser.add_argument("--initial-num-shards", type=int, default=31)
    parser.add_argument("--expected-effective-shards", type=int, default=46)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--eval-query-count", type=int, default=3000)
    parser.add_argument("--warmup-query-count", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--upper-search-ef", type=int, default=160)
    parser.add_argument("--method4-config", action="append", default=None, help="label=upper_k,base_ef,factor")
    parser.add_argument("--cpu-base", type=int, default=24)
    parser.add_argument("--controller-cpus", type=int, default=4)
    parser.add_argument("--worker-cpus", type=int, default=4)
    parser.add_argument("--cluster-timeout-sec", type=float, default=180.0)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--keep-cluster", action="store_true")
    parser.add_argument("--keep-storage", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def ports_for_worker_count(worker_count: int) -> dict[str, int]:
    offset = int(worker_count) * 100
    return {
        "controller_http": 6833 + offset,
        "controller_grpc": 6834 + offset,
        "shard_http_start": 6843 + offset,
        "shard_grpc_start": 6844 + offset,
    }


def cpu_range(start_cpu: int, count: int) -> str:
    if count <= 0:
        raise ValueError("CPU count must be positive")
    end_cpu = int(start_cpu) + int(count) - 1
    return str(start_cpu) if end_cpu == start_cpu else f"{start_cpu}-{end_cpu}"


def render_worker_count_compose(
    *,
    worker_count: int,
    image: str,
    host_http_port: int,
    host_grpc_port: int,
    shard_http_start: int,
    shard_grpc_start: int,
    storage_dir: Path,
    cpu_base: int,
    controller_cpus: int,
    worker_cpus: int,
) -> str:
    storage_dir = Path(storage_dir).resolve()
    controller_cpuset = cpu_range(cpu_base, controller_cpus)
    lines = [
        "services:",
        "  qdrant_controller:",
        f"    image: {image}",
        "    hostname: qdrant_controller",
        f"    cpuset: \"{controller_cpuset}\"",
        "    environment:",
        "      - QDRANT__SERVICE__GRPC_PORT=6334",
        "      - QDRANT__CLUSTER__ENABLED=true",
        "      - QDRANT__CLUSTER__P2P__PORT=6335",
        f"      - QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS={controller_cpus}",
        f"      - QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET={controller_cpus}",
        "      - QDRANT__LOG_LEVEL=INFO",
        "    ports:",
        f"      - \"{host_http_port}:6333\"",
        f"      - \"{host_grpc_port}:6334\"",
        "    command: [\"./qdrant\", \"--uri\", \"http://qdrant_controller:6335\"]",
        "    volumes:",
        f"      - {storage_dir / 'controller'}:/qdrant/storage",
        "    healthcheck:",
        "      test: [\"CMD-SHELL\", \"bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1\"]",
        "      interval: 5s",
        "      timeout: 5s",
        "      retries: 12",
        "",
    ]
    worker_cpu_base = int(cpu_base) + int(controller_cpus)
    for index in range(1, int(worker_count) + 1):
        cpuset = cpu_range(worker_cpu_base + (index - 1) * int(worker_cpus), worker_cpus)
        sleep_s = 3 + index * 2
        host_http = int(shard_http_start) + (index - 1) * 10
        host_grpc = int(shard_grpc_start) + (index - 1) * 10
        lines.extend(
            [
                f"  qdrant_shard_{index}:",
                f"    image: {image}",
                f"    hostname: qdrant_shard_{index}",
                f"    cpuset: \"{cpuset}\"",
                "    depends_on:",
                "      qdrant_controller:",
                "        condition: service_healthy",
                "    environment:",
                "      - QDRANT__SERVICE__GRPC_PORT=6334",
                "      - QDRANT__CLUSTER__ENABLED=true",
                "      - QDRANT__CLUSTER__P2P__PORT=6335",
                f"      - QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS={worker_cpus}",
                f"      - QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET={worker_cpus}",
                "      - QDRANT__LOG_LEVEL=INFO",
                "    ports:",
                f"      - \"{host_http}:6333\"",
                f"      - \"{host_grpc}:6334\"",
                "    command:",
                "      [",
                "        \"bash\",",
                "        \"-c\",",
                f"        \"sleep {sleep_s} && ./qdrant --bootstrap 'http://qdrant_controller:6335' --uri 'http://qdrant_shard_{index}:6335'\",",
                "      ]",
                "    volumes:",
                f"      - {storage_dir / f'shard_{index}'}:/qdrant/storage",
                "    healthcheck:",
                "      test: [\"CMD-SHELL\", \"bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1\"]",
                "      interval: 5s",
                "      timeout: 5s",
                "      retries: 12",
                "",
            ]
        )
    return "\n".join(lines)


def require_home_path(path: Path, label: str) -> None:
    resolved = Path(path).resolve()
    if not str(resolved).startswith("/home/"):
        raise ValueError(f"{label} must be under /home; got {resolved}")


def request_json(base_url: str, path: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_for_cluster(base_url: str, expected_peers: int, timeout_sec: float) -> dict[str, Any]:
    started = time.monotonic()
    last_error = ""
    while time.monotonic() - started < timeout_sec:
        try:
            result = request_json(base_url, "/cluster")["result"]
            peers = result.get("peers") or {}
            raft = result.get("raft_info") or {}
            if len(peers) >= expected_peers and raft.get("leader") is not None:
                return result
            last_error = f"peer_count={len(peers)}, raft={raft}"
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"cluster at {base_url} did not reach {expected_peers} peers: {last_error}")


def run_logged(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=Path.cwd(), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, text=True)


def docker_compose_command(compose_path: Path, project: str, *args: str) -> list[str]:
    return ["docker", "compose", "-f", str(compose_path), "-p", project, *args]


def root_owned_storage_cleanup_command(storage_dir: Path, image: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{Path(storage_dir).resolve()}:/cleanup",
        image,
        "bash",
        "-lc",
        "rm -rf /cleanup/* /cleanup/.[!.]* /cleanup/..?*",
    ]


def remove_storage_tree(storage_dir: Path, image: str, log_path: Path) -> None:
    if not storage_dir.exists():
        return
    run_logged(root_owned_storage_cleanup_command(storage_dir, image), log_path)
    shutil.rmtree(storage_dir)


def docker_stats_sampler(compose_path: Path, project: str, output_path: Path, stop: threading.Event) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        while not stop.is_set():
            ids_result = subprocess.run(
                docker_compose_command(compose_path, project, "ps", "-q"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            container_ids = [line.strip() for line in ids_result.stdout.splitlines() if line.strip()]
            if container_ids:
                stats_result = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "{{json .}}", *container_ids],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                )
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                for line in stats_result.stdout.splitlines():
                    if line.strip():
                        handle.write(json.dumps({"created_utc": timestamp, "raw": json.loads(line)}, sort_keys=True) + "\n")
                handle.flush()
            stop.wait(1.0)


def latest_summary(output_dir: Path) -> Path | None:
    summaries = sorted(Path(output_dir).glob("*/summary.json"))
    return summaries[-1] if summaries else None


def extract_build_cluster_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    collection_cluster = summary.get("collection_cluster") or {}
    return {
        "build_effective_num_shards": summary.get("num_shards"),
        "build_points_count": (summary.get("build") or {}).get("points_count"),
        "cluster_peer_count": collection_cluster.get("cluster_peer_count"),
        "cluster_shard_count": collection_cluster.get("cluster_shard_count"),
        "cluster_active_shards": collection_cluster.get("cluster_active_shards"),
    }


def build_collection_command(args: argparse.Namespace, base_url: str, collection: str, output_dir: Path) -> list[str]:
    command = [
        args.python,
        args.qdrant_tool,
        "--base-url",
        base_url,
        "--collection",
        collection,
        "--hdf5-path",
        args.hdf5_path,
        "--routing-mode",
        "faithful_original_rest",
        "--num-shards",
        str(args.initial_num_shards),
        "--top-k",
        "10",
        "--hnsw-m",
        "32",
        "--ef-construct",
        "100",
        "--upper-m",
        "32",
        "--upper-ef-construction",
        "100",
        "--upper-search-ef",
        str(args.upper_search_ef),
        "--upper-k-candidates",
        "16",
        "--base-ef-candidates",
        "16",
        "--factor-candidates",
        "10",
        "--tuning-query-count",
        "100",
        "--eval-query-count",
        "100",
        "--target-recall",
        "0.0",
        "--stability-repeats",
        "0",
        "--batch-size",
        str(args.batch_size),
        "--routed-execution-mode",
        "compact_multi_ep",
        "--routed-planning-mode",
        "materialized",
        "--routed-result-limit-mode",
        "top_k",
        "--search-dispatch-mode",
        "coordinator",
        "--shard-placement",
        "round_robin",
        "--placement-peer-uri-contains",
        "qdrant_shard_",
        "--output-dir",
        str(output_dir),
    ]
    if args.reuse_existing:
        command.append("--reuse-existing")
    return command


def latency_command(args: argparse.Namespace, base_url: str, collection: str, output_dir: Path) -> list[str]:
    command = [
        args.python,
        args.latency_tool,
        "--base-url",
        base_url,
        "--method4-collection",
        collection,
        "--method4-routing-source-collection",
        collection,
        "--hdf5-path",
        args.hdf5_path,
        "--num-shards",
        str(args.expected_effective_shards),
        "--eval-query-count",
        str(args.eval_query_count),
        "--warmup-query-count",
        str(args.warmup_query_count),
        "--batch-size",
        str(args.batch_size),
        "--repeats",
        str(args.repeats),
        "--upper-search-ef",
        str(args.upper_search_ef),
        "--skip-naive",
        "--skip-kmeans",
        "--skip-simple-kmeans",
        "--output-root",
        str(output_dir),
    ]
    for spec in args.method4_config or DEFAULT_METHOD4_CONFIGS:
        command.extend(["--method4-config", spec])
    return command


def run_worker_count(args: argparse.Namespace, worker_count: int, run_root: Path) -> dict[str, Any]:
    ports = ports_for_worker_count(worker_count)
    base_url = f"http://localhost:{ports['controller_http']}"
    project = f"method4-wc-{worker_count}-20260708"
    worker_root = run_root / f"worker_{worker_count}"
    storage_dir = worker_root / "docker_storage"
    compose_path = worker_root / "docker-compose.worker-count.yaml"
    logs_dir = worker_root / "logs"
    build_output = worker_root / "build"
    latency_output = worker_root / "latency"
    stats_output = worker_root / "docker_stats.jsonl"
    collection = f"method4_wc{worker_count}_glove_s31_fission_20260708"

    require_home_path(storage_dir, "storage_dir")
    worker_root.mkdir(parents=True, exist_ok=True)
    for subdir in ["controller", *[f"shard_{index}" for index in range(1, worker_count + 1)]]:
        (storage_dir / subdir).mkdir(parents=True, exist_ok=True)

    compose_text = render_worker_count_compose(
        worker_count=worker_count,
        image=args.image,
        host_http_port=ports["controller_http"],
        host_grpc_port=ports["controller_grpc"],
        shard_http_start=ports["shard_http_start"],
        shard_grpc_start=ports["shard_grpc_start"],
        storage_dir=storage_dir,
        cpu_base=args.cpu_base,
        controller_cpus=args.controller_cpus,
        worker_cpus=args.worker_cpus,
    )
    compose_path.write_text(compose_text, encoding="utf-8")

    env = os.environ.copy()
    env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    metadata: dict[str, Any] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "worker_count": worker_count,
        "base_url": base_url,
        "project": project,
        "collection": collection,
        "compose_path": str(compose_path),
        "storage_dir": str(storage_dir),
        "ports": ports,
        "method4_configs": args.method4_config or DEFAULT_METHOD4_CONFIGS,
        "cleanup": {
            "keep_cluster": bool(args.keep_cluster),
            "keep_storage": bool(args.keep_storage),
        },
    }
    cleanup_error: str | None = None
    try:
        run_logged(docker_compose_command(compose_path, project, "up", "-d"), logs_dir / "compose_up.log")
        metadata["cluster"] = wait_for_cluster(base_url, worker_count + 1, args.cluster_timeout_sec)
        if not args.latency_only:
            run_logged(build_collection_command(args, base_url, collection, build_output), logs_dir / "build_collection.log", env=env)
            summary_path = latest_summary(build_output)
            metadata["build_summary_path"] = str(summary_path) if summary_path else None
            if summary_path:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                metadata.update(extract_build_cluster_metadata(summary))
        if not args.build_only:
            stop = threading.Event()
            sampler = threading.Thread(
                target=docker_stats_sampler,
                args=(compose_path, project, stats_output, stop),
                daemon=True,
            )
            sampler.start()
            try:
                run_logged(latency_command(args, base_url, collection, latency_output), logs_dir / "latency.log", env=env)
            finally:
                stop.set()
                sampler.join(timeout=5.0)
            metadata["latency_output_dir"] = str(latency_output)
            metadata["docker_stats_jsonl"] = str(stats_output)
    finally:
        if not args.keep_cluster:
            run_logged(docker_compose_command(compose_path, project, "down"), logs_dir / "compose_down.log")
        if not args.keep_storage and storage_dir.exists():
            try:
                remove_storage_tree(storage_dir, args.image, logs_dir / "cleanup_storage.log")
                metadata["storage_removed_after_run"] = True
            except Exception as exc:  # pragma: no cover - preserves cleanup diagnostics for integration runs.
                cleanup_error = repr(exc)
                metadata["storage_removed_after_run"] = False
                metadata["storage_cleanup_error"] = cleanup_error
        else:
            metadata["storage_removed_after_run"] = False
        (worker_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        if cleanup_error is not None:
            raise RuntimeError(f"failed to remove temporary storage {storage_dir}: {cleanup_error}")
    return metadata


def main() -> int:
    args = parse_args()
    run_root = Path(args.output_root).resolve()
    require_home_path(run_root, "output_root")
    run_root.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    for worker_count in args.worker_counts:
        if worker_count <= 0:
            raise ValueError(f"worker_count must be positive: {worker_count}")
        metadata_rows.append(run_worker_count(args, worker_count, run_root))
    (run_root / "worker_count_run_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "worker_counts": args.worker_counts,
                "runs": metadata_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metadata_rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
