#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from c1_protocol import repository_commit, utc_timestamp


GRAPH_FILENAMES = {
    "graph.bin",
    "links.bin",
    "links_compressed.bin",
    "links_comp_vec.bin",
}


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: Sequence[str], *, capture: bool = False) -> str:
    print("[c1-section31]", " ".join(command), flush=True)
    completed = subprocess.run(
        list(command),
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return completed.stdout.strip() if capture else ""


def request_json(
    method: str,
    url: str,
    payload: Any | None = None,
    *,
    timeout: float = 300.0,
) -> dict[str, Any]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object from {url}")
    if result.get("status") not in {"ok", None}:
        raise RuntimeError(f"request failed: {method} {url}: {result}")
    return result


def wait_ready(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/readyz", timeout=5.0) as response:
                if int(response.status) == 200:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(0.5)
    raise TimeoutError(f"Qdrant did not become ready: {last_error}")


def wait_indexed(
    base_url: str,
    collection: str,
    point_count: int,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    stable = 0
    while time.monotonic() < deadline:
        response = request_json(
            "GET", f"{base_url}/collections/{collection}", timeout=15.0
        )
        last = dict(response["result"])
        complete = (
            last.get("status") == "green"
            and int(last.get("points_count") or 0) == point_count
            and int(last.get("indexed_vectors_count") or 0) == point_count
            # Qdrant retains one empty appendable segment beside the indexed
            # segment in this configuration, so either one or two segments is
            # valid as long as every vector is indexed and the optimizer is idle.
            and int(last.get("segments_count") or 0) in {1, 2}
            and (last.get("optimizer_status") == "ok")
        )
        stable = stable + 1 if complete else 0
        if stable >= 3:
            return last
        time.sleep(1.0)
    raise TimeoutError(f"index did not converge to one complete HNSW graph: {last}")


def graph_digest(storage: Path) -> tuple[str, list[dict[str, Any]]]:
    paths = sorted(
        path
        for path in storage.rglob("*")
        if path.is_file() and path.name in GRAPH_FILENAMES
    )
    if len(paths) != 2 or {path.name for path in paths} not in (
        {"graph.bin", "links.bin"},
        {"graph.bin", "links_compressed.bin"},
        {"graph.bin", "links_comp_vec.bin"},
    ):
        raise ValueError(
            f"expected exactly one HNSW graph/links pair under {storage}, got {paths}"
        )
    records = [
        {
            "filename": path.name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in paths
    ]
    canonical = json.dumps(
        [
            {
                "filename": row["filename"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in records
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), records


def create_and_upload(
    base_url: str,
    collection: str,
    *,
    point_count: int,
    dimension: int,
    vector_seed: int,
    upload_batch_size: int,
) -> None:
    request_json(
        "PUT",
        f"{base_url}/collections/{collection}",
        {
            "vectors": {"size": dimension, "distance": "Euclid"},
            "shard_number": 1,
            "replication_factor": 1,
            "on_disk_payload": True,
            "hnsw_config": {
                "m": 32,
                "ef_construct": 200,
                "full_scan_threshold": 10,
                "max_indexing_threads": 1,
                "on_disk": False,
            },
            "optimizers_config": {
                "default_segment_number": 1,
                "indexing_threshold": 10,
                "max_optimization_threads": 1,
                "flush_interval_sec": 1,
            },
        },
    )
    vectors = np.random.default_rng(vector_seed).standard_normal(
        (point_count, dimension), dtype=np.float32
    )
    for start in range(0, point_count, upload_batch_size):
        stop = min(start + upload_batch_size, point_count)
        points = [
            {"id": index, "vector": vectors[index].tolist()}
            for index in range(start, stop)
        ]
        request_json(
            "PUT",
            f"{base_url}/collections/{collection}/points?wait=true",
            {"points": points},
        )
        print(f"[c1-section31] uploaded {stop}/{point_count}", flush=True)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    retained_root = Path(args.retained_root).expanduser().resolve()
    invocation_root = retained_root / utc_timestamp().replace(":", "").replace("-", "")
    invocation_root.mkdir(parents=True, exist_ok=False)

    image_id = run(
        ["sudo", "-n", "docker", "image", "inspect", args.image_tag, "--format", "{{.Id}}"],
        capture=True,
    )
    if args.expected_image_id and image_id != args.expected_image_id:
        raise ValueError(
            f"image identity mismatch: expected {args.expected_image_id}, got {image_id}"
        )

    builds: list[dict[str, Any]] = []
    collection = "c1_section31_deterministic_hnsw"
    for build_number in (1, 2):
        build_root = invocation_root / f"build-{build_number}"
        storage = build_root / "storage"
        storage.mkdir(parents=True)
        container = f"c1-section31-{os.getpid()}-{build_number}"
        base_url = f"http://127.0.0.1:{args.port}"
        started = False
        try:
            run(
                [
                    "sudo", "-n", "docker", "run", "-d",
                    "--name", container,
                    "--cpuset-cpus", args.cpuset,
                    "--ulimit", "nofile=65536:65536",
                    "-p", f"127.0.0.1:{args.port}:6333",
                    "-e", f"QDRANT_HNSW_GRAPH_BUILD_SEED={args.graph_build_seed}",
                    "-e", "QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET=1",
                    "-v", f"{storage}:/qdrant/storage",
                    args.image_tag,
                ]
            )
            started = True
            wait_ready(base_url, args.timeout)
            create_and_upload(
                base_url,
                collection,
                point_count=args.point_count,
                dimension=args.dimension,
                vector_seed=args.vector_seed,
                upload_batch_size=args.upload_batch_size,
            )
            collection_info = wait_indexed(
                base_url, collection, args.point_count, args.timeout
            )
        finally:
            if started:
                run(["sudo", "-n", "docker", "stop", container])
                run(["sudo", "-n", "docker", "rm", container])
        digest, files = graph_digest(storage)
        builds.append(
            {
                "build_number": build_number,
                "retained_storage": str(storage),
                "graph_content_sha256": digest,
                "graph_files": files,
                "collection_info": collection_info,
            }
        )

    digests = {build["graph_content_sha256"] for build in builds}
    verified = len(digests) == 1
    payload = {
        "timestamp": utc_timestamp(),
        "record_type": "section31_deterministic_build_proof",
        "source_commit": repository_commit(Path(__file__).resolve().parents[3]),
        "image_tag": args.image_tag,
        "image_id": image_id,
        "graph_build_seed": args.graph_build_seed,
        "vector_generation_seed": args.vector_seed,
        "point_count": args.point_count,
        "dimension": args.dimension,
        "distance_metric": "Euclid",
        "hnsw_m": 32,
        "hnsw_ef_construction": 200,
        "max_indexing_threads": 1,
        "max_optimization_threads": 1,
        "identical_build_count": 2 if verified else 0,
        "graph_content_sha256": next(iter(digests)) if verified else None,
        "deterministic_construction_verified": verified,
        "authoritative_e2_e4_rerun_complete": False,
        "retained_build_root": str(invocation_root),
        "builds": builds,
        "status": "VERIFIED" if verified else "MISMATCH",
    }
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not verified:
        raise RuntimeError("independent deterministic HNSW builds did not match")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove deterministic HNSW construction with two independent builds"
    )
    parser.add_argument("--image-tag", default="orion-c1:20260821-deterministic-seed")
    parser.add_argument(
        "--expected-image-id",
        default="sha256:54819a6648dc706f654f3e541351c68814c0c1798312149c5e3ee099c183bfc8",
    )
    parser.add_argument("--graph-build-seed", type=int, default=20260821)
    parser.add_argument("--vector-seed", type=int, default=20260821)
    parser.add_argument("--point-count", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--upload-batch-size", type=int, default=1_000)
    parser.add_argument("--port", type=int, default=17433)
    parser.add_argument("--cpuset", default="20-23")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--retained-root",
        default=(
            "/proj/intelisys-PG0/exp/orion-distributed/"
            "c1-20260821-v2/section31-builds"
        ),
    )
    parser.add_argument(
        "--output",
        default="experiments/c1/runs/section31-deterministic-build-proof.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    execute(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
