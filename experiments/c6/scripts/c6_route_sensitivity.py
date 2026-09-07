#!/usr/bin/env python3
"""Generate exact held-out Orion route traces for navigation-K sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_protocol import sha256_path, utc_timestamp, write_json_atomic  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sift1m", "glove-200-angular"), required=True)
    parser.add_argument("--source-route-trace", required=True)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--query-start", type=int, default=1000)
    parser.add_argument("--query-count", type=int, default=9000)
    parser.add_argument("--navigation-k", type=int, required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def replace_once(data: bytes, old: bytes, new: bytes, field: str) -> bytes:
    count = data.count(old)
    if count != 1:
        raise ValueError(f"expected one {field} token, found {count}")
    return data.replace(old, new, 1)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.navigation_k <= 0 or args.query_start < 0 or args.query_count <= 0:
        raise ValueError("navigation K and query range must be positive")
    source_trace_path = Path(args.source_route_trace).expanduser().resolve()
    source_trace = json.loads(source_trace_path.read_text(encoding="utf-8"))
    source_artifact = Path(source_trace["artifact"]["path"]).expanduser().resolve()
    source_k = int(source_trace["artifact"]["upper_k"])
    source_ef = int(source_trace["artifact"]["upper_ef_search"])
    effective_ef = max(args.navigation_k, source_ef)
    artifact_bytes = source_artifact.read_bytes()
    artifact_bytes = replace_once(
        artifact_bytes,
        f'"upper_k":{source_k}'.encode(),
        f'"upper_k":{args.navigation_k}'.encode(),
        "upper_k",
    )
    artifact_bytes = replace_once(
        artifact_bytes,
        f'"upper_ef_search":{source_ef}'.encode(),
        f'"upper_ef_search":{effective_ef}'.encode(),
        "upper_ef_search",
    )

    import h5py

    with h5py.File(args.hdf5_path, "r") as handle:
        queries = handle["test"][
            args.query_start : args.query_start + args.query_count
        ].astype(np.float32, copy=True)
    if len(queries) != args.query_count:
        raise ValueError("query range is not fully present in the HDF5 dataset")
    if args.dataset == "glove-200-angular":
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("GloVe sensitivity queries contain zero-norm rows")
        queries /= norms
    queries = np.ascontiguousarray(queries, dtype=np.dtype("<f4"))

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / f"artifact-k{args.navigation_k}.json"
    queries_path = output / "queries-9000.f32le"
    trace_path = output / f"route-trace-k{args.navigation_k}.json"
    for path in (artifact_path, queries_path, trace_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    artifact_tmp = artifact_path.with_suffix(".json.tmp")
    query_tmp = queries_path.with_suffix(".f32le.tmp")
    artifact_tmp.write_bytes(artifact_bytes)
    query_tmp.write_bytes(queries.tobytes(order="C"))
    os.replace(artifact_tmp, artifact_path)
    os.replace(query_tmp, queries_path)

    command = [
        str(Path(args.binary).expanduser().resolve()),
        str(artifact_path),
        str(queries_path),
        str(args.query_count),
        str(queries.shape[1]),
        str(trace_path),
        "--per-query",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (output / f"route-trace-k{args.navigation_k}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output / f"route-trace-k{args.navigation_k}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"route trace exited {completed.returncode}: {completed.stderr[-1000:]}"
        )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    counts = {int(row["navigation_candidate_count"]) for row in trace["per_query"]}
    if counts != {args.navigation_k}:
        raise RuntimeError(f"unexpected candidate counts in trace: {sorted(counts)}")
    manifest = {
        "record_type": "c6_navigation_k_route_trace",
        "timestamp": utc_timestamp(),
        "dataset": args.dataset,
        "query_start": args.query_start,
        "query_count": args.query_count,
        "navigation_k": args.navigation_k,
        "effective_upper_ef_search": effective_ef,
        "source_route_trace": str(source_trace_path),
        "source_route_trace_sha256": sha256_path(source_trace_path),
        "source_artifact": str(source_artifact),
        "source_artifact_sha256": hashlib.sha256(source_artifact.read_bytes()).hexdigest(),
        "modified_artifact": str(artifact_path),
        "modified_artifact_sha256": sha256_path(artifact_path),
        "queries_sha256": sha256_path(queries_path),
        "route_trace": str(trace_path),
        "route_trace_sha256": sha256_path(trace_path),
    }
    write_json_atomic(output / f"route-trace-k{args.navigation_k}-manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
