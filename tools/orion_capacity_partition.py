#!/usr/bin/env python3
"""Build a capacity layout directly from raw vectors, using native bundle authority.

PyMetis proposes owner labels. Rust generates strict memberships and validates the
published layout. Feedback runs only after actual physical capacity is rejected.
No reference layout, query truth or replica-budget owner repair is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable

import numpy as np

MAX_CAPACITY_RATIO = 1.2  # Rust v4 contract; use integer checks in both implementations.


def check_limit(max_replication_factor: float | None) -> None:
    if max_replication_factor is not None and (
        not math.isfinite(max_replication_factor) or max_replication_factor < 1
    ):
        raise ValueError("maximum replication factor must be finite and at least 1")


def feasible(counts: np.ndarray, rows: int, max_replication_factor: float | None) -> bool:
    check_limit(max_replication_factor)
    total = sum(int(x) for x in counts)
    return (
        rows > 0 and len(counts) > 0 and min(counts) > 0 and total >= rows
        and int(max(counts)) * len(counts) * 5 <= total * 6
        and (max_replication_factor is None or total / rows <= max_replication_factor)
    )


def evaluate_memberships(hits: np.ndarray, owners: np.ndarray, upper_ids: np.ndarray,
                         shards: int, feedback: bool = False) -> tuple[np.ndarray, int, np.ndarray | None]:
    """Count actual physical copies while voting; optionally attribute their weight.

    No vectors move and no HNSW search is performed. Owner-copy completion is
    included before counting. This helper's counts are checked against Rust later.
    """
    if len(owners) != len(upper_ids) or (owners < 0).any() or (owners >= shards).any():
        raise ValueError("invalid owner labels")
    if len(np.unique(upper_ids)) != len(upper_ids) or (upper_ids < 0).any() or (upper_ids >= len(hits)).any():
        raise ValueError("invalid upper sample IDs")
    if hits.ndim != 2 or hits.shape[1] == 0:
        raise ValueError("empty attachment lists")
    sample_order = np.argsort(upper_ids)
    sorted_samples = upper_ids[sample_order]
    counts = np.zeros(shards, dtype=np.int64)
    weights = np.zeros(len(owners), dtype=np.float64) if feedback else None
    copies_added = 0
    for start in range(0, len(hits), 10000):
        h = hits[start:start + 10000]
        if (h < 0).any() or (h >= len(owners)).any():
            raise ValueError("unknown attachment entry")
        if (np.diff(np.sort(h, axis=1), axis=1) == 0).any():
            raise ValueError("duplicate attachment entry")
        destination = owners[h]
        rr = np.arange(len(h))
        votes = np.zeros((len(h), shards), dtype=np.int64)
        np.add.at(votes, (rr[:, None], destination), 1)
        maximum = votes.max(axis=1)
        members = votes == maximum[:, None]
        one = maximum == 1
        members[one] = False
        members[rr[one], destination[one, 0]] = True
        lo, hi = np.searchsorted(sorted_samples, [start, start + len(h)])
        entry_indices = sample_order[lo:hi]
        sample_rows = upper_ids[entry_indices] - start
        entry_owners = owners[entry_indices]
        copies_added += int((~members[sample_rows, entry_owners]).sum())
        members[sample_rows, entry_owners] = True
        counts += members.sum(axis=0)
        if weights is not None:
            responsibility = members[rr[:, None], destination] / votes[rr[:, None], destination]
            weights += np.bincount(h.ravel(), weights=responsibility.ravel(), minlength=len(owners))
            absent = votes[sample_rows, entry_owners] == 0
            np.add.at(weights, entry_indices[absent], 1.0)
    if weights is not None:
        if not np.isclose(weights.sum(), counts.sum(), rtol=1e-10, atol=1e-6):
            raise ValueError("membership responsibility does not conserve physical copies")
        weights = np.maximum(1, np.rint(weights * 1000)).astype(np.int64)
    return counts, copies_added, weights


def choose_partition(hits: np.ndarray, upper_ids: np.ndarray, shards: int,
                     partition: Callable[[np.ndarray], np.ndarray], *,
                     max_feedback_rounds: int = 5, max_replication_factor: float | None = None,
                     record: Callable[[dict], None] | None = None) -> tuple[np.ndarray, list[dict]]:
    if max_feedback_rounds < 0:
        raise ValueError("feedback round limit must be nonnegative")
    check_limit(max_replication_factor)
    weights = np.maximum(1, np.bincount(hits[:, 0], minlength=len(upper_ids))).astype(np.int64)
    history = []
    for iteration in range(max_feedback_rounds + 1):
        owners = np.asarray(partition(weights), dtype=np.int64)
        counts, added, _ = evaluate_memberships(hits, owners, upper_ids, shards)
        accepted = feasible(counts, len(hits), max_replication_factor)
        row = dict(iteration=iteration, accepted=bool(accepted), shard_counts=counts.tolist(),
                   physical_copies=int(counts.sum()), replication_factor=float(counts.sum() / len(hits)),
                   max_data_over_mean=float(counts.max() / counts.mean()), owner_entry_copies_added=added)
        history.append(row)
        if record:
            record(row)
        if accepted:
            return owners, history  # No weight feedback or second partition for a feasible first result.
        if iteration < max_feedback_rounds:
            _, _, weights = evaluate_memberships(hits, owners, upper_ids, shards, feedback=True)
    raise ValueError("no feasible capacity layout within feedback limit; no bundle published")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(root: Path, name: str, digest: str) -> Path:
    if not name or Path(name).name != name or name in (".", "..") or "\\" in name:
        raise ValueError("unsafe prepared filename")
    p = root / name
    if p.is_symlink() or sha256(p) != digest:
        raise ValueError("prepared checksum mismatch: " + name)
    return p


def plan_prepared(prepared: Path, evidence: Path, *, max_feedback_rounds: int = 5,
                  max_replication_factor: float | None = None, seed: int = 100) -> tuple[Path, dict]:
    # Check dependency before allocating a large graph. Only used at build time.
    try:
        import pymetis
    except ImportError as exc:
        raise RuntimeError("capacity construction requires pymetis (pip install pymetis)") from exc
    meta = json.loads((prepared / "capacity-input.json").read_text())
    rows = meta["point_count"]
    upper_ids = np.asarray(meta["upper_indices"], dtype=np.int64)
    size = len(upper_ids)
    if size == 0 or len(np.unique(upper_ids)) != size or (upper_ids < 0).any() or (upper_ids >= rows).any():
        raise ValueError("invalid upper sample IDs")
    k = min(meta["options"]["efs"], size)
    path = checked_path(prepared, meta["attachments_file"], meta["attachments_sha256"])
    dtype = np.dtype([("count", "<u4"), ("hits", "<u8", (k,))])
    if path.stat().st_size != rows * dtype.itemsize:
        raise ValueError("attachment size mismatch")
    records = np.memmap(path, dtype=dtype, mode="r")
    if not (records["count"] == k).all():
        raise ValueError("attachment count mismatch")
    order = np.argsort(upper_ids)
    sorted_ids = upper_ids[order]
    position = np.searchsorted(sorted_ids, records["hits"])
    if (position >= size).any() or not np.array_equal(sorted_ids[position], records["hits"]):
        raise ValueError("unknown attachment label")
    hits = order[position]
    first = hits[:, :1].astype(np.uint64)
    other = hits[:, 1:].astype(np.uint64)
    codes = (np.minimum(first, other) * size + np.maximum(first, other)).ravel()
    unique, multiplicity = np.unique(codes, return_counts=True)
    a = (unique // size).astype(np.int64)
    b = (unique % size).astype(np.int64)
    if (a == b).any():
        raise ValueError("duplicate attachment entry")
    src = np.r_[a, b]
    dst = np.r_[b, a]
    edgeweights = np.r_[multiplicity, multiplicity].astype(np.int64)
    edge_order = np.argsort(src, kind="stable")
    xadj = np.r_[0, np.cumsum(np.bincount(src, minlength=size))]
    neighbors = dst[edge_order]
    edgeweights = edgeweights[edge_order]
    shards = meta["options"]["shard_count"]
    options = dict(seed=seed, ufactor=30, ncuts=4, niter=20)
    graph = pymetis.CSRAdjacency(xadj, neighbors)

    def partition(weights):
        if shards == 1:
            return np.zeros(size, dtype=np.int64)
        _, labels = pymetis.part_graph(shards, adjacency=graph, vweights=weights,
                                      eweights=edgeweights, recursive=False,
                                      options=pymetis.Options(**options))
        return np.asarray(labels, dtype=np.int64)

    evidence.mkdir(parents=True, exist_ok=True)
    def record(row):
        with (evidence / "capacity-attempts.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    owners, history = choose_partition(hits, upper_ids, shards, partition,
                                      max_feedback_rounds=max_feedback_rounds,
                                      max_replication_factor=max_replication_factor, record=record)
    owner_file = evidence / "capacity-owners.json"
    owner_file.write_text(json.dumps(owners.tolist()) + "\n")
    report = dict(algorithm="attachment-cooccurrence-capacity", options=options,
                  pymetis_version=pymetis.version, prepared_manifest_sha256=sha256(prepared / "capacity-input.json"),
                  owners_sha256=sha256(owner_file), max_feedback_rounds=max_feedback_rounds,
                  max_replication_factor=max_replication_factor, max_physical_capacity_ratio=MAX_CAPACITY_RATIO, history=history,
                  feedback_rounds_used=len(history) - 1, source_layout_required=False,
                  uses_queries=False, replica_budget_owner_repair=False,
                  full_data_upper_search_passes=1)
    (evidence / "capacity-plan.json").write_text(json.dumps(report, indent=2) + "\n")
    return owner_file, report


def build_capacity(request: Path, output: Path, run_native: Callable[..., dict], *,
                   max_feedback_rounds: int = 5, max_replication_factor: float | None = None,
                   seed: int = 100) -> dict:
    check_limit(max_replication_factor)
    if max_feedback_rounds < 0:
        raise ValueError("feedback round limit must be nonnegative")
    if os.path.lexists(output):
        raise FileExistsError("refusing to overwrite capacity output: " + str(output))
    # Fail before the expensive full-data upper pass if the solver is unavailable.
    try:
        import pymetis  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("capacity construction requires pymetis (pip install pymetis)") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=".orion-capacity-work-", dir=output.parent))
    try:
        prepared = workspace / "prepared"
        run_native("prepare-capacity", "--request", str(request), "--output", str(prepared))
        owners, report = plan_prepared(prepared, workspace, max_feedback_rounds=max_feedback_rounds,
                                      max_replication_factor=max_replication_factor, seed=seed)
        meta = json.loads((prepared / "capacity-input.json").read_text())
        args = ["capacity-partition", "--source", str(prepared), "--owners", str(owners),
                "--output", str(output), "--generation", str(meta["options"]["generation"])]
        if max_replication_factor is not None:
            args += ["--max-replication-factor", str(max_replication_factor)]
        manifest = run_native(*args)
        if manifest["shard_counts"] != report["history"][-1]["shard_counts"]:
            raise RuntimeError("Python/native physical membership count mismatch")
        # Source vector and attachment bytes are already preserved in the final bundle.
        import shutil
        for name in ["capacity-plan.json", "capacity-attempts.jsonl", "capacity-owners.json"]:
            shutil.copy2(workspace / name, output / name)
        shutil.rmtree(workspace)
        return manifest
    except Exception as exc:
        (workspace / "FAILED.json").write_text(json.dumps({"error":str(exc)}, indent=2) + "\n")
        raise RuntimeError(f"{exc}; capacity build evidence: {workspace}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rust-layout-binary", required=True, type=Path)
    parser.add_argument("--max-feedback-rounds", type=int, default=5)
    parser.add_argument("--max-replication-factor", type=float)
    parser.add_argument("--partition-seed", type=int, default=100)
    args = parser.parse_args()
    def run(*command):
        result = subprocess.run([str(args.rust_layout_binary.resolve()), *command],
                                check=True, text=True, stdout=subprocess.PIPE)
        return json.loads(result.stdout)
    result = build_capacity(args.request, args.output, run, max_feedback_rounds=args.max_feedback_rounds,
                            max_replication_factor=args.max_replication_factor, seed=args.partition_seed)
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
