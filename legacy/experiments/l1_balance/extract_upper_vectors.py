#!/usr/bin/env python3
"""Extract artifact upper vectors without changing their order or float32 bits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--vectors-output", required=True)
    parser.add_argument("--labels-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    return parser.parse_args()


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact).expanduser().resolve()
    vectors_path = Path(args.vectors_output).expanduser().resolve()
    labels_path = Path(args.labels_output).expanduser().resolve()
    manifest_path = Path(args.manifest_output).expanduser().resolve()
    outputs = (vectors_path, labels_path, manifest_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite upper extraction outputs: "
            + ", ".join(str(path) for path in existing)
        )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    nodes = artifact.get("upper_nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("artifact has no upper_nodes")
    dimension = int((artifact.get("vector_schema") or {}).get("dimension") or 0)
    vectors = np.asarray([node["vector"] for node in nodes], dtype="<f4")
    labels = np.asarray([int(node["label"]) for node in nodes], dtype="<u8")
    if vectors.shape != (len(nodes), dimension):
        raise ValueError("upper vectors are not a finite rectangular matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("upper vectors contain non-finite values")
    if len(np.unique(labels)) != len(labels):
        raise ValueError("upper labels are not unique")

    vectors.tofile(vectors_path)
    labels.tofile(labels_path)
    manifest = {
        "format_version": 1,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_path(artifact_path),
        "generation": int(artifact["generation"]),
        "row_count": len(nodes),
        "dimension": dimension,
        "vectors": str(vectors_path),
        "vectors_sha256": sha256_path(vectors_path),
        "vectors_size_bytes": vectors_path.stat().st_size,
        "labels": str(labels_path),
        "labels_sha256": sha256_path(labels_path),
        "labels_size_bytes": labels_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"vectors={vectors_path}")
    print(f"labels={labels_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
