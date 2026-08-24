#!/usr/bin/env python3
"""Export audited C23 partition assignments for the Rust local evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c23.scripts import c23_e1, c23_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(c23_protocol.DATASETS), required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--orion-build-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    spec = c23_protocol.DATASETS[args.dataset]
    matrix = c23_e1.load_partition_matrix(
        dataset=args.dataset,
        point_count=spec.expected_train_rows,
        dataset_sha256=spec.sha256,
        baseline_root=args.baseline_root,
        orion_manifest_path=args.orion_build_manifest,
    )
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    configurations = []
    for row in matrix:
        method = str(row["method"])
        logical_shards = int(row["logical_shards"])
        partition_seed = int(row["metadata"]["partition_seed"])
        configuration_id = f"{method}-m{logical_shards}-seed{partition_seed}"
        path = output / f"{configuration_id}.u32le"
        np.asarray(row["assignments"], dtype="<u4").tofile(path)
        if path.stat().st_size != spec.expected_train_rows * 4:
            raise ValueError(f"raw assignment size mismatch: {path}")
        configurations.append(
            {
                "configuration_id": configuration_id,
                "partition_method": method,
                "partition_seed": partition_seed,
                "logical_shards": logical_shards,
                "point_count": spec.expected_train_rows,
                "raw_assignment_path": str(path),
                "raw_assignment_sha256": c23_e1.sha256_path(path),
                "source_artifact_path": row["artifact_path"],
                "source_artifact_sha256": row["artifact_sha256"],
            }
        )
    manifest = {
        "protocol_version": c23_protocol.PROTOCOL_VERSION,
        "timestamp": c23_e1.utc_timestamp(),
        "git_commit": c23_protocol.git_commit(REPO_ROOT),
        "dataset": args.dataset,
        "dataset_sha256": spec.sha256,
        "point_count": spec.expected_train_rows,
        "dimension": spec.dimension,
        "distance": spec.metric,
        "configuration_count": len(configurations),
        "configurations": configurations,
        "checks": {
            "complete_partition_matrix": "PASS",
            "source_artifact_checksums": "PASS",
            "raw_assignment_sizes": "PASS",
        },
    }
    c23_e1.write_json_new(output / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
