#!/usr/bin/env python3
"""Build a fixed-M C_CNBR owner with same-dataset P32 identity binding for C6."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from c6_protocol import sha256_path, utc_timestamp, write_json_atomic  # noqa: E402
from experiments.c1.scripts import c1_orion_bmr10_scaling_materialize as materializer  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=int, choices=(4, 8, 16, 32), required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--c6-record", type=Path, required=True)
    return parser.parse_args(argv)


def phase_a_p32_owner_hashes(path: str | Path) -> dict[str, str]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    owners = payload.get("owners") if isinstance(payload, dict) else None
    if not isinstance(owners, dict):
        raise ValueError("Phase-A manifest is missing owners")
    result = {}
    for name in ("N_native", "C_CNBR"):
        record = owners.get(name)
        owner = record.get("owner") if isinstance(record, dict) else None
        digest = owner.get("sha256") if isinstance(owner, dict) else None
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Phase-A manifest has an invalid {name} owner checksum")
        result[name] = digest
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    same_dataset_reference: dict[str, Any] | None = None
    original_n = materializer.P32_N_OWNER_SHA256
    original_c = materializer.P32_OWNER_SHA256
    try:
        if args.partitions == 32:
            hashes = phase_a_p32_owner_hashes(args.phase_a_manifest)
            materializer.P32_N_OWNER_SHA256 = hashes["N_native"]
            materializer.P32_OWNER_SHA256 = hashes["C_CNBR"]
            same_dataset_reference = {
                "phase_a_manifest": str(args.phase_a_manifest.expanduser().resolve()),
                "phase_a_manifest_sha256": sha256_path(args.phase_a_manifest),
                "N_native_owner_sha256": hashes["N_native"],
                "C_CNBR_owner_sha256": hashes["C_CNBR"],
            }
        result = materializer.build_owner(args)
    finally:
        materializer.P32_N_OWNER_SHA256 = original_n
        materializer.P32_OWNER_SHA256 = original_c
    record = {
        "record_type": "c6_fixed_owner_materialization",
        "format_version": 1,
        "timestamp": utc_timestamp(),
        "partitions": args.partitions,
        "same_dataset_p32_reference": same_dataset_reference,
        "materializer": {
            "path": str(Path(materializer.__file__).resolve()),
            "sha256": sha256_path(materializer.__file__),
        },
        "c6_wrapper": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_path(__file__),
        },
        "result": result,
    }
    write_json_atomic(args.c6_record, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
