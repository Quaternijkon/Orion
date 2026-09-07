#!/usr/bin/env python3
"""Recover a missing frozen f32le vector file from the checksum-bound HDF5 source."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=16384)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    digest = hashlib.sha256()
    try:
        with h5py.File(args.hdf5_path, "r") as handle, temporary.open("xb") as output:
            train = handle["train"]
            if train.ndim != 2 or len(train) == 0:
                raise ValueError("HDF5 train must be a non-empty matrix")
            for start in range(0, len(train), args.chunk_size):
                rows = train[start : start + args.chunk_size].astype(
                    np.float32, copy=True
                )
                if args.normalize:
                    norms = np.linalg.norm(rows, axis=1, keepdims=True)
                    norms[norms < 1e-12] = 1.0
                    rows /= norms
                rows = np.ascontiguousarray(rows, dtype="<f4")
                raw = memoryview(rows).cast("B")
                output.write(raw)
                digest.update(raw)
            output.flush()
            os.fsync(output.fileno())
        actual = digest.hexdigest()
        if actual != args.expected_sha256:
            raise ValueError(
                f"recovered vector checksum differs: {actual} != {args.expected_sha256}"
            )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"output={destination}")
    print(f"sha256={args.expected_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
