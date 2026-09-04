"""Convert an ann-benchmarks HDF5 dataset into npy arrays for the harness.

Splitting this out keeps HDF5 decoding entirely outside any measured path, and
lets client processes memory-map the query matrix instead of each decoding it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="unit-normalize vectors, required for angular datasets served with cosine",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5, "r") as handle:
        train = np.asarray(handle["train"], dtype=np.float32)
        test = np.asarray(handle["test"], dtype=np.float32)
        neighbors = np.asarray(handle["neighbors"], dtype=np.int64)

    if args.normalize:
        for matrix in (train, test):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.divide(matrix, np.maximum(norms, 1e-12), out=matrix)

    np.save(out_dir / "train.npy", train)
    np.save(out_dir / "queries.npy", test)
    np.save(out_dir / "ground_truth.npy", neighbors)
    print(
        f"train={train.shape} queries={test.shape} "
        f"ground_truth={neighbors.shape} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
