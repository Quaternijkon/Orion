#!/usr/bin/env python3
"""Create deterministic raw inputs for the Qdrant-native C23 reference-trace smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--points", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=25)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if args.points < 20 or args.queries < 2 or args.dimension <= 0:
        raise ValueError("smoke dimensions are too small")
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    vectors = rng.normal(size=(args.points, args.dimension)).astype(np.float32)
    anchors = rng.choice(args.points, size=args.queries, replace=False)
    queries = vectors[anchors] + rng.normal(
        scale=0.01,
        size=(args.queries, args.dimension),
    ).astype(np.float32)
    distances = np.sum((queries[:, None, :] - vectors[None, :, :]) ** 2, axis=2)
    candidates = np.argpartition(distances, kth=9, axis=1)[:, :10]
    candidate_distances = np.take_along_axis(distances, candidates, axis=1)
    order = np.argsort(candidate_distances, axis=1, kind="stable")
    truth = np.take_along_axis(candidates, order, axis=1).astype(np.uint32)

    vectors.astype("<f4", copy=False).tofile(output / "vectors.f32le")
    queries.astype("<f4", copy=False).tofile(output / "queries.f32le")
    truth.astype("<u4", copy=False).tofile(output / "ground_truth.u32le")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "points": args.points,
                "queries": args.queries,
                "dimension": args.dimension,
                "seed": args.seed,
                "distance": "euclid",
                "ground_truth_width": 10,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
