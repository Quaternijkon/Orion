#!/usr/bin/env python3
"""Generate the protocol-defined P0-P3 C6 tuning matrix for one shard count."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_protocol import EF_GRID, canonical_json_sha256, write_json_atomic  # noqa: E402


ALPHA_GRID = (1, 2, 4, 8, 16)
BETA_GRID = (4, 8, 16, 32, 64)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-shards", type=int, choices=(4, 8, 16, 32), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ef-min", type=int, default=min(EF_GRID))
    parser.add_argument("--ef-max", type=int, default=max(EF_GRID))
    return parser.parse_args(argv)


def generate_matrix(
    logical_shards: int,
    *,
    ef_grid: Sequence[int] = EF_GRID,
    alpha_grid: Sequence[int] = ALPHA_GRID,
    beta_grid: Sequence[int] = BETA_GRID,
    ef_min: int = min(EF_GRID),
    ef_max: int = max(EF_GRID),
) -> dict[str, object]:
    if logical_shards not in {4, 8, 16, 32}:
        raise ValueError("logical_shards must be one of 4, 8, 16, 32")
    if ef_min <= 0 or ef_max < ef_min:
        raise ValueError("invalid EF clamp")
    efs = sorted(set(int(value) for value in ef_grid))
    alphas = sorted(set(int(value) for value in alpha_grid))
    betas = sorted(set(int(value) for value in beta_grid))
    if not efs or efs[0] <= 0 or not alphas or alphas[0] < 0 or not betas or betas[0] <= 0:
        raise ValueError("matrix grids contain invalid values")
    fanouts = list(range(1, logical_shards + 1))
    matrix: dict[str, object] = {
        "format_version": 1,
        "logical_shards": logical_shards,
        "target_recall": 0.90,
        "ef_grid": efs,
        "alpha_grid": alphas,
        "beta_grid": betas,
        "ef_min": ef_min,
        "ef_max": ef_max,
        "P0": [
            {"fixed_p": fixed_p, "uniform_ef_search": ef_search}
            for fixed_p in fanouts
            for ef_search in efs
        ],
        "P1": [{"uniform_ef_search": ef_search} for ef_search in efs],
        "P2": [
            {
                "fixed_p": fixed_p,
                "alpha": alpha,
                "beta": beta,
                "ef_min": ef_min,
                "ef_max": ef_max,
            }
            for fixed_p in fanouts
            for alpha in alphas
            for beta in betas
        ],
        "P3": [
            {
                "alpha": alpha,
                "beta": beta,
                "ef_min": ef_min,
                "ef_max": ef_max,
            }
            for alpha in alphas
            for beta in betas
        ],
    }
    matrix["matrix_sha256"] = canonical_json_sha256(matrix)
    return matrix


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = generate_matrix(
        args.logical_shards, ef_min=args.ef_min, ef_max=args.ef_max
    )
    output = write_json_atomic(Path(args.output), matrix)
    print(json.dumps({"output": str(output), "matrix_sha256": matrix["matrix_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
