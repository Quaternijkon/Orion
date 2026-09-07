#!/usr/bin/env python3
"""Freeze dual-dataset BMR_10_CAP35 evidence before online use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.multi_assignment import capacity_bounded_bmr10 as cap


TOOL_PATH = "experiments/multi_assignment/synthesize_capacity_bounded_selection.py"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-screen", type=Path, required=True)
    parser.add_argument("--sift-screen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    paths = {
        "glove-200-angular": args.glove_screen.expanduser().resolve(strict=True),
        "sift1m": args.sift_screen.expanduser().resolve(strict=True),
    }
    rows: dict[str, Any] = {}
    for dataset, path in paths.items():
        screen = load_object(path)
        expected_parameters = {
            "min_load_ratio": cap.MIN_LOAD_RATIO,
            "max_load_ratio": cap.MAX_LOAD_RATIO,
            "max_vote_loss": cap.MAX_VOTE_LOSS,
            "max_passes": cap.MAX_PASSES,
        }
        if (
            screen.get("record_type")
            != "capacity_bounded_bmr10_offline_screen"
            or screen.get("dataset") != dataset
            or screen.get("status") != "PASS"
            or screen.get("parameters") != expected_parameters
            or not all(gate.get("pass") is True for gate in screen["gates"].values())
        ):
            raise ValueError(f"capacity-bounded screen did not pass: {dataset}")
        if screen["candidate"]["physical_point_count"] != screen["baseline"][
            "physical_point_count"
        ]:
            raise ValueError(f"capacity-bounded copy count drifted: {dataset}")
        if screen["identity"].get("owner_unchanged") is not True:
            raise ValueError(f"capacity-bounded owner changed: {dataset}")
        rows[dataset] = {
            "screen": file_record(path),
            "baseline": screen["baseline"],
            "candidate": screen["candidate"],
            "capacity_diagnostics": screen["capacity_diagnostics"],
            "gates": screen["gates"],
            "identity": screen["identity"],
        }

    selection = {
        "format_version": 1,
        "record_type": "capacity_bounded_bmr10_dual_dataset_selection",
        "status": "FROZEN_OFFLINE_PASS_ONLINE_PENDING",
        "selected_candidate": cap.CANDIDATE_ID,
        "fallback": "C_CNBR_BMR_10",
        "parameters": {
            "min_load_ratio": cap.MIN_LOAD_RATIO,
            "max_load_ratio": cap.MAX_LOAD_RATIO,
            "max_vote_loss": cap.MAX_VOTE_LOSS,
            "max_passes": cap.MAX_PASSES,
        },
        "rows": rows,
        "source_code": {
            "policy": file_record(Path(cap.__file__).resolve()),
            "synthesizer": file_record(Path(__file__).resolve()),
        },
        "online_adoption_required": True,
    }
    output.mkdir(parents=True)
    path = output / "selection-manifest.json"
    path.write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256_path(path)
    (output / "selection-manifest.json.sha256").write_text(
        digest + "\n", encoding="ascii"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    path = run(parse_args(argv))
    print(json.dumps({"selection": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
