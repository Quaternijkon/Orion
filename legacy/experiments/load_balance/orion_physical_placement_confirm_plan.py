#!/usr/bin/env python3
"""Build a five-stage formal confirmation plan from a completed local screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_SOURCE_PLAN = DEFAULT_ROOT / "refine-v3"
DEFAULT_SCREEN_SUMMARY = DEFAULT_ROOT / "online-v3/screen-local-summary.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "confirm-v1-plan"
BASELINE = "controller_aware_v1"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def strategy_mean(summary: dict[str, Any], strategy: str) -> float:
    return float(summary["strategy_summary"][strategy]["qps_mean_across_phases"])


def build(args: argparse.Namespace) -> Path:
    source_dir = args.source_plan.expanduser().resolve()
    screen_path = args.screen_summary.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    source_manifest_path = source_dir / "manifest.json"
    source_placements_path = source_dir / "placements.json"
    source_manifest = load_json(source_manifest_path)
    source = load_json(source_placements_path)
    screen = load_json(screen_path)
    baseline = str(args.baseline)
    if source_manifest.get("status") != "OFFLINE_REFINEMENT_ONLY":
        raise RuntimeError("unexpected source-plan status")
    if screen.get("status") != "SCREEN_COMPLETE":
        raise RuntimeError("local screen is not complete")
    if baseline not in screen["strategy_summary"]:
        raise RuntimeError(f"screen is missing requested baseline: {baseline}")

    finalist = str(screen["screen_winner"])
    if finalist == baseline:
        raise RuntimeError("screen did not identify a challenger to the baseline")
    backup = next(
        strategy
        for strategy in screen["ranking"]
        if strategy not in {baseline, finalist}
    )
    if strategy_mean(screen, finalist) <= strategy_mean(screen, baseline):
        raise RuntimeError("screen winner does not exceed the baseline mean")
    selected = (baseline, finalist, backup)
    missing = [strategy for strategy in selected if strategy not in source["placements"]]
    if missing:
        raise RuntimeError(f"source plan is missing selected strategies: {missing}")

    phases = (
        (f"{baseline.replace('_', '-')}-confirm-a", baseline),
        (f"{finalist.replace('_', '-')}-confirm-a", finalist),
        (f"{backup.replace('_', '-')}-confirm", backup),
        (f"{baseline.replace('_', '-')}-confirm-b", baseline),
        (f"{finalist.replace('_', '-')}-confirm-b", finalist),
    )
    placements_path = output / "placements.json"
    write_json_new(
        placements_path,
        {
            "format_version": 1,
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "controller_peer_id": source["controller_peer_id"],
            "peer_order": source["peer_order"],
            "peer_hosts": source["peer_hosts"],
            "confirmation_phases": [
                {"phase_id": phase_id, "strategy": strategy}
                for phase_id, strategy in phases
            ],
            "placements": {
                strategy: source["placements"][strategy] for strategy in selected
            },
        },
    )
    selection_path = output / "selection.json"
    write_json_new(
        selection_path,
        {
            "status": "READY_FOR_FORMAL_CONFIRMATION",
            "claim_boundary": "screen ranking only; formal five-repeat phases remain required",
            "baseline": baseline,
            "finalist": finalist,
            "backup": backup,
            "screen_qps_mean": {
                strategy: strategy_mean(screen, strategy) for strategy in selected
            },
            "finalist_screen_improvement_pct": 100.0
            * (strategy_mean(screen, finalist) / strategy_mean(screen, baseline) - 1.0),
            "confirmation_rule": (
                "both finalist phases must exceed both baseline phases and all five "
                "held-out Recall@10 values must satisfy the target"
            ),
        },
    )
    manifest_path = output / "manifest.json"
    write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": "execution plan for formal online confirmation; no QPS claim is made here",
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "inputs": {
                "source_manifest": {
                    "path": str(source_manifest_path),
                    "sha256": sha256_path(source_manifest_path),
                },
                "source_placements": {
                    "path": str(source_placements_path),
                    "sha256": sha256_path(source_placements_path),
                },
                "screen_summary": {
                    "path": str(screen_path),
                    "sha256": sha256_path(screen_path),
                },
            },
            "outputs": {
                "placements": {
                    "path": str(placements_path),
                    "sha256": sha256_path(placements_path),
                },
                "selection": {
                    "path": str(selection_path),
                    "sha256": sha256_path(selection_path),
                },
            },
        },
    )
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, default=DEFAULT_SOURCE_PLAN)
    parser.add_argument("--screen-summary", type=Path, default=DEFAULT_SCREEN_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", default=BASELINE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build(parse_args(argv))
    print(json.dumps({"manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
