#!/usr/bin/env python3
"""Plan or build native runtime profiles with independent upper EF and top-k.

`plan` only emits a JSON plan. `build` explicitly invokes the Rust profile CLI;
neither command installs a profile, starts servers, or runs a benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess
from typing import Any


MAX_USIZE = (1 << (8 * struct.calcsize("P"))) - 1
MAX_GENERATION = (1 << 64) - 1


def positive(value: Any, name: str) -> int:
    if type(value) is not int or not 0 < value <= MAX_USIZE:
        raise ValueError(f"{name} must be a positive usize")
    return value


def positive_values(config: dict[str, Any], name: str) -> list[int]:
    values = config[name]
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    result = [positive(value, name) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates")
    return result


def make_plan(
    config: dict[str, Any],
    source: Path,
    output_root: Path,
    binary: Path,
    first_generation: int,
) -> dict[str, Any]:
    allowed = {
        "schema_version", "name", "dataset", "execution_mode",
        "upper_ef_search", "upper_k_values", "dynamic_ef_base_values",
        "dynamic_ef_factor_values", "experiment_requirements",
    }
    if set(config) - allowed:
        raise ValueError(f"unknown config fields: {sorted(set(config) - allowed)}")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema_version")
    if config.get("execution_mode") != "manual_only":
        raise ValueError("execution_mode must be manual_only")
    upper_ef = positive(config["upper_ef_search"], "upper_ef_search")
    ks = positive_values(config, "upper_k_values")
    bases = positive_values(config, "dynamic_ef_base_values")
    factors = positive_values(config, "dynamic_ef_factor_values")
    source = source.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    binary = binary.expanduser().resolve()
    manifest_bytes = (source / "build-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    name = manifest["import_manifest_file"]
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
        raise ValueError("invalid import manifest filename")
    imported = json.loads((source / name).read_text())
    source_generation = imported["orion_generation"]
    if type(source_generation) is not int or not 0 < source_generation <= MAX_GENERATION:
        raise ValueError("invalid source generation")
    if type(first_generation) is not int or not source_generation < first_generation <= MAX_GENERATION:
        raise ValueError("first_generation must exceed the source generation and fit u64")
    upper_count = len(manifest["upper_indices"])
    if any(k > upper_ef or k > upper_count for k in ks):
        raise ValueError("every upper_k must be <= upper_ef_search and the upper node count")
    if any(base + factor * max(ks) > MAX_USIZE for base in bases for factor in factors):
        raise ValueError("linear lower EF overflows usize")
    combinations = list(itertools.product(ks, bases, factors))
    if first_generation + len(combinations) - 1 > MAX_GENERATION:
        raise ValueError("profile generations overflow u64")
    cases = []
    for index, (k, base, factor) in enumerate(combinations):
        generation = first_generation + index
        label = f"k{k}-uef{upper_ef}-b{base}-f{factor}-g{generation}"
        destination = output_root / label
        command = [
            str(binary), "profile", "--source", str(source),
            "--output", str(destination), "--generation", str(generation),
            "--upper-k", str(k), "--upper-ef-search", str(upper_ef),
            "--dynamic-ef-base", str(base), "--dynamic-ef-factor", str(factor),
        ]
        cases.append({
            "name": label, "generation": generation,
            "upper_k": k, "upper_ef_search": upper_ef,
            "dynamic_ef_base": base, "dynamic_ef_factor": factor,
            "maximum_requested_lower_ef": base + factor * k,
            "output": str(destination), "command": command,
        })
    return {
        "schema_version": 1, "state": "planned",
        "execution_mode": "manual_only", "source": str(source),
        "source_generation": source_generation,
        "source_build_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "upper_graph_sha256": manifest["upper_graph_sha256"],
        "output_root": str(output_root), "native_layout_binary": str(binary),
        "config": config,
        "lower_ef_rule": "base + factor * retained upper hits owned by this shard",
        "search_request_hnsw_ef_override": None,
        "cases": cases,
    }


def build_profiles(plan: dict[str, Any]) -> None:
    """Run only on an explicit `build` command; no installation or remote calls."""
    binary = Path(plan["native_layout_binary"])
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"native layout binary is not executable: {binary}")
    root = Path(plan["output_root"])
    for case in plan["cases"]:
        for path in (Path(case["output"]), root / (case["name"] + ".log")):
            if path.exists() or path.is_symlink():
                raise FileExistsError(path)
    root.mkdir(parents=True, exist_ok=True)
    for case in plan["cases"]:
        with (root / (case["name"] + ".log")).open("x") as log:
            subprocess.run(case["command"], stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Built {case['name']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "build"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="Existing immutable native bundle")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--native-layout-binary", required=True, type=Path)
    parser.add_argument("--first-generation", required=True, type=int,
                        help="Must exceed source AND currently active cluster generations")
    parser.add_argument("--plan-file", type=Path, help="Write the reviewable JSON plan to a new file")
    args = parser.parse_args(argv)
    plan = make_plan(json.loads(args.config.read_text()), args.source, args.output_root,
                     args.native_layout_binary, args.first_generation)
    if args.plan_file:
        args.plan_file.parent.mkdir(parents=True, exist_ok=True)
        with args.plan_file.open("x") as out:
            json.dump(plan, out, indent=2)
            out.write("\n")
    if args.action == "plan":
        print(json.dumps(plan, indent=2))
    else:
        build_profiles(plan)


if __name__ == "__main__":
    main()
