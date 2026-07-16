#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, NamedTuple


class BenchmarkRun(NamedTuple):
    run_id: str
    case_name: str
    preset: str
    args: dict[str, Any]
    output_dir: Path
    tags: dict[str, Any]


PRESET_ARGS: dict[str, dict[str, Any]] = {
    "orion": {
        "routing_mode": "faithful_original_rest",
        "routed_execution_mode": "compact_multi_ep",
        "routed_planning_mode": "materialized",
        "routed_result_limit_mode": "top_k",
        "search_dispatch_mode": "coordinator",
    },
    "naive": {
        "routing_mode": "naive_hash_all_shards",
        "search_dispatch_mode": "coordinator",
    },
    # KMeans-only ablation inside the Method4 routing framework: keep upper
    # routing / voting assignment path, but disable topology convergence and
    # fission so the L1 shard map remains the balanced KMeans initialization.
    "kmeans": {
        "routing_mode": "faithful_original_rest",
        "topology_iters": 0,
        "disable_fission": True,
        "routed_execution_mode": "compact_multi_ep",
        "routed_planning_mode": "materialized",
        "routed_result_limit_mode": "top_k",
        "search_dispatch_mode": "coordinator",
    },
    "method4_kmeans_ablation": {
        "routing_mode": "faithful_original_rest",
        "topology_iters": 0,
        "disable_fission": True,
        "routed_execution_mode": "compact_multi_ep",
        "routed_planning_mode": "materialized",
        "routed_result_limit_mode": "top_k",
        "search_dispatch_mode": "coordinator",
    },
    "kmeans_independent": {
        "routing_mode": "cpp_kmeans_baseline",
        "routed_execution_mode": "compact_multi_ep",
        "routed_planning_mode": "materialized",
        "routed_result_limit_mode": "top_k",
        "search_dispatch_mode": "coordinator",
    },
    "kmeans_simple_nprobe": {
        "routing_mode": "kmeans_simple_nprobe",
        "search_dispatch_mode": "coordinator",
    },
    # Legacy centroid routing is intentionally separate from the KMeans-only
    # Method4 ablation because it uses a different older routing pipeline.
    "legacy_centroid": {
        "routing_mode": "legacy_centroid",
        "routed_execution_mode": "compact_multi_ep",
        "search_dispatch_mode": "coordinator",
    },
}

PRESET_TAGS: dict[str, dict[str, Any]] = {
    "orion": {"partition_family": "orion"},
    "naive": {"partition_family": "naive_all_shards"},
    "kmeans": {"partition_family": "method4_kmeans_ablation"},
    "method4_kmeans_ablation": {"partition_family": "method4_kmeans_ablation"},
    "kmeans_independent": {"partition_family": "kmeans_independent"},
    "kmeans_simple_nprobe": {"partition_family": "kmeans_simple_nprobe"},
    "legacy_centroid": {"partition_family": "legacy_centroid"},
}

SUMMARY_FIELDS = [
    "run_id",
    "case_name",
    "preset",
    "partition_family",
    "routing_mode",
    "target_recall",
    "num_shards",
    "effective_num_shards",
    "upper_k",
    "base_ef",
    "factor",
    "batch_size",
    "recall",
    "recall_stdev",
    "qps",
    "qps_stdev",
    "final_qps",
    "avg_visited_shards",
    "avg_ef_per_visited_shard",
    "search_batch_calls",
    "index_expansion_ratio",
    "summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand and run Orion/naive/KMeans benchmark matrices on top of "
            "tools/qdrant_two_level_routing_experiment.py."
        )
    )
    parser.add_argument("--config", required=True, help="JSON benchmark matrix config.")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root for generated run directories. Defaults to config output_root or "
            "results/method4_benchmark_matrix."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional matrix run id. Defaults to current UTC-like timestamp.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually execute commands. Without this flag the tool only writes commands and manifest.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Do not execute; collect summary.json files from the expanded output dirs.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable. Defaults to config python or python3.",
    )
    parser.add_argument(
        "--harness",
        default=None,
        help=(
            "Experiment harness path. Defaults to config harness or "
            "tools/qdrant_two_level_routing_experiment.py."
        ),
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"benchmark config must be a JSON object: {path}")
    return data


def normalize_key(key: str) -> str:
    return key.strip().lstrip("-").replace("-", "_")


def normalize_args(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    return {normalize_key(str(key)): value for key, value in raw.items()}


def merge_args(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            merged[normalize_key(key)] = value
    return merged


def values_for_matrix(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def expand_matrix(matrix: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    keys = [normalize_key(key) for key in matrix]
    value_lists = [values_for_matrix(value) for value in matrix.values()]
    rows: list[dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        rows.append(dict(zip(keys, values)))
    return rows


def id_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:g}".replace(".", "p")
    else:
        text = str(value)
    allowed = []
    for char in text:
        if char.isalnum():
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "x"


def build_run_id(case_name: str, matrix_row: dict[str, Any]) -> str:
    if not matrix_row:
        return id_value(case_name)
    suffix = "__".join(f"{key}-{id_value(value)}" for key, value in matrix_row.items())
    return f"{id_value(case_name)}__{suffix}"


def format_collection(template: str, case_name: str, preset: str, run_id: str, args: dict[str, Any], tags: dict[str, Any]) -> str:
    context = {
        "case": case_name,
        "preset": preset,
        "run_id": run_id,
        **tags,
        **args,
    }
    return template.format(**context)


def expand_runs(spec: dict[str, Any], output_root: Path) -> list[BenchmarkRun]:
    defaults = normalize_args(spec.get("defaults"))
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark config requires a non-empty cases list")

    runs: list[BenchmarkRun] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be a JSON object")
        case_name = str(case.get("name") or case.get("preset") or f"case_{len(runs)}")
        preset = str(case.get("preset") or "orion")
        if preset not in PRESET_ARGS:
            raise ValueError(f"unknown preset {preset!r}; expected one of {sorted(PRESET_ARGS)}")

        preset_args = dict(PRESET_ARGS[preset])
        case_args = normalize_args(case.get("args"))
        case_tags = {
            **PRESET_TAGS.get(preset, {}),
            **normalize_args(case.get("tags")),
        }
        collection_template = case.get("collection_template")
        matrix_rows = expand_matrix(case.get("matrix"))

        for matrix_row in matrix_rows:
            run_id = build_run_id(case_name, matrix_row)
            args = merge_args(defaults, preset_args, case_args, matrix_row)
            if collection_template and "collection" not in args:
                args["collection"] = format_collection(
                    str(collection_template),
                    case_name,
                    preset,
                    run_id,
                    args,
                    case_tags,
                )
            output_dir = output_root / run_id
            runs.append(
                BenchmarkRun(
                    run_id=run_id,
                    case_name=case_name,
                    preset=preset,
                    args=args,
                    output_dir=output_dir,
                    tags=case_tags,
                )
            )
    return runs


def cli_key(key: str) -> str:
    return "--" + key.replace("_", "-")


def str_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def render_command(
    run: BenchmarkRun,
    python_executable: str,
    harness: str,
) -> list[str]:
    command = [python_executable, harness]
    for key, value in run.args.items():
        if value is None or value is False:
            continue
        if key == "output_dir":
            continue
        flag = cli_key(key)
        if value is True:
            command.append(flag)
        elif isinstance(value, list):
            command.append(flag)
            command.extend(str_value(item) for item in value)
        else:
            command.extend([flag, str_value(value)])
    command.extend(["--output-dir", str(run.output_dir)])
    return command


def latest_summary_path(output_dir: Path) -> Path | None:
    direct = output_dir / "summary.json"
    if direct.exists():
        return direct
    candidates = sorted(
        output_dir.glob("*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def first_number(*values: Any) -> Any:
    for value in values:
        if isinstance(value, (int, float)):
            return value
    return ""


def collect_summary_row(run: BenchmarkRun, summary_path: Path) -> dict[str, Any]:
    data = load_json(summary_path)
    final_row = data.get("final_row") if isinstance(data.get("final_row"), dict) else {}
    best_row = data.get("best_tuning_row") if isinstance(data.get("best_tuning_row"), dict) else {}
    stability = data.get("stability_summary") if isinstance(data.get("stability_summary"), dict) else {}
    original_routing = data.get("original_routing") if isinstance(data.get("original_routing"), dict) else {}
    simple_kmeans = data.get("kmeans_simple_nprobe") if isinstance(data.get("kmeans_simple_nprobe"), dict) else {}

    row = {
        "run_id": run.run_id,
        "case_name": run.case_name,
        "preset": run.preset,
        "partition_family": run.tags.get("partition_family", ""),
        "routing_mode": data.get("routing_mode", run.args.get("routing_mode", "")),
        "target_recall": run.args.get("target_recall", ""),
        "num_shards": data.get("initial_num_shards", run.args.get("num_shards", "")),
        "effective_num_shards": data.get("num_shards", ""),
        "upper_k": first_number(final_row.get("upper_k"), best_row.get("upper_k")),
        "base_ef": first_number(final_row.get("base_ef"), best_row.get("base_ef")),
        "factor": first_number(final_row.get("factor"), best_row.get("factor")),
        "batch_size": run.args.get("batch_size", ""),
        "recall": first_number(stability.get("recall_mean"), final_row.get("recall_at_k"), final_row.get("recall")),
        "recall_stdev": first_number(stability.get("recall_stdev")),
        "qps": first_number(stability.get("qps_mean"), final_row.get("qps")),
        "qps_stdev": first_number(stability.get("qps_stdev")),
        "final_qps": first_number(final_row.get("qps")),
        "avg_visited_shards": first_number(final_row.get("avg_visited_shards")),
        "avg_ef_per_visited_shard": first_number(
            final_row.get("avg_assigned_ef_per_visited_shard"),
            final_row.get("avg_ef_per_visited_shard"),
        ),
        "search_batch_calls": first_number(final_row.get("search_batch_calls")),
        "index_expansion_ratio": first_number(
            original_routing.get("expansion_ratio"),
            simple_kmeans.get("expansion_ratio"),
        ),
        "summary_path": str(summary_path),
    }
    return row


def collect_rows(runs: list[BenchmarkRun]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        summary_path = latest_summary_path(run.output_dir)
        if summary_path is None:
            rows.append(
                {
                    "run_id": run.run_id,
                    "case_name": run.case_name,
                    "preset": run.preset,
                    "partition_family": run.tags.get("partition_family", ""),
                    "summary_path": "",
                }
            )
            continue
        rows.append(collect_summary_row(run, summary_path))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = SUMMARY_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(
    matrix_dir: Path,
    spec: dict[str, Any],
    runs: list[BenchmarkRun],
    commands: list[list[str]],
) -> None:
    matrix_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "matrix_name": spec.get("matrix_name") or spec.get("name") or "method4_matrix",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": [
            {
                "run_id": run.run_id,
                "case_name": run.case_name,
                "preset": run.preset,
                "args": run.args,
                "output_dir": str(run.output_dir),
                "tags": run.tags,
                "command": command,
            }
            for run, command in zip(runs, commands)
        ],
    }
    (matrix_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (matrix_dir / "commands.sh").open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for command in commands:
            handle.write(shlex.join(command))
            handle.write("\n")


def execute_runs(runs: list[BenchmarkRun], commands: list[list[str]]) -> None:
    for run, command in zip(runs, commands):
        print(f"[run] {run.run_id}: {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    spec = load_json(args.config)
    matrix_name = str(spec.get("matrix_name") or spec.get("name") or Path(args.config).stem)
    output_root = Path(args.output_root or spec.get("output_root") or "results/method4_benchmark_matrix")
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    matrix_dir = output_root / matrix_name / run_id
    python_executable = args.python or spec.get("python") or "python3"
    harness = args.harness or spec.get("harness") or "tools/qdrant_two_level_routing_experiment.py"

    runs = expand_runs(spec, matrix_dir)
    commands = [render_command(run, python_executable, harness) for run in runs]
    write_manifest(matrix_dir, spec, runs, commands)

    if args.run and not args.collect_only:
        execute_runs(runs, commands)

    rows = collect_rows(runs)
    write_csv(matrix_dir / "matrix_summary.csv", rows)
    print(f"Wrote matrix files to: {matrix_dir}")
    print(f"Runs: {len(runs)}")
    if not args.run and not args.collect_only:
        print("Dry run only. Re-run with --run to execute commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
