#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import native_auto_shard_benchmark as benchmark


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")
CASE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
CPUSET_RE = re.compile(r"^[0-9]+(?:[-,][0-9]+)*$")
METHODS = ("hash_all", "orion", "simple_kmeans")
SHARED_ARGUMENTS = {
    "base_url",
    "hdf5_path",
    "topology",
    "deployment_manifest",
    "warmup_query_count",
    "eval_query_count",
    "top_k",
    "batch_size",
    "stability_repeats",
    "api",
    "vector_distance",
    "vector_name",
    "cargo_runner",
    "cargo_target_dir",
    "write_per_query_metrics",
}
CASE_ARGUMENTS = {
    "name",
    "method",
    "collection",
    "artifact",
    "hnsw_ef",
    "orion_route_trace",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or collect a native HashAll/Orion/Simple-KMeans benchmark matrix."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--collect-only", action="store_true")
    parser.add_argument("--taskset-cpus")
    return parser.parse_args(argv)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("matrix config root must be a JSON object")
    shared = data.get("shared")
    cases = data.get("cases")
    if not isinstance(shared, dict):
        raise ValueError("matrix config requires a shared object")
    if not isinstance(cases, list) or not cases:
        raise ValueError("matrix config requires a non-empty cases array")
    unknown_shared = sorted(set(shared) - SHARED_ARGUMENTS)
    if unknown_shared:
        raise ValueError(f"unsupported shared benchmark arguments: {unknown_shared}")
    required_shared = {"base_url", "hdf5_path", "topology"}
    missing_shared = sorted(required_shared - set(shared))
    if missing_shared:
        raise ValueError(f"matrix shared arguments are missing: {missing_shared}")

    names: set[str] = set()
    methods: set[str] = set()
    normalized_cases: list[dict[str, Any]] = []
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            raise ValueError(f"matrix case must be an object: {raw_case!r}")
        unknown_case = sorted(set(raw_case) - CASE_ARGUMENTS)
        if unknown_case:
            raise ValueError(f"unsupported case arguments: {unknown_case}")
        name = str(raw_case.get("name") or "")
        method = str(raw_case.get("method") or "")
        collection = str(raw_case.get("collection") or "")
        if not CASE_NAME_RE.fullmatch(name):
            raise ValueError(f"unsafe or empty case name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate matrix case name: {name}")
        if method not in METHODS:
            raise ValueError(f"unsupported matrix method: {method!r}")
        if not collection:
            raise ValueError(f"case {name} requires collection")
        artifact = raw_case.get("artifact")
        hnsw_ef = raw_case.get("hnsw_ef")
        orion_route_trace = raw_case.get("orion_route_trace")
        if orion_route_trace is not None and not isinstance(orion_route_trace, bool):
            raise ValueError(f"case {name} orion_route_trace must be a boolean")
        if orion_route_trace is not None and method != "orion":
            raise ValueError(
                f"case {name} must not provide orion_route_trace for method {method}"
            )
        if method == "hash_all":
            if artifact is not None:
                raise ValueError(f"HashAll case {name} must not provide artifact")
            if isinstance(hnsw_ef, bool) or not isinstance(hnsw_ef, int) or hnsw_ef <= 0:
                raise ValueError(f"HashAll case {name} requires positive hnsw_ef")
        else:
            if hnsw_ef is not None:
                raise ValueError(f"routed case {name} must not provide hnsw_ef")
        names.add(name)
        methods.add(method)
        normalized_cases.append(dict(raw_case))
    if methods != set(METHODS):
        raise ValueError(
            f"matrix must contain all three methods {list(METHODS)}; found {sorted(methods)}"
        )

    targets = data.get("same_recall_targets", [0.90, 0.95])
    if not isinstance(targets, list) or not targets:
        raise ValueError("same_recall_targets must be a non-empty array")
    normalized_targets = [float(target) for target in targets]
    if any(not 0.0 <= target <= 1.0 for target in normalized_targets):
        raise ValueError("same-recall targets must be between zero and one")
    window = float(data.get("same_recall_window", 0.003))
    if window < 0:
        raise ValueError("same_recall_window must be non-negative")
    return {
        "config_path": str(config_path),
        "shared": dict(shared),
        "cases": normalized_cases,
        "same_recall_targets": normalized_targets,
        "same_recall_window": window,
    }


def matrix_directory(output_root: str | Path, run_id: str, *, must_exist: bool) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"unsafe matrix run id: {run_id!r}")
    root = Path(output_root).expanduser().resolve()
    if root == REPO_ROOT or REPO_ROOT in root.parents:
        raise ValueError("matrix output root must be outside the repository")
    matrix_dir = root / run_id
    if must_exist:
        if not matrix_dir.is_dir():
            raise FileNotFoundError(f"matrix run directory does not exist: {matrix_dir}")
    else:
        matrix_dir.mkdir(parents=True, exist_ok=False)
    return matrix_dir


def config_path_argument(value: Any) -> str:
    return str(value)


def benchmark_command(
    shared: dict[str, Any],
    case: dict[str, Any],
    output_dir: Path,
    taskset_cpus: str | None,
) -> list[str]:
    if taskset_cpus is not None and not CPUSET_RE.fullmatch(taskset_cpus):
        raise ValueError(f"invalid taskset CPU list: {taskset_cpus!r}")
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/native_auto_shard_benchmark.py"),
        "--method",
        str(case["method"]),
        "--collection",
        str(case["collection"]),
        "--output-dir",
        str(output_dir),
    ]
    ordered_shared = [
        "base_url",
        "hdf5_path",
        "topology",
        "deployment_manifest",
        "warmup_query_count",
        "eval_query_count",
        "top_k",
        "batch_size",
        "stability_repeats",
        "api",
        "vector_distance",
        "vector_name",
        "cargo_runner",
        "cargo_target_dir",
    ]
    for key in ordered_shared:
        value = shared.get(key)
        if value is None:
            continue
        command.extend([f"--{key.replace('_', '-')}", config_path_argument(value)])
    if bool(shared.get("write_per_query_metrics")):
        command.append("--write-per-query-metrics")
    if case.get("artifact") is not None:
        command.extend(["--artifact", str(case["artifact"])])
    if case.get("hnsw_ef") is not None:
        command.extend(["--hnsw-ef", str(case["hnsw_ef"])])
    if case.get("orion_route_trace") is True:
        command.append("--orion-route-trace")
    return ["taskset", "-c", taskset_cpus, *command] if taskset_cpus else command


def execute_cases(
    matrix_dir: Path,
    config: dict[str, Any],
    taskset_cpus: str | None,
) -> list[dict[str, Any]]:
    cases_root = matrix_dir / "cases"
    logs_root = matrix_dir / "logs"
    cases_root.mkdir()
    logs_root.mkdir()
    records: list[dict[str, Any]] = []
    for case in config["cases"]:
        case_output = cases_root / str(case["name"])
        command = benchmark_command(config["shared"], case, case_output, taskset_cpus)
        started = time.time()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        (logs_root / f"{case['name']}.stdout.log").write_text(
            result.stdout, encoding="utf-8"
        )
        (logs_root / f"{case['name']}.stderr.log").write_text(
            result.stderr, encoding="utf-8"
        )
        record = {
            "name": case["name"],
            "method": case["method"],
            "collection": case["collection"],
            "artifact": case.get("artifact"),
            "hnsw_ef": case.get("hnsw_ef"),
            "orion_route_trace": case.get("orion_route_trace", False),
            "output_dir": str(case_output),
            "command": command,
            "command_shell": shlex.join(command),
            "returncode": result.returncode,
            "started_epoch": started,
            "completed_epoch": time.time(),
        }
        records.append(record)
        if result.returncode != 0:
            raise RuntimeError(
                f"native benchmark case {case['name']} failed with exit code "
                f"{result.returncode}; inspect {logs_root}"
            )
    return records


def parse_csv_value(value: str | None) -> Any:
    if value is None or value.strip() == "":
        return None
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() and all(c not in text.lower() for c in ".e") else number


def load_case_result(matrix_dir: Path, case: dict[str, Any]) -> dict[str, Any]:
    case_dir = matrix_dir / "cases" / str(case["name"])
    manifest_path = case_dir / "run_manifest.json"
    metrics_path = case_dir / "final_metrics.csv"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"case {case['name']} is missing run_manifest.json or final_metrics.csv"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"case {case['name']} final_metrics.csv must contain one row")
    metrics = {key: parse_csv_value(value) for key, value in rows[0].items()}
    if str(manifest.get("method")) != str(case["method"]):
        raise RuntimeError(
            f"case {case['name']} run manifest method mismatch: "
            f"config={case['method']}, manifest={manifest.get('method')}"
        )
    if str(manifest.get("collection")) != str(case["collection"]):
        raise RuntimeError(
            f"case {case['name']} run manifest collection mismatch: "
            f"config={case['collection']}, manifest={manifest.get('collection')}"
        )
    if str(metrics.get("method")) != str(case["method"]):
        raise RuntimeError(
            f"case {case['name']} method mismatch: config={case['method']}, "
            f"metrics={metrics.get('method')}"
        )
    if case.get("orion_route_trace") is True:
        trace = manifest.get("orion_route_trace")
        if not isinstance(trace, dict) or trace.get("status") != "verified":
            raise RuntimeError(
                f"case {case['name']} requires a verified Orion route trace manifest"
            )
        expected_source = benchmark.ORION_ROUTE_TRACE_SOURCE
        if (
            metrics.get("visited_shards_source") != expected_source
            or metrics.get("ef_sum_source") != expected_source
            or not isinstance(metrics.get("visited_shards"), (int, float))
            or not isinstance(metrics.get("ef_sum_per_query"), (int, float))
        ):
            raise RuntimeError(
                f"case {case['name']} is missing exact Orion route-trace metrics"
            )
    point = {
        "case_name": case["name"],
        "method": case["method"],
        "collection": case["collection"],
        "artifact": case.get("artifact"),
        "hnsw_ef": case.get("hnsw_ef"),
        "orion_route_trace": case.get("orion_route_trace", False),
        **metrics,
        "raw_output_dir": str(case_dir),
        "run_manifest_path": str(manifest_path),
    }
    return {"case": case, "manifest": manifest, "point": point}


def provenance_fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    dataset = manifest.get("dataset") or {}
    deployment = manifest.get("deployment") or {}
    image = deployment.get("image") or {}
    repository = deployment.get("repository") or {}
    benchmark_repository = manifest.get("repository") or {}
    parameters = manifest.get("parameters") or {}
    cluster = manifest.get("cluster_preflight") or {}
    topology = manifest.get("topology")
    image_identity = image.get("id") or image.get("digest")
    if not dataset.get("sha256"):
        raise RuntimeError("case manifest is missing dataset SHA-256 provenance")
    if not image_identity:
        raise RuntimeError("case manifest is missing deployment image identity")
    if not repository.get("commit"):
        raise RuntimeError("case manifest is missing deployment commit provenance")
    if not benchmark_repository.get("commit"):
        raise RuntimeError("case manifest is missing benchmark-client commit provenance")
    if not isinstance(topology, dict):
        raise RuntimeError("case manifest is missing topology provenance")
    process_affinity = manifest.get("process_affinity")
    if not isinstance(process_affinity, list) or not process_affinity:
        raise RuntimeError("case manifest is missing benchmark CPU-affinity provenance")
    return {
        "dataset_sha256": dataset["sha256"],
        "dataset_shapes": {
            "train": dataset.get("train_shape"),
            "test": dataset.get("test_shape"),
            "neighbors": dataset.get("neighbors_shape"),
        },
        "image_identity": image_identity,
        "image_tag": image.get("tag"),
        "deployment_commit": repository.get("commit"),
        "benchmark_commit": benchmark_repository.get("commit"),
        "benchmark_dirty": benchmark_repository.get("dirty"),
        "topology_sha256": canonical_sha256(topology),
        "peer_uris": cluster.get("peers"),
        "process_affinity": process_affinity,
        "api": manifest.get("api"),
        "vector_distance": parameters.get("vector_distance"),
        "vector_name": parameters.get("vector_name"),
        "eval_query_count": parameters.get("eval_query_count"),
        "top_k": parameters.get("top_k"),
        "batch_size": parameters.get("batch_size"),
    }


def validate_shared_provenance(results: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {str(result["point"]["method"]) for result in results}
    if methods != set(METHODS):
        raise RuntimeError(f"collected results do not contain all three methods: {sorted(methods)}")
    fingerprints = [provenance_fingerprint(result["manifest"]) for result in results]
    expected = fingerprints[0]
    mismatches = [
        {
            "case": results[index]["point"]["case_name"],
            "actual": fingerprint,
        }
        for index, fingerprint in enumerate(fingerprints[1:], start=1)
        if fingerprint != expected
    ]
    if mismatches:
        raise RuntimeError(
            "native matrix shared provenance mismatch: "
            + json.dumps({"expected": expected, "mismatches": mismatches}, sort_keys=True)
        )
    return expected


def pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for point in points:
        recall = float(point["recall_at_k"])
        qps = float(point["qps"])
        dominated = any(
            other is not point
            and float(other["recall_at_k"]) >= recall
            and float(other["qps"]) >= qps
            and (
                float(other["recall_at_k"]) > recall or float(other["qps"]) > qps
            )
            for other in points
        )
        if not dominated:
            frontier.append(dict(point))
    return sorted(frontier, key=lambda row: (float(row["recall_at_k"]), -float(row["qps"])))


def same_recall_selection(
    frontier: list[dict[str, Any]], target: float, window: float
) -> dict[str, Any]:
    within = [
        point for point in frontier if abs(float(point["recall_at_k"]) - target) <= window
    ]
    if within:
        selected = max(within, key=lambda row: float(row["qps"]))
        status = "strict"
    else:
        selected = min(
            frontier,
            key=lambda row: (
                abs(float(row["recall_at_k"]) - target),
                -float(row["qps"]),
            ),
        )
        status = "nearest"
    return {
        **selected,
        "target_recall": target,
        "recall_delta": float(selected["recall_at_k"]) - target,
        "recall_match_status": status,
        "recall_window": window,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_plots(matrix_dir: Path, points: list[dict[str, Any]]) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/orion-native-matrix-matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate native matrix plots") from exc
    specifications = [
        ("qps", "QPS", "recall_qps.png"),
        ("latency_p95_ms", "P95 latency (ms)", "recall_latency.png"),
        ("visited_shards", "Visited logical shards/query", "recall_visited_shards.png"),
        ("ef_sum_per_query", "EF sum/query", "recall_ef_sum.png"),
    ]
    written: list[str] = []
    for field, label, filename in specifications:
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        plotted = False
        for method in METHODS:
            xy = [
                (float(row["recall_at_k"]), float(row[field]))
                for row in points
                if row["method"] == method
                and isinstance(row.get(field), (int, float))
                and math.isfinite(float(row[field]))
            ]
            if not xy:
                continue
            xy.sort()
            axis.plot([x for x, _ in xy], [y for _, y in xy], marker="o", label=method)
            plotted = True
        if plotted:
            axis.set_xlabel("Recall@K")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figure.savefig(matrix_dir / filename, dpi=160)
            written.append(filename)
        plt.close(figure)
    return written


def collect_results(
    matrix_dir: Path,
    config: dict[str, Any],
    case_records: list[dict[str, Any]] | None,
    run_id: str,
    mode: str,
    taskset_cpus: str | None,
) -> dict[str, Any]:
    results = [load_case_result(matrix_dir, case) for case in config["cases"]]
    shared_provenance = validate_shared_provenance(results)
    points = [result["point"] for result in results]
    frontiers: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for method in METHODS:
        method_frontier = pareto_frontier(
            [point for point in points if point["method"] == method]
        )
        frontiers.extend(method_frontier)
        for target in config["same_recall_targets"]:
            selections.append(
                same_recall_selection(
                    method_frontier,
                    float(target),
                    float(config["same_recall_window"]),
                )
            )
    write_csv(matrix_dir / "recall_qps_points.csv", points)
    write_csv(matrix_dir / "pareto_frontier.csv", frontiers)
    write_csv(matrix_dir / "same_recall_selection.csv", selections)
    plots = write_plots(matrix_dir, points)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": config["config_path"],
        "config_sha256": hashlib.sha256(
            Path(config["config_path"]).read_bytes()
        ).hexdigest(),
        "shared": config["shared"],
        "shared_provenance": shared_provenance,
        "same_recall_targets": config["same_recall_targets"],
        "same_recall_window": config["same_recall_window"],
        "taskset_cpus": taskset_cpus,
        "cases": case_records
        or [
            {
                **case,
                "output_dir": str(matrix_dir / "cases" / str(case["name"])),
            }
            for case in config["cases"]
        ],
        "artifacts": {
            "recall_qps_points": "recall_qps_points.csv",
            "pareto_frontier": "pareto_frontier.csv",
            "same_recall_selection": "same_recall_selection.csv",
            "plots": plots,
        },
    }
    temporary = matrix_dir / "run_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(matrix_dir / "run_manifest.json")
    return manifest


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    if args.taskset_cpus is not None and not CPUSET_RE.fullmatch(args.taskset_cpus):
        raise ValueError(f"invalid taskset CPU list: {args.taskset_cpus!r}")
    matrix_dir = matrix_directory(
        args.output_root,
        args.run_id,
        must_exist=bool(args.collect_only),
    )
    case_records = (
        None
        if args.collect_only
        else execute_cases(matrix_dir, config, args.taskset_cpus)
    )
    collect_results(
        matrix_dir,
        config,
        case_records,
        args.run_id,
        "collect_only" if args.collect_only else "run",
        args.taskset_cpus,
    )
    return matrix_dir


def main(argv: list[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(json.dumps({"matrix_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
