#!/usr/bin/env python3
"""Build Method4 claim cost tables from existing artifacts and live metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DATA_CONTAINER_MARKER = "qdrant_shard_"
CONTROLLER_CONTAINER_MARKER = "qdrant_controller"
MISSING_COST_METRICS = "build_time;memory_rss;request_bytes;controller_cpu"
BUILD_WALL_TIME_FIELDS = {
    "build_secs",
    "create_secs",
    "upsert_secs",
    "wait_index_secs",
    "total_build_secs",
    "duration_secs",
    "elapsed_secs",
    "wall_secs",
    "wall_s",
}
BUILD_RSS_TOKENS = ("rss", "resident", "maxrss", "maximum_resident")


DEFAULT_CASES: list[dict[str, str]] = [
    {
        "label": "bench095_orion_default_095",
        "family": "Orion",
        "strategy": "default_rr_s31",
        "collection": "bench095_rr_orion_s31",
        "build_path": "results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/orion_r095_eff46/20260623_124839/builds.csv",
        "performance_path": "results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/orion_r095_eff46/20260623_124839/final_metrics.csv",
    },
    {
        "label": "bench095_simple_kmeans_095",
        "family": "Simple KMeans",
        "strategy": "nprobe_s46",
        "collection": "bench095_cpp_kmeans_s46",
        "build_path": "results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/kmeans_simple_r095_s46/20260623_123127/builds.csv",
        "performance_path": "results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/kmeans_simple_r095_s46/20260623_123127/final_metrics.csv",
    },
    {
        "label": "bench095_naive_all_shards_095",
        "family": "Naive",
        "strategy": "all_shards_s46",
        "collection": "bench095_rr_naive_s46",
        "build_path": "results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/naive_r095_s46/20260623_124612/builds.csv",
        "performance_path": "results/method4_benchmark_matrix/method4_recall_sweep_simple_kmeans_nprobe_20260623/simple_nprobe_20260623/naive_r095_s46/20260623_124612/final_metrics.csv",
    },
    {
        "label": "current_method4_map_095",
        "family": "Orion",
        "strategy": "current_method4map",
        "collection": "qdrant_controller_idea_method4map_full_20260601",
        "build_path": "results/qdrant_compare_current_idea_vs_naive_095_idea_b200/20260603_094930/builds.csv",
        "performance_path": "results/qdrant_compare_current_idea_vs_naive_095_idea_b200/20260603_094930/final_metrics.csv",
    },
    {
        "label": "current_naive_all_shards_095",
        "family": "Naive",
        "strategy": "current_naive46",
        "collection": "qdrant_controller_naive46_full_20260521",
        "build_path": "results/qdrant_compare_current_idea_vs_naive_095_naive_ef76_b100/20260603_095227/builds.csv",
        "performance_path": "results/qdrant_compare_current_idea_vs_naive_095_naive_ef76_b100/20260603_095227/final_metrics.csv",
    },
    {
        "label": "orion_multiassign_default_095",
        "family": "Orion",
        "strategy": "orion_default",
        "collection": "bench_ma_orion_o118_default_s31",
        "build_path": "results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/orion_o118_default_r095/20260629_111048/builds.csv",
        "performance_path": "results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_low_recover/eval3000_20260629/orion_o118_default_r095/20260629_111048/final_metrics.csv",
    },
    {
        "label": "orion_multiassign_w2c2_095",
        "family": "Orion",
        "strategy": "orion_w2c2",
        "collection": "bench_ma_orion_o149_w2c2_s31",
        "build_path": "results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o149_w2c2_r095/20260629_112639/builds.csv",
        "performance_path": "results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o149_w2c2_r095/20260629_112639/final_metrics.csv",
    },
    {
        "label": "orion_multiassign_w2c3_095",
        "family": "Orion",
        "strategy": "orion_w2c3",
        "collection": "bench_ma_orion_o184_w2c3_s31",
        "build_path": "results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o184_w2c3_r095/20260629_113958/builds.csv",
        "performance_path": "results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_orion_high_recover/eval3000_20260629/orion_o184_w2c3_r095/20260629_113958/final_metrics.csv",
    },
    {
        "label": "simple_multiassign_a1014_095",
        "family": "Simple KMeans",
        "strategy": "simple_a1.014",
        "collection": "bench_ma_simple_s189_a1014_s46",
        "build_path": "results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/simple_s189_a1014_r095/20260629_123900/builds.csv",
        "performance_path": "results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_simple_s189_only/eval3000_20260629/simple_s189_a1014_r095/20260629_123900/final_metrics.csv",
    },
]


def int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_build_row(row: dict[str, Any]) -> dict[str, Any]:
    points = int_or_none(row.get("points_count")) or 0
    indexed = int_or_none(row.get("indexed_vectors_count")) or points
    logical = int_or_none(row.get("logical_points_count")) or points
    assigned = int_or_none(row.get("assigned_points_count")) or indexed or points
    expansion = float(assigned / logical) if logical else None
    return {
        "collection": row.get("collection", ""),
        "points_count": points,
        "indexed_vectors_count": indexed,
        "segments_count": int_or_none(row.get("segments_count")) or 0,
        "logical_points_count": logical,
        "assigned_points_count": assigned,
        "index_expansion_ratio": expansion,
        "cluster_shard_count": int_or_none(row.get("cluster_shard_count")),
        "cluster_peer_count": int_or_none(row.get("cluster_peer_count")),
        "cluster_active_shards": int_or_none(row.get("cluster_active_shards")),
        "shard_placement": row.get("shard_placement", ""),
    }


def read_first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def build_artifact_capabilities(path: Path, fieldnames: list[str]) -> dict[str, Any]:
    fields = {field for field in fieldnames if field}
    lowered = {field.lower() for field in fields}
    has_build_wall_time = bool(lowered & BUILD_WALL_TIME_FIELDS)
    has_build_rss = any(any(token in field for token in BUILD_RSS_TOKENS) for field in lowered)

    if "results/hnsw/" in path.as_posix() or "total_build_secs" in lowered:
        artifact_family = "hnsw_baseline_build_timing"
        applicability = "not_method4_distributed"
        conclusion = (
            "Contains standalone HNSW/Qdrant build duration columns, but it is not a Method4 "
            "distributed collection build artifact and has no build-stage RSS."
        )
    elif {"collection", "points_count", "indexed_vectors_count"} <= lowered and (
        "cluster_shard_count" in lowered or "assigned_points_count" in lowered
    ):
        artifact_family = "method4_collection_snapshot"
        applicability = "direct_count_snapshot_only"
        conclusion = (
            "Method4/Qdrant collection snapshot: indexed-vector, assignment, shard, and peer "
            "counts are usable, but there is no build duration or build-stage RSS."
        )
    else:
        artifact_family = "legacy_collection_snapshot"
        applicability = "indirect_or_not_method4"
        conclusion = "Legacy collection build snapshot with insufficient build-time/RSS fields for Method4 cost evidence."

    return {
        "artifact_family": artifact_family,
        "has_build_wall_time": has_build_wall_time,
        "has_build_rss": has_build_rss,
        "method4_cost_applicability": applicability,
        "conclusion": conclusion,
    }


def collect_build_stage_artifact_audit(results_root: Path) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for path in sorted(results_root.rglob("builds.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            row_count = sum(1 for _ in reader)
        caps = build_artifact_capabilities(path, fieldnames)
        family = caps["artifact_family"]
        group = groups.setdefault(
            family,
            {
                "artifact_family": family,
                "file_count": 0,
                "row_count": 0,
                "example_path": str(path),
                "schema_columns": ";".join(fieldnames),
                "has_build_wall_time": caps["has_build_wall_time"],
                "has_build_rss": caps["has_build_rss"],
                "method4_cost_applicability": caps["method4_cost_applicability"],
                "conclusion": caps["conclusion"],
            },
        )
        group["file_count"] += 1
        group["row_count"] += row_count
        group["has_build_wall_time"] = bool(group["has_build_wall_time"] or caps["has_build_wall_time"])
        group["has_build_rss"] = bool(group["has_build_rss"] or caps["has_build_rss"])

    order = {
        "method4_collection_snapshot": 0,
        "hnsw_baseline_build_timing": 1,
        "legacy_collection_snapshot": 2,
    }
    return sorted(groups.values(), key=lambda row: (order.get(str(row["artifact_family"]), 99), str(row["artifact_family"])))


def build_source_audit_rows(cases: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        build_path = Path(case["build_path"])
        if build_path.exists():
            with build_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
        else:
            fieldnames = []
        caps = build_artifact_capabilities(build_path, fieldnames)
        fields_lower = {field.lower() for field in fieldnames}
        has_indexing_time = bool(fields_lower & {"wait_index_secs", "indexing_secs", "index_build_secs"})
        rows.append(
            {
                "label": case["label"],
                "family": case["family"],
                "strategy": case["strategy"],
                "collection": case["collection"],
                "build_source": str(build_path),
                "build_source_schema": ";".join(fieldnames),
                "artifact_family": caps["artifact_family"],
                "has_build_wall_time": caps["has_build_wall_time"],
                "has_indexing_time": has_indexing_time,
                "has_build_rss": caps["has_build_rss"],
                "safe_to_fill_cost_build_time": bool(
                    caps["has_build_wall_time"] and caps["method4_cost_applicability"] != "not_method4_distributed"
                ),
                "safe_to_fill_cost_build_rss": bool(
                    caps["has_build_rss"] and caps["method4_cost_applicability"] != "not_method4_distributed"
                ),
                "notes": caps["conclusion"],
            }
        )
    return rows


def storage_summary_for_collection(storage_rows: list[dict[str, Any]], collection: str) -> dict[str, Any]:
    selected = [r for r in storage_rows if r.get("collection") == collection]
    data_bytes = sum(
        int_or_none(r.get("bytes")) or 0
        for r in selected
        if DATA_CONTAINER_MARKER in str(r.get("container", ""))
    )
    total_bytes = sum(int_or_none(r.get("bytes")) or 0 for r in selected)
    source = "docker_exec_du" if selected else ""
    return {
        "storage_data_shards_bytes": data_bytes if selected else "",
        "storage_total_with_controller_bytes": total_bytes if selected else "",
        "storage_data_shards_gib": (data_bytes / (1024**3)) if selected else "",
        "storage_source": source,
    }


def qdrant_collection_info(base_url: str, collection: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/collections/{collection}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}
    return payload.get("result") or {}


def docker_container_names() -> list[str]:
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [
        name
        for name in names
        if "qdrant-controller-qdrant_shard_" in name or "qdrant-controller-qdrant_controller" in name
    ]


def collect_docker_storage_rows(containers: list[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for container in containers or docker_container_names():
        command = "du -sb /qdrant/storage/collections/* 2>/dev/null || true"
        try:
            result = subprocess.run(
                ["docker", "exec", container, "sh", "-lc", command],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        for line in result.stdout.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            byte_count, path = parts
            rows.append(
                {
                    "container": container,
                    "collection": Path(path).name,
                    "bytes": int_or_none(byte_count) or 0,
                    "path": path,
                    "container_role": "data_shard"
                    if DATA_CONTAINER_MARKER in container
                    else ("controller" if CONTROLLER_CONTAINER_MARKER in container else "unknown"),
                }
            )
    return rows


def build_cost_row(
    *,
    label: str,
    family: str,
    strategy: str,
    collection: str,
    build: dict[str, Any],
    live_info: dict[str, Any],
    storage: dict[str, Any],
    performance: dict[str, Any],
    build_source: str,
    performance_source: str,
) -> dict[str, Any]:
    logical = build.get("logical_points_count") or ""
    assigned = build.get("assigned_points_count") or ""
    expansion = build.get("index_expansion_ratio")
    return {
        "label": label,
        "family": family,
        "strategy": strategy,
        "collection": collection,
        "live_status": live_info.get("status", ""),
        "logical_points_count": logical,
        "assigned_points_count": assigned,
        "indexed_vectors_count": build.get("indexed_vectors_count", ""),
        "live_points_count": live_info.get("points_count", ""),
        "live_indexed_vectors_count": live_info.get("indexed_vectors_count", ""),
        "index_expansion_ratio": expansion if expansion is not None else "",
        "segments_count": build.get("segments_count", ""),
        "cluster_shard_count": build.get("cluster_shard_count", ""),
        "cluster_peer_count": build.get("cluster_peer_count", ""),
        "cluster_active_shards": build.get("cluster_active_shards", ""),
        "shard_placement": build.get("shard_placement", ""),
        "storage_data_shards_bytes": storage.get("storage_data_shards_bytes", ""),
        "storage_data_shards_gib": storage.get("storage_data_shards_gib", ""),
        "storage_total_with_controller_bytes": storage.get("storage_total_with_controller_bytes", ""),
        "storage_source": storage.get("storage_source", ""),
        "build_time_sec": "",
        "memory_rss_bytes": "",
        "request_bytes_per_query": "",
        "controller_cpu": "",
        "recall_at_10": float_or_none(performance.get("recall_at_k")) or "",
        "qps": float_or_none(performance.get("qps")) or "",
        "avg_visited_shards": float_or_none(performance.get("avg_visited_shards")) or "",
        "avg_search_requests_per_query": float_or_none(performance.get("avg_search_requests_per_query")) or "",
        "avg_candidate_groups_per_query": float_or_none(performance.get("avg_candidate_groups_per_query")) or "",
        "avg_returned_candidates_per_query": float_or_none(performance.get("avg_returned_candidates_per_query")) or "",
        "missing_cost_metrics": MISSING_COST_METRICS,
        "build_source": build_source,
        "performance_source": performance_source,
    }


def fmt_csv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: fmt_csv(row.get(name, "")) for name in fieldnames})


def build_rows_for_cases(base_url: str, storage_rows: list[dict[str, Any]], cases: list[dict[str, str]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for case in cases:
        build_path = Path(case["build_path"])
        perf_path = Path(case["performance_path"])
        build = normalize_build_row(read_first_csv_row(build_path))
        performance = read_first_csv_row(perf_path)
        collection = case["collection"]
        live_info = qdrant_collection_info(base_url, collection)
        storage = storage_summary_for_collection(storage_rows, collection)
        output_rows.append(
            build_cost_row(
                label=case["label"],
                family=case["family"],
                strategy=case["strategy"],
                collection=collection,
                build=build,
                live_info=live_info,
                storage=storage,
                performance=performance,
                build_source=str(build_path),
                performance_source=str(perf_path),
            )
        )
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:6833")
    parser.add_argument("--output-root", default="results/method4_claim_cost_analysis_20260704")
    parser.add_argument("--skip-docker-storage", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_dir = Path(args.output_root) / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    storage_rows = [] if args.skip_docker_storage else collect_docker_storage_rows()
    cost_rows = build_rows_for_cases(args.base_url, storage_rows, DEFAULT_CASES)
    build_artifact_audit_rows = collect_build_stage_artifact_audit(Path("results"))

    cost_fields = [
        "label",
        "family",
        "strategy",
        "collection",
        "live_status",
        "logical_points_count",
        "assigned_points_count",
        "indexed_vectors_count",
        "live_points_count",
        "live_indexed_vectors_count",
        "index_expansion_ratio",
        "segments_count",
        "cluster_shard_count",
        "cluster_peer_count",
        "cluster_active_shards",
        "shard_placement",
        "storage_data_shards_bytes",
        "storage_data_shards_gib",
        "storage_total_with_controller_bytes",
        "storage_source",
        "build_time_sec",
        "memory_rss_bytes",
        "request_bytes_per_query",
        "controller_cpu",
        "recall_at_10",
        "qps",
        "avg_visited_shards",
        "avg_search_requests_per_query",
        "avg_candidate_groups_per_query",
        "avg_returned_candidates_per_query",
        "missing_cost_metrics",
        "build_source",
        "performance_source",
    ]
    storage_fields = ["container", "container_role", "collection", "bytes", "path"]
    build_artifact_audit_fields = [
        "artifact_family",
        "file_count",
        "row_count",
        "example_path",
        "schema_columns",
        "has_build_wall_time",
        "has_build_rss",
        "method4_cost_applicability",
        "conclusion",
    ]

    write_csv(output_dir / "claim_cost_main_collections.csv", cost_rows, cost_fields)
    write_csv(output_dir / "claim_cost_storage_du.csv", storage_rows, storage_fields)
    write_csv(output_dir / "claim_cost_build_stage_artifact_audit.csv", build_artifact_audit_rows, build_artifact_audit_fields)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "cases": DEFAULT_CASES,
        "storage_rows": len(storage_rows),
        "build_artifact_audit_rows": len(build_artifact_audit_rows),
        "notes": [
            "Storage bytes are read-only docker exec du -sb sums over /qdrant/storage/collections/<collection>.",
            "storage_data_shards_bytes sums qdrant_shard containers only; controller metadata is reported separately.",
            f"Missing cost metrics are explicit: {MISSING_COST_METRICS}.",
            "Build-stage artifact audit scans existing builds.csv files and records whether their schemas contain build duration or RSS columns.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
