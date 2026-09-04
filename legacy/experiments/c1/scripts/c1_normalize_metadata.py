#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from c1_completion_audit import RUN_METADATA_FIELDS
from c1_protocol import repository_commit


DATASETS = ("sift1m", "glove-200-angular")
METHODS = ("random", "kmeans")
PHYSICAL_MACHINE_COUNTS = (1, 2, 4)
LOGICAL_SHARD_COUNTS = (1, 2, 4, 8, 16, 32)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def partition_seed(path: str | Path) -> int:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    return int(metadata["partition_seed"])


def fmean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return statistics.fmean(materialized)


def e1_metadata_rows(root: Path, repo: Path) -> list[dict[str, Any]]:
    rows = load_csv(root / "runs" / "c1_physical_scaleout.csv")
    expected = {
        (dataset, method, physical)
        for dataset in DATASETS
        for method in METHODS
        for physical in PHYSICAL_MACHINE_COUNTS
    }
    actual = {
        (
            row["dataset"],
            row["partition_method"],
            int(row["physical_machine_count"]),
        )
        for row in rows
    }
    if actual != expected:
        raise ValueError(f"E1 metadata coverage mismatch: {actual ^ expected}")

    dataset_records = {
        dataset: load_json(root / "runs" / f"{dataset}.dataset.json")
        for dataset in DATASETS
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        source = load_json(Path(row["source_result"]).resolve())
        repetitions = list(source["repetitions"])
        if not repetitions:
            raise ValueError(f"E1 source has no repetitions: {row['source_result']}")
        dataset = row["dataset"]
        dataset_record = dataset_records[dataset]
        shapes = dataset_record["hdf5_shapes"]
        mapping = dict(source["logical_to_physical_mapping"])
        physical_hosts = sorted(set(mapping.values()))
        query_count = int(repetitions[0]["measurement_query_count"])
        warmup_count = int(repetitions[0]["warmup_query_count"])
        if any(
            int(rep["measurement_query_count"]) != query_count
            or int(rep["warmup_query_count"]) != warmup_count
            for rep in repetitions
        ):
            raise ValueError(f"E1 repetition query counts differ: {row['source_result']}")
        output.append(
            {
                "record_type": "normalized_e1_physical_configuration",
                "experiment_id": source["experiment_id"],
                "timestamp": source["timestamp"],
                "git_commit": repository_commit(repo),
                "dataset": dataset,
                "dataset_checksum": source["dataset_sha256"],
                "partition_method": row["partition_method"],
                "partition_seed": partition_seed(row["partition_artifact"]),
                "graph_build_seed": "not_applicable_e1_uses_independent_repetitions",
                "physical_machine_count": int(row["physical_machine_count"]),
                "logical_shard_count": int(row["logical_shard_count"]),
                "logical_to_physical_mapping": json.dumps(
                    mapping, sort_keys=True, separators=(",", ":")
                ),
                "vector_count": int(shapes["train"][0]),
                "dimension": int(shapes["train"][1]),
                "distance_metric": dataset_record["dataset"]["qdrant_distance"],
                "k": 10,
                "target_recall": float(source["target_recall"]),
                "achieved_recall": float(row["achieved_recall_mean"]),
                "routing_fanout": int(row["routing_fanout"]),
                "efSearch": int(row["ef_search"]),
                "HNSW_M": 32,
                "HNSW_efConstruction": 200,
                "query_count": query_count,
                "warmup_query_count": warmup_count,
                "CPU_affinity": json.dumps(
                    {
                        "aggregator": source["benchmark_cpu_affinity"],
                        "workers": {
                            host: source["worker_cpu_affinity"][host]
                            for host in physical_hosts
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "worker_threads": json.dumps(
                    {host: 16 for host in physical_hosts},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "aggregator_threads": int(source["client_processes"]),
                "machine_hostname": json.dumps(physical_hosts, separators=(",", ":")),
                "index_memory_bytes": (
                    "not_applicable_legacy_e1_collection_deleted_before_"
                    "section32_normalization"
                ),
                "qps": float(row["qps_mean"]),
                "mean_latency": float(row["mean_latency_us"]),
                "p50_latency": float(row["p50_latency_us"]),
                "p95_latency": float(row["p95_latency_us"]),
                "p99_latency": float(row["p99_latency_us"]),
                "mean_shards_per_query": float(row["mean_shards_per_query"]),
                "p95_shards_per_query": float(row["p95_shards_per_query"]),
                "distance_computations_per_query": float(
                    row["mean_distance_computations_per_query"]
                ),
                "nodes_visited_per_query": float(row["mean_nodes_visited_per_query"]),
                "worker_cpu_time_per_query": float(
                    row["mean_worker_cpu_time_us_per_query"]
                ),
                "routing_cpu_time_per_query": fmean(
                    float(rep["routing_cpu_seconds"]) * 1_000_000.0
                    / int(rep["measurement_query_count"])
                    for rep in repetitions
                ),
                "aggregator_cpu_utilization": float(
                    row["aggregator_cpu_utilization_pct_of_reserved_cores"]
                ),
                "worker_cpu_utilization": float(row["max_worker_cpu_utilization_pct"]),
                "network_bytes_per_query": fmean(
                    float(rep["network_bytes_per_query"]) for rep in repetitions
                ),
                "repetition_count": int(row["repetition_count"]),
                "qps_sample_std": float(row["qps_sample_std"]),
                "source_result": str(Path(row["source_result"]).resolve()),
            }
        )
    return output


def e3_metadata_rows(root: Path, source_stem: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    graph_build_seeds: set[int] = set()
    for dataset in DATASETS:
        for method in METHODS:
            for logical_shards in LOGICAL_SHARD_COUNTS:
                path = (
                    root
                    / "runs"
                    / f"{source_stem}-{dataset}-{method}-m{logical_shards}-summary.json"
                )
                summary = load_json(path)
                if summary.get("result_status") != "VALID_E3":
                    raise ValueError(f"E3 source is not valid: {path}")
                if summary.get("deterministic_graph_construction") is not True:
                    raise ValueError(f"E3 source is not deterministic: {path}")
                if (
                    int(summary.get("hnsw_max_indexing_threads") or 0) != 1
                    or int(summary.get("max_optimization_threads") or 0) != 1
                ):
                    raise ValueError(f"E3 source is not single-thread constructed: {path}")
                graph_build_seeds.add(int(summary["graph_build_seed"]))
                normalized = {
                    field: summary[field] for field in RUN_METADATA_FIELDS
                }
                normalized.update(
                    {
                        "record_type": "normalized_e3_local_search_configuration",
                        "source_result": str(path.resolve()),
                    }
                )
                rows.append(normalized)
    if len(graph_build_seeds) != 1:
        raise ValueError(f"E3 metadata mixes graph-build seeds: {graph_build_seeds}")
    return rows


def validate(rows: Sequence[dict[str, Any]]) -> None:
    if len(rows) != 36:
        raise ValueError(f"expected 36 E1/E3 metadata rows, got {len(rows)}")
    for index, row in enumerate(rows, start=1):
        missing = [
            field
            for field in RUN_METADATA_FIELDS
            if field not in row or row[field] is None or row[field] == ""
        ]
        if missing:
            raise ValueError(f"metadata row {index} missing fields: {missing}")


def write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    rows = e1_metadata_rows(root, repo)
    rows.extend(e3_metadata_rows(root, args.e3_source_stem))
    validate(rows)
    output = Path(args.output).expanduser().resolve()
    write_jsonl_atomic(output, rows)
    result = {
        "record_type": "c1_normalized_run_metadata",
        "output": str(output),
        "row_count": len(rows),
        "e1_row_count": 12,
        "e3_row_count": 24,
        "field_count": len(RUN_METADATA_FIELDS),
        "status": "COMPLETE",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize C1 Section 32 run metadata")
    parser.add_argument("--repo", default="/users/dry/Orion")
    parser.add_argument("--root", default="experiments/c1")
    parser.add_argument("--e3-source-stem", default="stage12-e3-deterministic")
    parser.add_argument(
        "--output", default="experiments/c1/runs/c1_run_metadata.jsonl"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    execute(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
