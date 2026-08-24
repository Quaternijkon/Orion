#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DATASETS = ("sift1m", "glove-200-angular")
METHODS = ("random", "kmeans")
LOGICAL_SHARDS = (1, 2, 4, 8, 16, 32)
PHYSICAL_WORKERS = (1, 2, 4)
TARGET_RECALL = 0.90
PHYSICAL_HOSTS = ("10.10.1.1", "10.10.1.2", "10.10.1.3", "10.10.1.4")
BENCHMARK_CPU_AFFINITY = tuple(range(20, 32))
WORKER_CPU_AFFINITY = "0-19"

DISTRIBUTED_QUERY_FIELDS = {
    "query_id",
    "dataset",
    "partition_method",
    "logical_shards",
    "physical_hosts",
    "target_recall",
    "achieved_recall",
    "queried_shards",
    "queried_shard_ids",
    "routing_latency_us",
    "routing_cpu_time_us",
    "local_search_latency_us_per_shard",
    "distance_computations_per_shard",
    "nodes_visited_per_shard",
    "worker_cpu_time_us_per_shard",
    "response_bytes_per_shard",
    "end_to_end_latency_us",
}

E3_EVENT_FIELDS = {
    "query_id",
    "dataset",
    "partition_method",
    "logical_shards",
    "selected_fanout",
    "ef_search",
    "shard_id",
    "route_rank",
    "physical_host",
    "shard_point_count",
    "ground_truth_points_in_shard",
    "ground_truth_hits",
    "distance_computations",
    "nodes_visited",
    "worker_cpu_time_us",
    "local_search_latency_us",
    "response_bytes",
}

RUN_METADATA_FIELDS = (
    "experiment_id",
    "timestamp",
    "git_commit",
    "dataset",
    "dataset_checksum",
    "partition_method",
    "partition_seed",
    "graph_build_seed",
    "physical_machine_count",
    "logical_shard_count",
    "logical_to_physical_mapping",
    "vector_count",
    "dimension",
    "distance_metric",
    "k",
    "target_recall",
    "achieved_recall",
    "routing_fanout",
    "efSearch",
    "HNSW_M",
    "HNSW_efConstruction",
    "query_count",
    "warmup_query_count",
    "CPU_affinity",
    "worker_threads",
    "aggregator_threads",
    "machine_hostname",
    "index_memory_bytes",
    "qps",
    "mean_latency",
    "p50_latency",
    "p95_latency",
    "p99_latency",
    "mean_shards_per_query",
    "p95_shards_per_query",
    "distance_computations_per_query",
    "nodes_visited_per_query",
    "worker_cpu_time_per_query",
    "routing_cpu_time_per_query",
    "aggregator_cpu_utilization",
    "worker_cpu_utilization",
    "network_bytes_per_query",
)

REQUIRED_FINAL_FIGURES = (
    "c1_fig1_physical_scaleout.pdf",
    "c1_fig2_fanout_vs_logical_shards.pdf",
    "c1_fig3_required_fanout_cdf.pdf",
    "c1_fig4_local_search_work.pdf",
    "c1_fig5_aggregate_work_decomposition.pdf",
    "c1_fig6_projected_logical_scaling.pdf",
    "c1_combined_motivation.pdf",
)

REQUIRED_PROTOCOL_FIGURES = (
    "fig_c1_physical_scaleout.pdf",
    "fig_c1_scaling_efficiency.pdf",
    "fig_c1_fanout_vs_shards.pdf",
    "fig_c1_oracle_fanout_cdf.pdf",
    "fig_c1_actual_fanout_cdf.pdf",
    "fig_c1_local_distance_computations.pdf",
    "fig_c1_local_nodes_visited.pdf",
    "fig_c1_local_cpu_time.pdf",
    "fig_c1_aggregate_work.pdf",
    "fig_c1_model_vs_observed.pdf",
    "fig_c1_projected_scaling.pdf",
)

REQUIRED_TABLES = (
    "c1_physical_scaleout.csv",
    "c1_fanout_summary.csv",
    "c1_local_work_summary.csv",
    "c1_aggregate_work_summary.csv",
    "c1_model_accuracy.csv",
    "c1_physical_scale_table.tex",
)


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    section: str
    description: str
    status: str
    detail: str
    evidence: tuple[str, ...]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        records.append(payload)
    return records


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def csv_header_and_count(path: Path) -> tuple[set[str], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV is empty: {path}") from error
        return set(header), sum(1 for _ in reader)


def require_recorded_sha256(path_value: str, expected_sha256: str) -> Path:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_path(path)
    if actual != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected={expected_sha256}; actual={actual}"
        )
    return path


def json_field(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def expected_round_robin_mapping(logical_shards: int) -> dict[str, str]:
    host_count = min(logical_shards, len(PHYSICAL_HOSTS))
    return {
        str(shard): PHYSICAL_HOSTS[shard % host_count]
        for shard in range(logical_shards)
    }


def candidate_recall(row: dict[str, Any]) -> float:
    return float(row.get("recall_at_10", row.get("achieved_recall")))


def select_protocol_tuning_candidate(
    candidates: Sequence[dict[str, Any]], method: str
) -> dict[str, Any]:
    feasible = [row for row in candidates if candidate_recall(row) >= TARGET_RECALL]
    if not feasible:
        raise ValueError(f"no feasible tuning candidate for {method}")
    if method == "random":
        return min(
            feasible,
            key=lambda row: (int(row["ef_search"]), -candidate_recall(row)),
        )
    if method != "kmeans":
        raise ValueError(f"unknown partition method: {method}")
    best_work = min(
        float(row["mean_aggregate_distance_computations"]) for row in feasible
    )
    near_best = [
        row
        for row in feasible
        if float(row["mean_aggregate_distance_computations"]) <= best_work * 1.02
    ]
    return min(
        near_best,
        key=lambda row: (
            int(row["fanout"]),
            float(row["mean_aggregate_distance_computations"]),
            int(row["ef_search"]),
        ),
    )


def tuning_signature(row: dict[str, Any]) -> tuple[int, int, float, float]:
    return (
        int(row["fanout"]),
        int(row["ef_search"]),
        float(row["mean_aggregate_distance_computations"]),
        candidate_recall(row),
    )


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def pdf_text_and_pages(path: Path) -> tuple[str, int]:
    import pymupdf

    document = pymupdf.open(path)
    return "\n".join(page.get_text() for page in document), len(document)


def relative_evidence(root: Path, paths: Iterable[Path]) -> tuple[str, ...]:
    values: list[str] = []
    for path in paths:
        resolved = path.resolve()
        try:
            rendered = str(resolved.relative_to(root.resolve()))
        except ValueError:
            rendered = str(resolved)
        if path.is_file():
            rendered += f" sha256={sha256_path(path)}"
        values.append(rendered)
    return tuple(values)


class Audit:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.requirements: list[Requirement] = []

    def add(
        self,
        requirement_id: str,
        section: str,
        description: str,
        passed: bool,
        detail: str,
        evidence: Iterable[Path] = (),
    ) -> None:
        self.requirements.append(
            Requirement(
                requirement_id=requirement_id,
                section=section,
                description=description,
                status="PASS" if passed else "FAIL",
                detail=detail,
                evidence=relative_evidence(self.root, evidence),
            )
        )

    def guard(
        self,
        requirement_id: str,
        section: str,
        description: str,
        evidence: Iterable[Path],
        check: Callable[[], tuple[bool, str]],
    ) -> None:
        try:
            passed, detail = check()
        except Exception as error:
            passed, detail = False, f"{type(error).__name__}: {error}"
        self.add(requirement_id, section, description, passed, detail, evidence)


def configuration_keys(
    rows: Iterable[dict[str, Any]], count_field: str
) -> set[tuple[str, str, int]]:
    return {
        (str(row["dataset"]), str(row["partition_method"]), int(row[count_field]))
        for row in rows
    }


def expected_keys(counts: Sequence[int]) -> set[tuple[str, str, int]]:
    return {
        (dataset, method, count)
        for dataset in DATASETS
        for method in METHODS
        for count in counts
    }


def evaluate_local_work_evidence(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    index = {
        (str(row["dataset"]), str(row["partition_method"]), int(row["logical_shard_count"])): row
        for row in rows
    }
    expected = expected_keys(LOGICAL_SHARDS)
    if set(index) != expected:
        missing = sorted(expected - set(index))
        extra = sorted(set(index) - expected)
        raise ValueError(f"E3 configuration mismatch: missing={missing}; extra={extra}")

    comparisons = 0
    slower_than_ideal = 0
    at_or_below_ideal = 0
    for dataset in DATASETS:
        for method in METHODS:
            baseline = index[(dataset, method, 1)]
            baseline_distance = float(
                baseline["mean_distance_computations_per_searched_shard"]
            )
            baseline_nodes = float(
                baseline["mean_nodes_visited_per_searched_shard"]
            )
            if not all(
                math.isfinite(value) and value > 0
                for value in (baseline_distance, baseline_nodes)
            ):
                raise ValueError(f"invalid E3 baseline for {dataset}/{method}")
            for shards in LOGICAL_SHARDS:
                row = index[(dataset, method, shards)]
                if row["status"] != "VALID_E3":
                    raise ValueError(
                        f"invalid E3 status for {dataset}/{method}/M={shards}: "
                        f"{row['status']}"
                    )
                values = (
                    float(row["mean_distance_computations_per_searched_shard"]),
                    float(row["mean_nodes_visited_per_searched_shard"]),
                )
                if not all(math.isfinite(value) and value > 0 for value in values):
                    raise ValueError(
                        f"non-positive or non-finite E3 work for "
                        f"{dataset}/{method}/M={shards}"
                    )
                if shards == 1:
                    continue
                ideal = 1.0 / shards
                for value, baseline_value in zip(
                    values, (baseline_distance, baseline_nodes), strict=True
                ):
                    comparisons += 1
                    if value / baseline_value > ideal:
                        slower_than_ideal += 1
                    else:
                        at_or_below_ideal += 1

    return {
        "configuration_count": len(index),
        "comparison_count": comparisons,
        "slower_than_ideal_count": slower_than_ideal,
        "at_or_below_ideal_count": at_or_below_ideal,
        "slow_local_work_supported": at_or_below_ideal == 0,
        "c1_c_status": "SUPPORTED" if at_or_below_ideal == 0 else "CONTRADICTED",
    }


def evaluate_sensitivity_evidence(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    targets = (0.89, 0.90, 0.91)
    index = {
        (
            str(row["dataset"]),
            str(row["partition_method"]),
            int(row["logical_shard_count"]),
            round(float(row["sensitivity_target_recall"]), 2),
        ): row
        for row in rows
    }
    expected = {
        (dataset, method, shards, target)
        for dataset in DATASETS
        for method in METHODS
        for shards in (4, 16)
        for target in targets
    }
    if set(index) != expected:
        missing = sorted(expected - set(index))
        extra = sorted(set(index) - expected)
        raise ValueError(
            f"E5 configuration mismatch: missing={missing}; extra={extra}"
        )
    contradicted_checks: list[str] = []
    for dataset in DATASETS:
        for method in METHODS:
            for target in targets:
                lower = index[(dataset, method, 4, target)]
                upper = index[(dataset, method, 16, target)]
                if (
                    float(lower["achieved_tuning_recall"]) < target
                    or float(upper["achieved_tuning_recall"]) < target
                ):
                    raise ValueError(
                        f"E5 recall target miss for {dataset}/{method}/{target:.2f}"
                    )
                local_ratio = (
                    float(upper["mean_local_distance_computations"])
                    / float(lower["mean_local_distance_computations"])
                )
                preserved = (
                    local_ratio > 0.25
                    and int(upper["selected_fanout"])
                    >= int(lower["selected_fanout"])
                    and float(upper["mean_aggregate_distance_computations"])
                    >= float(lower["mean_aggregate_distance_computations"])
                )
                if not preserved:
                    contradicted_checks.append(f"{dataset}/{method}/{target:.2f}")
    return {
        "configuration_count": len(index),
        "all_trends_preserved": not contradicted_checks,
        "contradicted_checks": contradicted_checks,
        "status": (
            "OBSERVED_TREND_STABLE"
            if not contradicted_checks
            else "CONTRADICTED_QUALITATIVE_TREND_REVERSAL"
        ),
    }


def audit_protocol(root: Path) -> Audit:
    audit = Audit(root)
    runs = root / "runs"
    figures = root / "figures"
    plan = root / "PLAN.md"
    authoritative_plan = root.parents[1] / "plan" / "C1 Experimental Protocol_ Scale-Out Bottleneck in Scatter-Gather Graph ANN Search.md"
    status = root / "STATUS.md"
    results = root / "RESULTS.md"
    manifest = runs / "manifest.jsonl"
    physical_path = runs / "c1_physical_scaleout.csv"
    fanout_path = runs / "c1_fanout_summary.csv"
    local_path = runs / "c1_local_work_summary.csv"
    aggregate_path = runs / "c1_aggregate_work_summary.csv"
    model_path = runs / "c1_model_accuracy.csv"
    sensitivity_path = runs / "c1_recall_sensitivity.csv"
    final_record_path = runs / "stage12-c1-deterministic-final-record.json"
    final_summary_path = runs / "stage12-c1-deterministic-final-summary.json"
    section31_path = runs / "section31-deterministic-build-proof.json"
    metadata_path = runs / "c1_run_metadata.jsonl"
    cleanup_path = runs / "c1_final_cleanup-proof.json"

    audit.guard(
        "S01-objective-scope",
        "1-2",
        "Objective, causal chain, and physical-versus-logical scope are preserved.",
        (final_record_path,),
        lambda: (
            (lambda record: record.get("result_scope") == {
                "m_1_2_4": "measured_physical_scaleout",
                "m_8_16_32": "logical_shard_mechanism_measurements_only",
                "m_gt_4_throughput_projection": "withheld",
            })(load_json(final_record_path)),
            "Final record distinguishes measured physical scale-out, logical-shard mechanism measurements, and withheld projections.",
        ),
    )
    audit.guard(
        "S03-datasets-and-split",
        "3",
        "Exactly SIFT1M and glove-200-angular use disjoint 1,000-query tuning and 9,000-query measurement sets.",
        (runs / "sift1m.dataset.json", runs / "glove-200-angular.dataset.json"),
        lambda: (
            all(
                (lambda data: data["tuning_query_count"] == 1000
                 and data["measurement_query_count"] == 9000
                 and data["query_sets_disjoint"] is True
                 and data["measurement_query_range"] == [1000, 10000])(
                    load_json(runs / f"{dataset}.dataset.json")
                )
                for dataset in DATASETS
            ),
            "Both audited dataset manifests retain exact checksums and disjoint official-query splits.",
        ),
    )
    def check_baselines() -> tuple[bool, str]:
        rows = load_csv(fanout_path)
        metadata = load_jsonl(metadata_path)
        valid = configuration_keys(rows, "logical_shard_count") == expected_keys(
            LOGICAL_SHARDS
        )
        valid &= all(row["partition_method"] in METHODS for row in rows)
        valid &= all(
            int(float(row["actual_fanout_mean"]))
            == int(row["logical_shard_count"])
            for row in rows
            if row["partition_method"] == "random"
        )
        valid &= all(record["partition_method"] in METHODS for record in metadata)
        valid &= all(int(record["partition_seed"]) == 20260820 for record in metadata)
        valid &= all(
            not any("orion" in str(key).lower() for key in record)
            for record in metadata
        )
        e3_metadata = [
            record
            for record in metadata
            if record["record_type"] == "normalized_e3_local_search_configuration"
        ]
        valid &= all(int(record["graph_build_seed"]) == 20260821 for record in e3_metadata)
        return (
            bool(valid),
            "The complete matrix uses only seeded Random/K-Means layouts, every Random serving row broadcasts, and normalized authoritative records contain no Orion-specific fields.",
        )

    audit.guard(
        "S04-baselines",
        "4",
        "Only deterministic Random and centroid-ranked K-Means baselines are used; Random broadcasts.",
        (fanout_path, metadata_path),
        check_baselines,
    )

    def check_configurations_and_placement() -> tuple[bool, str]:
        physical_rows = load_csv(physical_path)
        local_rows = load_csv(local_path)
        valid = configuration_keys(
            physical_rows, "physical_machine_count"
        ) == expected_keys(PHYSICAL_WORKERS)
        valid &= configuration_keys(local_rows, "logical_shard_count") == expected_keys(
            LOGICAL_SHARDS
        )
        for row in physical_rows:
            machines = int(row["physical_machine_count"])
            source = require_recorded_sha256(
                row["source_result"], row["source_result_sha256"]
            )
            require_recorded_sha256(
                row["partition_artifact"], row["partition_artifact_sha256"]
            )
            record = load_json(source)
            expected_mapping = expected_round_robin_mapping(machines)
            valid &= record["logical_to_physical_mapping"] == expected_mapping
            valid &= int(record["physical_machine_count"]) == machines
            valid &= tuple(record["benchmark_cpu_affinity"]) == BENCHMARK_CPU_AFFINITY
            worker_affinity = record["worker_cpu_affinity"]
            valid &= set(expected_mapping.values()).issubset(worker_affinity)
            valid &= all(value == WORKER_CPU_AFFINITY for value in worker_affinity.values())
        for row in local_rows:
            shards = int(row["logical_shard_count"])
            source = require_recorded_sha256(
                row["source_summary"], row["source_summary_sha256"]
            )
            record = load_json(source)
            expected_mapping = expected_round_robin_mapping(shards)
            valid &= record["logical_to_physical_mapping"] == expected_mapping
            valid &= int(record["physical_machine_count"]) == min(shards, 4)
            valid &= tuple(record["benchmark_cpu_affinity"]) == BENCHMARK_CPU_AFFINITY
            valid &= tuple(record["CPU_affinity"]["aggregator"]) == BENCHMARK_CPU_AFFINITY
            valid &= set(expected_mapping.values()).issubset(
                record["CPU_affinity"]["workers"]
            )
            valid &= all(
                value == WORKER_CPU_AFFINITY
                for value in record["CPU_affinity"]["workers"].values()
            )
            valid &= int(record["competing_logical_shards"]) == 0
            valid &= int(record["query_concurrency_per_shard"]) == 1
            valid &= len(record["shards"]) == shards
            valid &= len(record["shard_resources"]) == shards
        return (
            bool(valid),
            "E1 has 12 physical rows and E3 has 24 logical rows; every source hash, distinct physical placement, deterministic round-robin mapping, fixed affinity, and isolated shard count validates.",
        )

    audit.guard(
        "S05-configurations-placement",
        "5",
        "Physical M=1,2,4 and logical M=1,2,4,8,16,32 matrices are complete with deterministic round-robin placement.",
        (physical_path, local_path),
        check_configurations_and_placement,
    )
    audit.add(
        "S06-continuous-recording",
        "6",
        "PLAN, STATUS, RESULTS, runs, figures, scripts, and logs are maintained without overwriting corrections.",
        all(path.exists() for path in (plan, status, results, runs, figures, root / "scripts", root / "logs"))
        and plan.read_text(encoding="utf-8").rstrip("\n")
        == authoritative_plan.read_text(encoding="utf-8").rstrip("\n")
        and "Stage 10 completion-audit correction" in results.read_text(encoding="utf-8"),
        "Required directory contract exists; PLAN matches the authoritative protocol and RESULTS retains the superseding correction.",
        (plan, status, results),
    )
    def check_instrumentation() -> tuple[bool, str]:
        physical_rows = load_csv(physical_path)
        aggregate_rows = load_csv(aggregate_path)
        valid = True
        distributed_files = 0
        distributed_rows = 0
        summary_fields = {
            "mean_shards_per_query",
            "median_shards_per_query",
            "p95_shards_per_query",
            "mean_distance_computations_per_query",
            "mean_nodes_visited_per_query",
            "mean_worker_cpu_time_us_per_query",
            "mean_routing_latency_us",
            "mean_latency_us",
            "p50_latency_us",
            "p95_latency_us",
            "p99_latency_us",
            "completed_qps",
            "aggregator_cpu_utilization_pct_of_reserved_cores",
            "resource_utilization",
            "network_bytes_per_query",
        }
        seen_sources: set[Path] = set()
        for row in physical_rows:
            source = require_recorded_sha256(
                row["source_result"], row["source_result_sha256"]
            )
            if source in seen_sources:
                continue
            seen_sources.add(source)
            record = load_json(source)
            for repetition in record["repetitions"]:
                valid &= summary_fields.issubset(repetition)
                raw = require_recorded_sha256(
                    repetition["raw_per_query"], repetition["raw_per_query_sha256"]
                )
                headers, row_count = csv_header_and_count(raw)
                valid &= DISTRIBUTED_QUERY_FIELDS.issubset(headers)
                valid &= row_count == int(repetition["measurement_query_count"])
                distributed_files += 1
                distributed_rows += row_count

        e3_files = 0
        e3_rows = 0
        seen_e3: set[Path] = set()
        for row in aggregate_rows:
            source = require_recorded_sha256(
                row["source_e3_per_search"], row["source_e3_per_search_sha256"]
            )
            if source in seen_e3:
                continue
            seen_e3.add(source)
            headers, row_count = csv_header_and_count(source)
            valid &= E3_EVENT_FIELDS.issubset(headers)
            valid &= row_count == int(row["query_count"]) * int(
                float(row["actual_fanout_mean"])
            )
            e3_files += 1
            e3_rows += row_count
        return (
            bool(valid),
            f"Validated all mandatory fields and recorded hashes for {distributed_files} E1 raw query files ({distributed_rows} rows) and {e3_files} isolated E3 event files ({e3_rows} rows).",
        )

    audit.guard(
        "S07-instrumentation",
        "7",
        "Distributed per-query and isolated per-shard evidence contains routing, work, CPU, bytes, and latency counters.",
        (physical_path, aggregate_path),
        check_instrumentation,
    )
    def check_tuning() -> tuple[bool, str]:
        physical_rows = load_csv(physical_path)
        local_rows = load_csv(local_path)
        physical_index = {
            (row["dataset"], row["partition_method"], int(row["physical_machine_count"])): row
            for row in physical_rows
        }
        local_index = {
            (row["dataset"], row["partition_method"], int(row["logical_shard_count"])): row
            for row in local_rows
        }
        manifest_candidates = {
            canonical_json(record)
            for record in load_jsonl(manifest)
            if record.get("record_type") == "tuning_candidate"
        }
        valid = all(
            float(row["achieved_recall_min"]) >= TARGET_RECALL
            for row in physical_rows
        ) and all(
            float(row["selected_tuning_recall"]) >= TARGET_RECALL
            for row in local_rows
        )
        artifact_count = 0
        candidate_count = 0
        for stem, shard_counts, index in (
            ("stage1", PHYSICAL_WORKERS, physical_index),
            ("stage12-e3-deterministic", LOGICAL_SHARDS, local_index),
        ):
            for dataset in DATASETS:
                for method in METHODS:
                    for shards in shard_counts:
                        path = runs / f"{stem}-{dataset}-{method}-m{shards}-tuning-pinned.json"
                        artifact = load_json(path)
                        candidates = artifact["candidates"]
                        selected = artifact["selected"]
                        expected = select_protocol_tuning_candidate(candidates, method)
                        valid &= tuning_signature(selected) == tuning_signature(expected)
                        valid &= all(int(row["query_count"]) == 1000 for row in candidates)
                        valid &= all(
                            (candidate_recall(row) >= TARGET_RECALL)
                            == (row["status"] == "VALID")
                            for row in candidates
                        )
                        valid &= all(
                            canonical_json(row) in manifest_candidates for row in candidates
                        )
                        table_row = index[(dataset, method, shards)]
                        valid &= int(table_row["ef_search"]) == int(selected["ef_search"])
                        valid &= int(table_row["routing_fanout"]) == int(selected["fanout"])
                        artifact_count += 1
                        candidate_count += len(candidates)
        return (
            bool(valid),
            f"Replayed the Random/K-Means tuning rule for {artifact_count} artifacts and verified all {candidate_count} candidates are retained verbatim in the manifest; every selected measurement point passes recall.",
        )

    audit.guard(
        "S08-tuning",
        "8",
        "All selected configurations meet Recall@10 >= 0.90 and tuning candidates are retained.",
        (manifest, physical_path, local_path),
        check_tuning,
    )

    def check_e1() -> tuple[bool, str]:
        rows = load_csv(physical_path)
        valid = configuration_keys(rows, "physical_machine_count") == expected_keys(PHYSICAL_WORKERS)
        seen_sources: set[Path] = set()
        source_count = 0
        repetition_count = 0
        for row in rows:
            repetitions = int(row["repetition_count"])
            cv = float(row["qps_coefficient_of_variation"])
            valid &= repetitions >= 3 and (cv <= 0.05 or repetitions >= 5)
            valid &= float(row["achieved_recall_min"]) >= TARGET_RECALL
            valid &= not row["bottleneck_flags"].strip()
            valid &= float(row["scaling_efficiency"]) <= 1.0 + 1e-12
            workers = int(row["physical_machine_count"])
            if workers > 1:
                valid &= float(row["normalized_qps_to_m1"]) < workers
                valid &= float(row["scaling_efficiency"]) < 1.0
            source = require_recorded_sha256(
                row["source_result"], row["source_result_sha256"]
            )
            if source in seen_sources:
                continue
            seen_sources.add(source)
            source_count += 1
            record = load_json(source)
            grid = [int(value) for value in record["concurrency_grid"]]
            valid &= bool(grid) and grid[0] == 1
            valid &= all(right == left * 2 for left, right in zip(grid, grid[1:]))
            sweep_concurrency = [int(item["concurrency"]) for item in record["sweep"]]
            valid &= sweep_concurrency == grid[: len(sweep_concurrency)]
            selection = record["saturation_selection"]
            valid &= selection.get("knee_observed") is True
            valid &= int(selection["selected_concurrency"]) == int(
                record["selected_concurrency"]
            )
            valid &= selection["stop_reason"] in {
                "p99_exceeded_multiplier",
                "qps_improved_less_than_5pct_twice",
            }
            valid &= int(record["selected_concurrency"]) in sweep_concurrency
            valid &= int(record["repetition_count"]) == len(record["repetitions"])
            for repetition in record["repetitions"]:
                repetition_count += 1
                valid &= int(repetition["warmup_query_count"]) >= 1000
                valid &= int(repetition["measurement_query_count"]) >= 10000
                valid &= int(repetition["query_count"]) == int(
                    repetition["measurement_query_count"]
                )
                valid &= float(repetition["achieved_recall"]) >= TARGET_RECALL
                valid &= float(repetition["offered_qps"]) > 0
                valid &= float(repetition["completed_qps"]) > 0
                valid &= not repetition["bottleneck_flags"]
                valid &= repetition["persistent_aggregator_queue_growth"] is False
                valid &= (
                    float(repetition["aggregator_cpu_utilization_pct_of_reserved_cores"])
                    <= 85.0
                )
                valid &= (
                    float(repetition["routing_share_of_routing_plus_worker_cpu_pct"])
                    <= 25.0
                )
                valid &= (
                    float(repetition["resource_utilization"]["max_network_utilization_pct"])
                    <= 85.0
                )
                valid &= int(repetition["aggregator_waiting_queue_depth_end"]) == 0
        return (
            bool(valid),
            f"Validated all 12 E1 rows against {source_count} unique closed-loop sweep records and {repetition_count} formal repetitions, including warmup, >=10,000 measurements, recall, CV/CI inputs, offered/completed QPS, and bottleneck gates.",
        )

    audit.guard(
        "E1-physical-scaleout",
        "9-13",
        "Closed-loop physical scale-out evidence is statistically valid and not aggregator/network limited.",
        (physical_path,),
        check_e1,
    )

    def check_e2() -> tuple[bool, str]:
        rows = load_csv(fanout_path)
        valid = configuration_keys(rows, "logical_shard_count") == expected_keys(LOGICAL_SHARDS)
        row_index = {
            (row["dataset"], row["partition_method"], int(row["logical_shard_count"])): row
            for row in rows
        }
        for row in rows:
            valid &= float(row["measurement_recall"]) >= TARGET_RECALL
            valid &= int(row["query_count"]) == 9000
            oracle_counts = {
                int(key): int(value)
                for key, value in json.loads(row["oracle_distribution_counts"]).items()
            }
            actual_counts = {
                int(key): int(value)
                for key, value in json.loads(row["actual_distribution_counts"]).items()
            }
            valid &= sum(oracle_counts.values()) == 9000
            valid &= sum(actual_counts.values()) == 9000
            valid &= all(1 <= value <= int(row["logical_shard_count"]) for value in oracle_counts)
            valid &= all(1 <= value <= int(row["logical_shard_count"]) for value in actual_counts)

        per_query_rows = 0
        heterogeneous_groups = 0
        for dataset in DATASETS:
            record_path = runs / f"stage12-e2-deterministic-{dataset}-fanout-record.json"
            record = load_json(record_path)
            per_query_path = runs / "per_query" / f"stage12-e2-deterministic-{dataset}-fanout.csv"
            valid &= sha256_path(per_query_path) == record["evidence_sha256"]["per_query"]
            grouped: dict[tuple[str, int], dict[str, Any]] = {}
            for event in load_csv(per_query_path):
                key = (event["partition_method"], int(event["logical_shard_count"]))
                group = grouped.setdefault(
                    key,
                    {
                        "count": 0,
                        "recall": 0.0,
                        "oracle": Counter(),
                        "actual": Counter(),
                    },
                )
                group["count"] += 1
                group["recall"] += float(event["achieved_recall"])
                oracle = int(event["oracle_minimum_fanout"])
                actual = int(event["actual_selected_fanout"])
                group["oracle"][oracle] += 1
                group["actual"][actual] += 1
                valid &= 1000 <= int(event["query_id"]) < 10000
                valid &= len(json.loads(event["actual_selected_shard_ids"])) == actual
                valid &= 1 <= oracle <= int(event["logical_shard_count"])
                per_query_rows += 1
            valid &= set(grouped) == {
                (method, shards) for method in METHODS for shards in LOGICAL_SHARDS
            }
            for (method, shards), group in grouped.items():
                table_row = row_index[(dataset, method, shards)]
                valid &= group["count"] == 9000
                valid &= math.isclose(
                    group["recall"] / group["count"],
                    float(table_row["measurement_recall"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                valid &= dict(sorted(group["oracle"].items())) == {
                    int(key): int(value)
                    for key, value in json.loads(
                        table_row["oracle_distribution_counts"]
                    ).items()
                }
                valid &= dict(sorted(group["actual"].items())) == {
                    int(key): int(value)
                    for key, value in json.loads(
                        table_row["actual_distribution_counts"]
                    ).items()
                }
                oracle_values = list(group["oracle"].elements())
                valid &= math.isclose(
                    statistics.fmean(oracle_values),
                    float(table_row["oracle_fanout_mean"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                if len(group["oracle"]) > 1:
                    heterogeneous_groups += 1
        valid &= per_query_rows == 216000
        valid &= heterogeneous_groups > 0
        return (
            bool(valid),
            f"Recomputed all 24 actual/oracle distributions and held-out recalls from {per_query_rows} E2 rows; {heterogeneous_groups} groups have heterogeneous oracle fan-out.",
        )

    audit.guard(
        "E2-fanout",
        "14-17",
        "Oracle and actual fan-out distributions are complete for M=1..32.",
        (fanout_path,),
        check_e2,
    )

    def check_e3() -> tuple[bool, str]:
        rows = load_csv(local_path)
        evidence = evaluate_local_work_evidence(rows)
        valid = True
        resource_rows = 0
        cache_transition_present = False
        for row in rows:
            source = require_recorded_sha256(
                row["source_summary"], row["source_summary_sha256"]
            )
            record = load_json(source)
            shards = int(row["logical_shard_count"])
            valid &= record["result_status"] == "VALID_E3"
            valid &= record["deterministic_graph_construction"] is True
            valid &= int(record["graph_build_seed"]) == 20260821
            valid &= int(record["hnsw_max_indexing_threads"]) == 1
            valid &= int(record["max_optimization_threads"]) == 1
            valid &= int(record["query_start"]) == 1000
            valid &= int(record["query_stop"]) == 10000
            valid &= int(record["query_count"]) == 9000
            valid &= int(record["searched_shard_events"]) == 9000 * int(
                record["routing_fanout"]
            )
            valid &= int(record["competing_logical_shards"]) == 0
            valid &= int(record["query_concurrency_per_shard"]) == 1
            valid &= math.isclose(
                float(record["mean_distance_computations_per_searched_shard"]),
                float(row["mean_distance_computations_per_searched_shard"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            valid &= math.isclose(
                float(record["mean_nodes_visited_per_searched_shard"]),
                float(row["mean_nodes_visited_per_searched_shard"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            resources = record["shard_resources"]
            valid &= set(resources) == {str(shard) for shard in range(shards)}
            for resource in resources.values():
                resource_rows += 1
                valid &= all(
                    int(resource[field]) > 0
                    for field in (
                        "point_count",
                        "vector_storage_size_bytes",
                        "graph_index_size_bytes",
                        "total_local_index_size_bytes",
                        "machine_llc_size_bytes",
                        "qdrant_resident_set_size_bytes",
                    )
                )
                valid &= (
                    int(resource["total_local_index_size_bytes"])
                    == int(resource["vector_storage_size_bytes"])
                    + int(resource["graph_index_size_bytes"])
                )
            cache_transition_present |= row["cache_regime"] == "mixed_or_below_llc"
        if cache_transition_present:
            results_text = results.read_text(encoding="utf-8")
            valid &= "LLC" in results_text and (
                "below" in results_text or "mixed" in results_text
            )
        return (
            bool(valid),
            "E3 covers all 24 isolated configurations and all "
            f"{resource_rows} shard-resource records with fixed affinity/concurrency, deterministic construction, positive distance/node work, and cache-regime evidence; "
            f"C1-c={evidence['c1_c_status']} from "
            f"{evidence['slower_than_ideal_count']} above-ideal and "
            f"{evidence['at_or_below_ideal_count']} at-or-below-ideal comparisons.",
        )

    audit.guard(
        "E3-local-work",
        "18-23",
        "Isolated fixed-quality local graph work, resource sizes, cache regime, and reference trends are complete, including contradictory trends.",
        (local_path,),
        check_e3,
    )

    def check_e4() -> tuple[bool, str]:
        rows = load_csv(aggregate_path)
        model_rows = load_csv(model_path)
        valid = configuration_keys(rows, "logical_shard_count") == expected_keys(LOGICAL_SHARDS)
        valid &= len(model_rows) == len(DATASETS) * len(METHODS)
        row_index = {
            (row["dataset"], row["partition_method"], int(row["logical_shard_count"])): row
            for row in rows
        }
        for row in rows:
            shards = int(row["logical_shard_count"])
            valid &= int(row["query_count"]) == 9000
            valid &= float(row["observed_distance_mean"]) > 0
            valid &= float(row["observed_distance_p95"]) > 0
            valid &= float(row["observed_nodes_mean"]) > 0
            valid &= float(row["observed_nodes_p95"]) > 0
            valid &= float(row["observed_worker_cpu_time_us_mean"]) > 0
            valid &= float(row["observed_worker_cpu_time_us_p95"]) > 0
            valid &= float(row["model_relative_error"]) < 1e-9
            valid &= math.isclose(
                float(row["modeled_distance_mean"]),
                float(row["actual_fanout_mean"])
                * float(row["mean_local_distance_per_searched_shard"]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            valid &= math.isclose(
                float(row["ideal_query_work_reduction"]),
                1.0 / shards,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            require_recorded_sha256(
                row["source_e3_per_search"], row["source_e3_per_search_sha256"]
            )
            if shards > 4:
                valid &= not row["m_gt_4_projection"].strip()
                valid &= not row["physical_e1_normalized_qps"].strip()
            else:
                valid &= bool(row["physical_e1_normalized_qps"].strip())
        for row in model_rows:
            valid &= int(row["configuration_count"]) == 6
            valid &= int(row["physical_point_count"]) == 3
            valid &= float(row["pearson_model_vs_observed"]) > 0.99
            valid &= float(row["spearman_model_vs_observed"]) > 0.99
            valid &= float(row["mean_relative_error"]) < 1e-9
            valid &= float(row["maximum_relative_error"]) < 1e-9
            valid &= row["physical_attribution_status"] == "INSUFFICIENT"
            valid &= (
                row["m_gt_4_projection_status"]
                == "WITHHELD_PHYSICAL_SANITY_CHECK_FAILED"
            )

        per_query_rows = 0
        for dataset in DATASETS:
            suffix = (
                "sift1m-final-aggregate-work"
                if dataset == "sift1m"
                else "glove-200-angular-aggregate-work"
            )
            record = load_json(runs / f"stage12-e4-deterministic-{suffix}-record.json")
            per_query_path = runs / "per_query" / f"stage12-e4-deterministic-{suffix}.csv"
            valid &= sha256_path(per_query_path) == record["evidence_sha256"]["per_query"]
            grouped: dict[tuple[str, int], dict[str, Any]] = {}
            for event in load_csv(per_query_path):
                key = (event["partition_method"], int(event["logical_shard_count"]))
                group = grouped.setdefault(
                    key,
                    {"count": 0, "distance": [], "nodes": [], "cpu": []},
                )
                group["count"] += 1
                group["distance"].append(float(event["observed_distance_computations"]))
                group["nodes"].append(float(event["observed_nodes_visited"]))
                group["cpu"].append(float(event["observed_worker_cpu_time_us"]))
                per_query_rows += 1
            valid &= set(grouped) == {
                (method, shards) for method in METHODS for shards in LOGICAL_SHARDS
            }
            for (method, shards), group in grouped.items():
                table_row = row_index[(dataset, method, shards)]
                valid &= group["count"] == 9000
                valid &= math.isclose(
                    statistics.fmean(group["distance"]),
                    float(table_row["observed_distance_mean"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                valid &= math.isclose(
                    statistics.fmean(group["nodes"]),
                    float(table_row["observed_nodes_mean"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                valid &= math.isclose(
                    statistics.fmean(group["cpu"]),
                    float(table_row["observed_worker_cpu_time_us_mean"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
        valid &= per_query_rows == 216000
        return (
            bool(valid),
            f"Recomputed all 24 observed/model means from {per_query_rows} E4 rows; distance/node/CPU means and p95 fields are present, model error is effectively zero, physical sanity statistics are complete, and M>4 projections are withheld.",
        )

    audit.guard(
        "E4-decomposition",
        "24-28",
        "Observed aggregate work, fan-out x local-work model, accuracy statistics, and projection boundaries are complete.",
        (aggregate_path, model_path),
        check_e4,
    )
    def check_e5() -> tuple[bool, str]:
        evidence = evaluate_sensitivity_evidence(load_csv(sensitivity_path))
        record = load_json(final_record_path)
        valid = record.get("e5_recall_sensitivity_status") == evidence["status"]
        if evidence["contradicted_checks"]:
            valid &= (
                "e5_recall_sensitivity_contains_qualitative_trend_reversal"
                in record.get("anomalies", [])
            )
        return (
            bool(valid),
            "E5 has 24 valid neighborhood rows and records "
            f"{evidence['status']} with contradicted checks="
            f"{evidence['contradicted_checks']}.",
        )

    audit.guard(
        "E5-sensitivity",
        "29",
        "M=4 and M=16 neighborhoods at recall 0.89, 0.90, and 0.91 are complete and any qualitative reversal is reported.",
        (sensitivity_path, final_record_path),
        check_e5,
    )
    results_text = results.read_text(encoding="utf-8")
    order_markers = (
        "Stage 0 instrumentation checkpoint",
        "Stage 1 partition and CPU-counter gate",
        "Stage 2 SIFT1M E3 M=32 logical-shard checkpoint",
        "Stage 3 SIFT1M E2 fan-out checkpoint",
        "Stage 6 corrected SIFT1M E1 physical scale-out checkpoint",
        "Stage 7 corrected SIFT1M E4 checkpoint",
        "Stage 8 GloVe E3 M=1 checkpoint",
        "Stage 8 GloVe E2 fan-out analysis",
        "Stage 8 corrected GloVe E1 physical scale-out",
        "Stage 9 GloVe E4 and cross-dataset projection gate",
        "Stage 10 completion-audit correction",
        "Stage 12 deterministic completion and strict protocol audit",
    )
    order_positions = [results_text.find(marker) for marker in order_markers]
    audit.add(
        "S30-execution-order",
        "30",
        "Stages execute in protocol order and contradictions are recorded before finalization.",
        all(position >= 0 for position in order_positions)
        and order_positions == sorted(order_positions),
        "RESULTS preserves the ordered SIFT validation, GloVe replication, contradiction correction, deterministic rerun, and final audit checkpoints without overwriting history.",
        (results,),
    )

    def check_section31() -> tuple[bool, str]:
        proof = load_json(section31_path)
        required = {
            "graph_build_seed", "image_id", "max_indexing_threads",
            "max_optimization_threads", "identical_build_count",
            "graph_content_sha256", "source_commit",
        }
        valid = required.issubset(proof)
        valid &= int(proof["max_indexing_threads"]) == 1
        valid &= int(proof["max_optimization_threads"]) == 1
        valid &= int(proof["identical_build_count"]) >= 2
        valid &= proof.get("deterministic_construction_verified") is True
        valid &= proof.get("authoritative_e2_e4_rerun_complete") is True
        evidence = proof.get("authoritative_rerun_evidence") or {}
        valid &= len(evidence.get("e2_records") or {}) == 2
        valid &= len(evidence.get("e3_configuration_summaries") or {}) == 24
        valid &= len(evidence.get("e4_records") or {}) == 3
        valid &= bool(evidence.get("normalized_metadata"))
        valid &= bool(evidence.get("final_cleanup"))
        valid &= len(proof["builds"]) == int(proof["identical_build_count"])
        valid &= all(
            build["graph_content_sha256"] == proof["graph_content_sha256"]
            and build["collection_info"]["status"] == "green"
            and int(build["collection_info"]["indexed_vectors_count"])
            == int(proof["point_count"])
            for build in proof["builds"]
        )
        graph_file_hashes = [
            {entry["filename"]: entry["sha256"] for entry in build["graph_files"]}
            for build in proof["builds"]
        ]
        valid &= all(hashes == graph_file_hashes[0] for hashes in graph_file_hashes)
        for records in (
            evidence["e2_records"],
            evidence["e3_matrix_record"],
            evidence["e4_records"],
            evidence["normalized_metadata"],
            evidence["final_cleanup"],
        ):
            valid &= all(
                sha256_path(Path(path)) == expected_sha256
                for path, expected_sha256 in records.items()
            )
        for dataset in DATASETS:
            for method in METHODS:
                for shards in LOGICAL_SHARDS:
                    key = f"{dataset}/{method}/m{shards}"
                    summary = runs / f"stage12-e3-deterministic-{dataset}-{method}-m{shards}-summary.json"
                    valid &= sha256_path(summary) == evidence[
                        "e3_configuration_summaries"
                    ][key]
        return bool(valid), "Section 31 proof verifies equal graph/file hashes for two independent green builds and revalidates every recorded E2/E3/E4/metadata/cleanup evidence hash in the authoritative rerun."

    audit.guard(
        "S31-statistical-treatment",
        "31",
        "E1 repetitions and deterministic E2-E4 construction satisfy the statistical protocol.",
        (section31_path,),
        check_section31,
    )

    def check_metadata() -> tuple[bool, str]:
        records = load_jsonl(metadata_path)
        missing: Counter[str] = Counter()
        for record in records:
            for field in RUN_METADATA_FIELDS:
                if field not in record or record[field] is None or record[field] == "":
                    missing[field] += 1
        record_types = Counter(record.get("record_type") for record in records)
        valid = len(records) == 36 and not missing
        valid &= record_types == {
            "normalized_e1_physical_configuration": 12,
            "normalized_e3_local_search_configuration": 24,
        }
        e1_records = [
            record
            for record in records
            if record["record_type"] == "normalized_e1_physical_configuration"
        ]
        e3_records = [
            record
            for record in records
            if record["record_type"] == "normalized_e3_local_search_configuration"
        ]
        valid &= configuration_keys(
            e1_records, "physical_machine_count"
        ) == expected_keys(PHYSICAL_WORKERS)
        valid &= configuration_keys(
            e3_records, "logical_shard_count"
        ) == expected_keys(LOGICAL_SHARDS)
        expected_checksums = {
            "sift1m": "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984",
            "glove-200-angular": "4839085e5a8bb293434a1a66e1aa0193afc3f07c6797a85f1dbd91656172da20",
        }
        expected_dimensions = {"sift1m": 128, "glove-200-angular": 200}
        expected_metrics = {"sift1m": "Euclid", "glove-200-angular": "Cosine"}
        for record in records:
            shards = int(record["logical_shard_count"])
            valid &= record["dataset_checksum"] == expected_checksums[record["dataset"]]
            valid &= int(record["dimension"]) == expected_dimensions[record["dataset"]]
            valid &= record["distance_metric"] == expected_metrics[record["dataset"]]
            valid &= int(record["k"]) == 10
            valid &= float(record["target_recall"]) == TARGET_RECALL
            valid &= float(record["achieved_recall"]) >= TARGET_RECALL
            valid &= int(record["physical_machine_count"]) == min(shards, 4)
            valid &= json_field(record["logical_to_physical_mapping"]) == expected_round_robin_mapping(
                shards
            )
            affinity = json_field(record["CPU_affinity"])
            valid &= tuple(affinity["aggregator"]) == BENCHMARK_CPU_AFFINITY
            valid &= all(
                value == WORKER_CPU_AFFINITY
                for value in affinity["workers"].values()
            )
            valid &= Path(record["source_result"]).is_file()
        for record in e1_records:
            valid &= int(record["query_count"]) >= 10000
            valid &= int(record["warmup_query_count"]) >= 1000
            valid &= int(record["repetition_count"]) >= 3
            valid &= float(record["qps"]) > 0
            valid &= int(record["aggregator_threads"]) == 128
        for record in e3_records:
            valid &= int(record["query_count"]) == 9000
            valid &= int(record["warmup_query_count"]) == 0
            valid &= int(record["graph_build_seed"]) == 20260821
            valid &= record["qps"] == "not_applicable_isolated_e3"
            valid &= record["aggregator_threads"] == "not_applicable_isolated_e3"
        return (
            bool(valid),
            f"Normalized metadata rows={len(records)}; "
            f"types={dict(record_types)}; missing={dict(missing)}",
        )

    audit.guard(
        "S32-run-metadata",
        "32",
        "Every authoritative run saves every required metadata field or an explicit non-applicable value.",
        (metadata_path,),
        check_metadata,
    )
    def check_contradiction_handling() -> tuple[bool, str]:
        record = load_json(final_record_path)
        local_evidence = evaluate_local_work_evidence(load_csv(local_path))
        sensitivity_evidence = evaluate_sensitivity_evidence(
            load_csv(sensitivity_path)
        )
        expected_c1_c = local_evidence["c1_c_status"]
        expected_statuses = {
            "c1_a_physical_sublinear_scaling": "SUPPORTED",
            "c1_b_high_logical_shard_fanout": "CONTRADICTED",
            "c1_c_slow_local_work_decrease": expected_c1_c,
            "c1_d_fanout_times_local_work_decomposition": "INSUFFICIENT",
        }
        valid = record["subclaim_status"] == expected_statuses
        valid &= record["c1_status"] == "INSUFFICIENT"
        valid &= (
            record["section31_index_construction_status"]
            == "VERIFIED_DETERMINISTIC_SINGLE_BUILD_AUTHORITATIVE_RERUN"
        )
        valid &= (
            record.get("mechanism_checks", {}).get("slow_local_work_supported")
            is local_evidence["slow_local_work_supported"]
        )
        valid &= (
            record.get("observed_mechanism_status", {}).get(
                "slow_local_work_decrease"
            )
            == expected_c1_c
        )
        valid &= (
            record.get("e5_recall_sensitivity_status")
            == sensitivity_evidence["status"]
        )
        valid &= (
            "e2_e4_use_one_unseeded_hnsw_index_instance_per_configuration"
            not in record["anomalies"]
        )
        invalid_summary_path = (
            runs
            / "stage12-e3-deterministic-glove-200-angular-kmeans-m2-invalid-recall-ef128-summary.json"
        )
        invalid_tuning_path = (
            runs
            / "stage12-e3-deterministic-glove-200-angular-kmeans-m2-invalid-recall-selection-ef128-tuning-pinned.json"
        )
        invalid_raw_path = (
            runs
            / "per_query"
            / "stage12-e3-deterministic-glove-200-angular-kmeans-m2-invalid-recall-ef128.csv"
        )
        invalid_summary = load_json(invalid_summary_path)
        canonical_summary = load_json(
            runs / "stage12-e3-deterministic-glove-200-angular-kmeans-m2-summary.json"
        )
        valid &= invalid_tuning_path.is_file() and invalid_raw_path.is_file()
        valid &= invalid_summary["result_status"] == "INVALID_RECALL"
        valid &= float(invalid_summary["measurement_recall"]) < TARGET_RECALL
        valid &= int(invalid_summary["ef_search"]) == 128
        valid &= canonical_summary["result_status"] == "VALID_E3"
        valid &= float(canonical_summary["measurement_recall"]) >= TARGET_RECALL
        valid &= int(canonical_summary["ef_search"]) == 192
        return (
            bool(valid),
            "The superseding record applies Section 33 without generalizing K-Means or "
            f"physical causality, records measured C1-c as {expected_c1_c}, and "
            f"records E5 as {sensitivity_evidence['status']}; the failed GloVe K-Means M=2 ef=128 run remains INVALID_RECALL with raw evidence while the retuned ef=192 point passes.",
        )

    audit.guard(
        "S33-contradiction-handling",
        "33",
        "SIFT K-Means small fan-out, measured local-work contradictions, and failed physical attribution are reported without changing the baseline.",
        (
            final_record_path,
            local_path,
            sensitivity_path,
            results,
            runs
            / "stage12-e3-deterministic-glove-200-angular-kmeans-m2-invalid-recall-ef128-summary.json",
            runs
            / "stage12-e3-deterministic-glove-200-angular-kmeans-m2-invalid-recall-selection-ef128-tuning-pinned.json",
            runs
            / "per_query"
            / "stage12-e3-deterministic-glove-200-angular-kmeans-m2-invalid-recall-ef128.csv",
        ),
        check_contradiction_handling,
    )

    figure_paths = [figures / name for name in (*REQUIRED_FINAL_FIGURES, *REQUIRED_PROTOCOL_FIGURES)]

    def check_figures() -> tuple[bool, str]:
        page_counts = {path.name: pdf_text_and_pages(path)[1] for path in figure_paths}
        distance_text, _ = pdf_text_and_pages(figures / "fig_c1_local_distance_computations.pdf")
        cdf_text, _ = pdf_text_and_pages(figures / "c1_fig3_required_fanout_cdf.pdf")
        aggregate_text, _ = pdf_text_and_pages(figures / "fig_c1_aggregate_work.pdf")
        projection_text, _ = pdf_text_and_pages(figures / "fig_c1_projected_scaling.pdf")
        valid = all(count == 1 for count in page_counts.values())
        valid &= "Normalized log(N/M) reference" in distance_text
        valid &= all(f"M={shards}" in cdf_text for shards in (4, 16, 32))
        valid &= "Ideal 1/M query-work reduction" in aggregate_text
        valid &= "Projected from measured aggregate graph-search work" in projection_text
        return bool(valid), f"All {len(figure_paths)} PDFs are single-page and contain the required CDF, reference, ideal, and projection labels."

    audit.guard(
        "S34-final-figures",
        "13,17,23,28,34",
        "All protocol and publication-ready figures exist with required panels, M values, references, and labels.",
        figure_paths,
        check_figures,
    )
    table_paths = [runs / name for name in REQUIRED_TABLES]
    def check_tables() -> tuple[bool, str]:
        expected_table_shapes = {
            "c1_physical_scaleout.csv": (
                12,
                {
                    "dataset", "partition_method", "physical_machine_count",
                    "achieved_recall_mean", "qps_mean", "scaling_efficiency",
                    "mean_shards_per_query", "mean_distance_computations_per_query",
                },
            ),
            "c1_fanout_summary.csv": (
                24,
                {
                    "dataset", "partition_method", "logical_shard_count",
                    "actual_fanout_mean", "oracle_fanout_mean",
                    "actual_distribution_counts", "oracle_distribution_counts",
                },
            ),
            "c1_local_work_summary.csv": (
                24,
                {
                    "dataset", "partition_method", "logical_shard_count",
                    "mean_distance_computations_per_searched_shard",
                    "mean_nodes_visited_per_searched_shard",
                    "mean_worker_cpu_time_us_per_searched_shard", "cache_regime",
                },
            ),
            "c1_aggregate_work_summary.csv": (
                24,
                {
                    "dataset", "partition_method", "logical_shard_count",
                    "observed_distance_mean", "observed_distance_p95",
                    "observed_worker_cpu_time_us_mean", "modeled_distance_mean",
                },
            ),
            "c1_model_accuracy.csv": (
                4,
                {
                    "dataset", "partition_method", "pearson_model_vs_observed",
                    "spearman_model_vs_observed", "mean_relative_error",
                    "physical_attribution_status",
                },
            ),
        }
        valid = True
        for name, (expected_count, required_headers) in expected_table_shapes.items():
            path = runs / name
            headers, count = csv_header_and_count(path)
            valid &= count == expected_count
            valid &= required_headers.issubset(headers)
        latex = (runs / "c1_physical_scale_table.tex").read_text(encoding="utf-8")
        valid &= all(
            label in latex
            for label in (
                "Dataset", "Partition", "Workers", "Recall", "QPS",
                "Efficiency", "Fan-out", "Dist./query",
            )
        )
        valid &= sum(
            1
            for line in latex.splitlines()
            if ("SIFT1M" in line or "GloVe-200-angular" in line) and "&" in line
        ) == 12
        return (
            bool(valid),
            "All five CSVs have the required schemas and exact 12/24/24/24/4 row counts; the compact LaTeX table has all eight required columns and 12 physical M=1,2,4 rows.",
        )

    audit.guard(
        "S35-final-tables",
        "35",
        "All five CSV tables and the compact LaTeX table exist and are non-empty.",
        table_paths,
        check_tables,
    )
    audit.guard(
        "S36-final-decision",
        "36",
        "Final decision contains explicit C1-a through C1-d statuses and never marks complete C1 supported when a subclaim fails.",
        (final_record_path, results),
        lambda: (
            (lambda record, final_results: (
                set(record["subclaim_status"].values())
                <= {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT"}
                and record["c1_status"] != "SUPPORTED"
                and all(
                    marker in final_results
                    for marker in (
                        "C1-a `SUPPORTED`",
                        "C1-b `CONTRADICTED`",
                        "C1-c `CONTRADICTED`",
                        "C1-d physical attribution `INSUFFICIENT`",
                        "complete C1 `INSUFFICIENT`",
                    )
                )
                and "M=8,16,32 remain logical-shard mechanism measurements only"
                in final_results
                and "every M>4 throughput projection remains withheld" in final_results
            ))(
                load_json(final_record_path),
                " ".join(
                    results.read_text(encoding="utf-8").split(
                        "Stage 12 deterministic completion and strict protocol audit", 1
                    )[1].split()
                ),
            ),
            "The superseding record has all four explicit statuses and complete C1 remains INSUFFICIENT.",
        ),
    )
    audit.guard(
        "S37-paper-claim",
        "37",
        "The paper-ready template is withheld or weakened wherever a subclaim is unsupported.",
        (final_summary_path,),
        lambda: (
            load_json(final_summary_path).get("paper_ready_claim") is None,
            "No unsupported paper-ready causal claim is emitted while C1-b and C1-d remain unresolved.",
        ),
    )

    def check_cleanup() -> tuple[bool, str]:
        proof = load_json(cleanup_path)
        peers = proof.get("peers") or {}
        return (
            set(peers) == {"10.10.1.1", "10.10.1.2", "10.10.1.3", "10.10.1.4"}
            and all(int(value.get("http_status")) == 404 for value in peers.values())
            and proof.get("controller_storage_absent") is True,
            "Final collection is absent on all four peers and from controller storage.",
        )

    audit.guard(
        "LIFECYCLE-final-cleanup",
        "5,30",
        "The final experiment collection is deleted and verified absent on all four peers.",
        (cleanup_path,),
        check_cleanup,
    )

    def check_manifest() -> tuple[bool, str]:
        records = load_jsonl(manifest)
        ids = Counter(record.get("experiment_id") for record in records)
        stage10 = load_json(runs / "stage10-c1-final-record.json")
        stage11_path = runs / "stage11-c1-report-repair-record.json"
        stage11 = load_json(stage11_path)
        stage12 = load_json(final_record_path)
        valid = ids[stage10["experiment_id"]] == 1
        valid &= ids[stage11["experiment_id"]] == 1
        valid &= ids[stage12["experiment_id"]] == 1
        valid &= stage11["supersedes"]["experiment_id"] == stage10["experiment_id"]
        valid &= stage11["supersedes"]["record_sha256"] == sha256_path(
            runs / "stage10-c1-final-record.json"
        )
        valid &= stage12["supersedes"]["experiment_id"] == stage11["experiment_id"]
        valid &= stage12["supersedes"]["record_sha256"] == sha256_path(stage11_path)
        return (
            bool(valid),
            "Manifest counts: "
            f"stage10={ids[stage10['experiment_id']]}, "
            f"stage11={ids[stage11['experiment_id']]}, "
            f"stage12={ids[stage12['experiment_id']]}",
        )

    audit.guard(
        "PROVENANCE-manifest-supersession",
        "6,33,36",
        "Historical and superseding final records are unique, hash-linked, and preserved.",
        (
            manifest,
            runs / "stage10-c1-final-record.json",
            runs / "stage11-c1-report-repair-record.json",
            final_record_path,
        ),
        check_manifest,
    )
    return audit


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# C1 Protocol Completion Audit",
        "",
        f"Overall status: `{payload['overall_status']}`",
        "",
        f"Requirements: {payload['counts']['total']} total, {payload['counts']['pass']} pass, {payload['counts']['fail']} fail.",
        "",
        "| ID | Section | Status | Requirement | Detail |",
        "|---|---:|---|---|---|",
    ]
    for row in payload["requirements"]:
        detail = str(row["detail"]).replace("|", "\\|").replace("\n", " ")
        description = str(row["description"]).replace("|", "\\|")
        lines.append(
            f"| `{row['requirement_id']}` | {row['section']} | {row['status']} | {description} | {detail} |"
        )
    lines.extend(["", "## Evidence", ""])
    for row in payload["requirements"]:
        lines.append(f"### {row['requirement_id']}")
        lines.append("")
        if row["evidence"]:
            lines.extend(f"- `{value}`" for value in row["evidence"])
        else:
            lines.append("- No readable evidence artifact was available.")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def execute(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    audit = audit_protocol(root)
    counts = Counter(row.status for row in audit.requirements)
    payload = {
        "record_type": "c1_protocol_completion_audit",
        "protocol_sha256": sha256_path(root / "PLAN.md"),
        "overall_status": "PASS" if counts["FAIL"] == 0 else "FAIL",
        "counts": {
            "total": len(audit.requirements),
            "pass": counts["PASS"],
            "fail": counts["FAIL"],
        },
        "requirements": [asdict(row) for row in audit.requirements],
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    write_json_atomic(output_json, payload)
    write_markdown(output_markdown, payload)
    print(json.dumps(payload["counts"] | {"overall_status": payload["overall_status"]}, sort_keys=True))
    return 0 if payload["overall_status"] == "PASS" or args.allow_incomplete else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit every C1 protocol section")
    parser.add_argument("--root", default="experiments/c1")
    parser.add_argument(
        "--output-json",
        default="experiments/c1/runs/c1_completion_audit.json",
    )
    parser.add_argument(
        "--output-markdown",
        default="experiments/c1/COMPLETION_AUDIT.md",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
