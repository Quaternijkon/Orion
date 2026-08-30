from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.c23.scripts import c23_linear_matrix as matrix
from experiments.c23.scripts import c23_linear_synthesis as synthesis


def test_expected_configuration_matrix_has_42_unique_points() -> None:
    configurations = synthesis.expected_configurations()

    assert len(configurations) == 42
    assert ("sift1m", "common_m1", 1) in configurations
    assert ("glove-200-angular", "orion_no_refinement", 32) in configurations


def test_minimum_shards_many_uses_m_plus_one_failure_sentinel() -> None:
    contributions = np.asarray(
        [
            [5, 4, 1, 0],
            [3, 3, 3, 1],
            [2, 2, 2, 2],
        ],
        dtype=np.int16,
    )

    observed = synthesis.minimum_shards_many(contributions)

    assert observed.tolist() == [2, 3, 5]


def test_paired_bootstrap_is_deterministic() -> None:
    orion = np.asarray([1.0, 2.0, 3.0, 4.0])
    kmeans = np.asarray([2.0, 3.0, 4.0, 5.0])

    first = synthesis.paired_bootstrap(orion, kmeans, seed=17, replicates=1000)
    second = synthesis.paired_bootstrap(orion, kmeans, seed=17, replicates=1000)

    assert first == second
    assert first == (-1.0, -1.0, -1.0)


def test_resource_contract_enforces_exact_m_over_32_capacity() -> None:
    logical_shards = 8
    placements = []
    for shard in range(logical_shards):
        slot = shard // 4
        placements.append(
            {
                "shard_id": shard,
                "node_index": shard % 4,
                "host_slot": slot,
                "physical_core": slot,
                "cpuset": f"{slot},{slot + 16}",
            }
        )
    manifest = {
        "run_id": "test-m8",
        "logical_shards": logical_shards,
        "resource_contract": {
            "checks": {
                "linear_memory_capacity": "PASS",
                "m32_uses_all_declared_server_cores": "PASS",
                "one_physical_core_per_logical_shard": "PASS",
                "total_cpu_capacity_equals_M": "PASS",
            },
            "logical_shards": logical_shards,
            "physical_core_equivalents": logical_shards,
            "m32_capacity_fraction": logical_shards / 32,
            "memory_bytes_per_shard": 8 * 1024**3,
            "total_memory_capacity_bytes": logical_shards * 8 * 1024**3,
            "physical_hosts_with_shards": 4,
            "placement": placements,
            "shards_per_host": {str(node): 2 for node in range(4)},
        },
    }

    synthesis.validate_resource_contract(manifest)


def test_phase_snapshots_require_monotonic_cgroup_cpu_and_no_oom() -> None:
    snapshots = {}
    for phase_index, phase in enumerate(synthesis.RESOURCE_PHASES):
        snapshots[phase] = {
            "status": "PASS",
            "run_id": "test-m1",
            "logical_shards": 1,
            "timestamp": f"2026-08-24T00:00:0{phase_index}Z",
            "checks": {"all_containers_pass": "PASS"},
            "runtime": [
                {
                    "placement": {
                        "shard_id": 0,
                        "node_index": 0,
                        "cpuset": "0,16",
                    },
                    "runtime": {
                        "cgroup_cpu_stat": (
                            f"usage_usec {(phase_index + 1) * 100}\n"
                            f"user_usec {(phase_index + 1) * 60}\n"
                            f"system_usec {(phase_index + 1) * 40}\n"
                            "nr_throttled 0\nthrottled_usec 0"
                        ),
                        "cgroup_memory_current": 1024,
                        "cgroup_memory_max": str(8 * 1024**3),
                        "cgroup_memory_swap_max": "0",
                        "oom_killed": False,
                        "running": True,
                    },
                }
            ],
        }
    manifest = {
        "run_id": "test-m1",
        "dataset": "sift1m",
        "partition_method": "common_m1",
        "logical_shards": 1,
        "resource_phase_snapshots": snapshots,
    }

    rows = synthesis.validate_phase_snapshots(manifest)

    assert len(rows) == 5
    assert rows[-1]["cpu_usage_usec_since_previous_snapshot"] == 100


def test_matrix_resume_rejects_manifest_without_all_phase_snapshots(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = {
        "status": "PASS",
        "dataset": "sift1m",
        "partition_method": "random",
        "logical_shards": 2,
        "ef_search": 24,
        "query_range": [1000, 10000],
        "checks": {
            "benchmark_client_affinity": "PASS",
            "complete_e3_grid": "PASS",
            "resource_after_upload": "PASS",
            "resource_after_index": "PASS",
            "resource_after_e4": "PASS",
            "resource_after_e3": "PASS",
        },
        "resource_phase_snapshots": {phase: {} for phase in synthesis.RESOURCE_PHASES},
        "cleanup": {"restored": {"status": "PASS"}},
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert matrix.valid_completed(
        manifest_path,
        dataset="sift1m",
        method="random",
        logical_shards=2,
        ef_search=24,
        query_start=1000,
        query_count=9000,
    )

    del payload["resource_phase_snapshots"]["after_e4"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert not matrix.valid_completed(
        manifest_path,
        dataset="sift1m",
        method="random",
        logical_shards=2,
        ef_search=24,
        query_start=1000,
        query_count=9000,
    )


def test_single_hnsw_gate_audits_exact_segment_telemetry() -> None:
    manifest = {
        "logical_shards": 1,
        "upload": {"expected_points_per_shard": [100]},
        "index_gate": {
            "status": "PASS",
            "expected_nonempty_hnsw_segments": 1,
            "expected_total_segments": 2,
            "observed_total_segments": 2,
            "segment_telemetry": [
                {
                    "indexing_threshold_kib": 10,
                    "segments": [
                        {
                            "segment_type": "plain",
                            "num_vectors": 0,
                            "num_indexed_vectors": 0,
                        },
                        {
                            "segment_type": "indexed",
                            "num_vectors": 100,
                            "num_indexed_vectors": 100,
                        },
                    ],
                }
            ],
            "tail_compaction": {
                "status": "PASS",
                "affected_shards": [0],
                "temporary_indexing_threshold_kib": 1,
                "restored_indexing_threshold_kib": 10,
            },
        },
    }

    synthesis.validate_single_hnsw_index_gate(manifest, "test-m1")

    manifest["index_gate"]["segment_telemetry"][0]["segments"][0][
        "num_vectors"
    ] = 1
    with pytest.raises(ValueError, match="segment telemetry"):
        synthesis.validate_single_hnsw_index_gate(manifest, "test-m1")


def test_verdicts_are_derived_independently_for_c2_and_c3() -> None:
    summaries = []
    fixed = []
    bootstrap = []
    for dataset in synthesis.DATASETS:
        for logical_shards in synthesis.LOGICAL_SHARDS:
            summaries.extend(
                [
                    {
                        "dataset": dataset,
                        "partition_method": "kmeans",
                        "logical_shards": logical_shards,
                        "traversal_weighted_edge_cut": 0.25,
                        "max_size_over_mean": 1.2,
                    },
                    {
                        "dataset": dataset,
                        "partition_method": "orion",
                        "logical_shards": logical_shards,
                        "traversal_weighted_edge_cut": 0.35,
                        "max_size_over_mean": 1.3,
                    },
                ]
            )
            fixed.extend(
                [
                    {
                        "dataset": dataset,
                        "partition_method": method,
                        "logical_shards": logical_shards,
                        "local_recall_threshold": 0.90,
                        "success_fraction": 1.0,
                        "required_distance_computations_mean": 100.0,
                    }
                    for method in ("kmeans", "orion")
                ]
            )
            bootstrap.extend(
                [
                    {
                        "dataset": dataset,
                        "logical_shards": logical_shards,
                        "metric": "common_ef_mean_local_target_recall",
                        "conclusion": "INCONCLUSIVE",
                        "bootstrap_ci95_low": -0.01,
                        "bootstrap_ci95_high": 0.01,
                        "relative_difference_vs_kmeans": 0.0,
                    },
                    {
                        "dataset": dataset,
                        "logical_shards": logical_shards,
                        "metric": "P_HNSW",
                        "conclusion": "KMEANS_FAVORED",
                        "bootstrap_ci95_low": 0.1,
                        "bootstrap_ci95_high": 0.2,
                        "relative_difference_vs_kmeans": 0.1,
                    },
                    {
                        "dataset": dataset,
                        "logical_shards": logical_shards,
                        "metric": "W90",
                        "conclusion": "INCONCLUSIVE",
                        "bootstrap_ci95_low": -0.01,
                        "bootstrap_ci95_high": 0.01,
                        "relative_difference_vs_kmeans": 0.0,
                    },
                ]
            )

    verdicts = synthesis.build_verdicts(summaries, fixed, bootstrap, [])

    assert verdicts["C2"]["status"] == "CONTRADICTED"
    assert verdicts["C3"]["status"] == "CONTRADICTED"
