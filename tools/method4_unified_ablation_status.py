#!/usr/bin/env python3
"""Write a status matrix for the planned Method4 unified ablation table.

This is intentionally a status/caveat table, not a strict same-run ablation
performance matrix. It prevents mixing existing evidence with incompatible
measurement scopes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "ablation_group",
    "variant",
    "mechanism_disabled",
    "evidence_scope",
    "status",
    "decision",
    "available_metrics",
    "primary_artifacts",
    "raw_run_dirs",
    "caveat",
    "next_step",
]


def default_status_rows() -> list[dict[str, Any]]:
    return [
        {
            "ablation_group": "full_method4_reference",
            "variant": "Full Method4 / current selected configs",
            "mechanism_disabled": "none",
            "evidence_scope": "cross-claim reference rows",
            "status": "available_but_multi_source",
            "decision": "use_as_reference_index_not_strict_unified_cell",
            "available_metrics": "Recall@10;QPS;visited_shards;EF-sum;selected P95/P99 batch latency",
            "primary_artifacts": (
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_d_method4_vs_naive_same_recall.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_d_high_recall_latency_pairs.csv"
            ),
            "raw_run_dirs": (
                "results/qdrant_compare_current_idea_vs_naive_095_idea_b200/20260603_094930/; "
                "results/method4_claim_d_high_recall_latency_20260704/"
            ),
            "caveat": "Reference rows come from several matched experiments, not one unified ablation image.",
            "next_step": "Use only as cross-claim reference unless a strict same-image ablation run is added.",
        },
        {
            "ablation_group": "dynamic_ef",
            "variant": "Dynamic EF vs fixed EF",
            "mechanism_disabled": "dynamic per-shard EF disabled in fixed-EF control",
            "evidence_scope": "same live Orion collection, selected 0.95 batch-latency supplement plus prior QPS frontier",
            "status": "available",
            "decision": "include",
            "available_metrics": "Recall@10;QPS;visited_shards;EF-sum;P95/P99 batch latency;GT-hit EP relevance",
            "primary_artifacts": (
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_c_dynamic_vs_fixed_performance_summary.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_c_dynamic_vs_fixed_latency_deltas.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_c_routed_ep_relevance_summary.csv"
            ),
            "raw_run_dirs": (
                "results/method4_claim_c_orion_frontier_fixed_095_robust_confirm_20260625/; "
                "results/method4_claim_c_orion_frontier_dynamic_095_robust_confirm_20260625/; "
                "results/method4_claim_c_dynamic_vs_fixed_latency_20260704/analysis_20260704_235925/"
            ),
            "caveat": "Latency supplement is client-observed batch endpoint time, not server-internal lower-search trace.",
            "next_step": "Optional only: add server-side lower-search latency tracing for internal P99 wording.",
        },
        {
            "ablation_group": "compact_request_execution",
            "variant": "compact_current vs grouped/materialized and client-expanded modes",
            "mechanism_disabled": "compact multi-EP request/execution disabled in controls",
            "evidence_scope": "current-runtime execution-mode matrix",
            "status": "available_with_overhead_metric_gap",
            "decision": "include_with_caveat",
            "available_metrics": "Recall@10;QPS;P95/P99 batch latency;request_objects_per_query;request_body_bytes/query;response_body_bytes/query;route_planning_ms/query;container_network_bytes/query;controller_cpu_pct",
            "primary_artifacts": (
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_e_execution_mode_latency_matrix.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_e_execution_mode_latency_deltas.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_e_payload_bytes_summary.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_e_wire_bytes_summary.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_e_planning_time_summary.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_e_container_overhead_summary.csv"
            ),
            "raw_run_dirs": (
                "results/method4_claim_e_execution_mode_latency_20260704/analysis_20260704_225847/; "
                "results/method4_claim_e_planning_time_20260705/analysis_20260705_021347/; "
                "results/method4_claim_e_container_overhead_20260705/analysis_20260705_024218/"
            ),
            "caveat": "Does not recreate historical old binary; client-observed request/response body bytes, client-side route-planning time, and Docker container-level CPU/network counters are present, but physical NIC capture and controller-internal CPU attribution are missing.",
            "next_step": "Instrument physical NIC bytes or controller-internal CPU only if those stronger overhead metrics are needed.",
        },
        {
            "ablation_group": "worker_local_premerge",
            "variant": "direct-peer local pre-merge vs direct-peer no-premerge",
            "mechanism_disabled": "worker-local pre-merge disabled in direct-peer control",
            "evidence_scope": "direct-peer simulation plus server QPS A/B evidence",
            "status": "available_with_runtime_caveat",
            "decision": "include_with_caveat",
            "available_metrics": "Recall@10;QPS;P95/P99 batch latency;candidate groups/query;returned candidates/query",
            "primary_artifacts": (
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_f_premerge_batch_latency_matrix.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_f_premerge_batch_latency_deltas.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_f_worker_local_premerge.csv"
            ),
            "raw_run_dirs": "results/method4_claim_f_premerge_batch_latency_20260704/analysis_20260704_222700/",
            "caveat": "Batch matrix is direct-peer simulation/current coordinator path, not production enabled-vs-disabled P95/P99 server restart A/B.",
            "next_step": "Optional: restart server with pre-merge enabled/disabled for strict production P95/P99 A/B.",
        },
        {
            "ablation_group": "physical_placement",
            "variant": "method4-aware placement vs round-robin and size-balanced",
            "mechanism_disabled": "method4-aware co-routing placement disabled in controls",
            "evidence_scope": "matched physical layout offline + online batch-latency matrix",
            "status": "available",
            "decision": "include_as_placement_submatrix",
            "available_metrics": "worker load CV;P95 max worker EF;Recall@10;QPS;P95/P99 batch latency",
            "primary_artifacts": (
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_g_online_batch_latency_summary.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_g_online_batch_latency_deltas.csv; "
                "results/method4_claim_g_evidence_20260704/"
            ),
            "raw_run_dirs": "results/method4_claim_g_batch_latency_20260704/",
            "caveat": "Online gains are modest and this is a matched placement submatrix, not a mechanism-off single row.",
            "next_step": "Optional: add higher-concurrency matched-layout run if stronger placement tail claim is needed.",
        },
        {
            "ablation_group": "multi_assignment",
            "variant": "Orion single assignment vs voting/default multi-assignment",
            "mechanism_disabled": "multi-assignment disabled in single-assignment control",
            "evidence_scope": "frontier/oracle evidence only; no strict 3-repeat online latency cell",
            "status": "partial_proxy_only",
            "decision": "do_not_use_as_strict_unified_latency_cell",
            "available_metrics": "Recall@10;QPS single-run frontier;index expansion;oracle_gt_miss",
            "primary_artifacts": (
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_b_multiassign_expansion_qps.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_b_orion_oracle_gt_miss.csv"
            ),
            "raw_run_dirs": "results/method4_benchmark_matrix/method4_multiassign_expansion_qps_20260629_*/",
            "caveat": "Online rows are mostly single-run frontier/final evals and currently lack P95/P99 batch latency in the strict unified matrix scope.",
            "next_step": "Restore or rebuild single/default multi-assignment live collections and run selected 0.95 configs with 3 repeats and batch P95/P99.",
        },
        {
            "ablation_group": "topology_no_fission",
            "variant": "topology-aware no-fission/no-auto-sharding vs full fission",
            "mechanism_disabled": "fission/auto-sharding disabled while retaining topology convergence",
            "evidence_scope": "offline oracle plus selected 0.95 online P95/P99 latency cell",
            "status": "available_selected_latency_cell",
            "decision": "include_as_selected_online_ablation_cell",
            "available_metrics": "Recall@10;QPS;visited_shards;EF-sum;P95/P99 batch latency;offline oracle routed shards;oracle_gt_miss;GT entropy;edge cut",
            "primary_artifacts": (
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_a_partition_oracle_matrix.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_a_partition_online_latency_submatrix.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_a_partition_online_latency_submatrix_deltas.csv; "
                "results/method4_claim_coverage_20260704/derived_claim_tables/"
                "claim_a_topology_no_fission_config_sensitivity.csv"
            ),
            "raw_run_dirs": (
                "results/method4_claim_a_partition_oracle_20260704/analysis_20260704_213047/; "
                "results/strict_ablation_topology_no_fission_20260705/reuse_tune_095/20260705_012313/; "
                "results/method4_claim_a_partition_online_latency_20260704/topology_no_fission_selected/"
                "analysis_20260705_012559/"
            ),
            "caveat": "Selected 0.95 client-observed batch latency cell plus same-run config sensitivity, not the full random/KMeans/topology/full-fission online partition-family matrix.",
            "next_step": "Strict multi-assignment online P95/P99 cell remains missing; random/full partition-family online matrix remains optional for Claim A breadth.",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Method4 unified ablation status matrix.")
    parser.add_argument(
        "--output",
        default="results/method4_claim_coverage_20260704/derived_claim_tables/unified_ablation_status_matrix.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    write_csv(output, default_status_rows())
    print(output)


if __name__ == "__main__":
    main()
