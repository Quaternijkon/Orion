#!/usr/bin/env python3
"""Run a controlled online A/B test of physical Orion shard placements.

The routed artifact, queries, lower HNSW configuration, 64-core Qdrant resource
contract, and batch size remain fixed.  Only RF=1 numeric shard ownership moves.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_TOOL = Path(__file__).resolve().with_name("orion_physical_placement_plan.py")
VIRTUAL_SCALE_TOOL = REPO_ROOT / "experiments/c1/scripts/c1_orion_virtual_scale.py"
DEFAULT_PLAN_DIR = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "orion-load-balance-r090-20260825/plan"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "orion-load-balance-r090-20260825/online-v1"
)
DEFAULT_LAYOUT_DIR = Path(
    "/users/dry/orion-distributed/native-20260818-scale4-32-r080-095-v1/"
    "artifacts/scale32/orion-r090-u48-b50-f14-g3248141"
)
DEFAULT_HDF5 = Path("/users/dry/orion-distributed/datasets/glove-200-angular.hdf5")
DEFAULT_TOPOLOGY = REPO_ROOT / "experiments/c1/topology-amd-4node.json"
DEFAULT_BASE_URL = "http://10.10.1.1:6333"
DEFAULT_RUN_ID = "c1-20260821-v2"
DEFAULT_COLLECTION = "orion_lb_r090_p24_20260825"
DEFAULT_IMPORTER = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/cargo-target-native/"
    "release/examples/orion_numeric_shard_import"
)
DEFAULT_CARGO_TARGET = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/cargo-target-native"
)
PHASES = (
    ("round-robin-a", "round_robin"),
    ("controller-tail-a", "controller_aware_tail"),
    ("size-balanced-a", "size_balanced"),
    ("round-robin-b", "round_robin"),
    ("controller-tail-b", "controller_aware_tail"),
)
EXPECTED_SHARD_COUNT = 32
EXPECTED_POINT_COUNT = 1_363_369
EXPECTED_GENERATION = 3_248_141
EXPECTED_ARTIFACT_SHA256 = "8936c14a242af48ccb675825f5a193b3307585a2a588087622bac3276577708e"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


orion_scale = load_module(VIRTUAL_SCALE_TOOL, "orion_vscale_for_placement_online")
hashall = orion_scale.hashall
simple = orion_scale.simple
fixed = orion_scale.fixed


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def run_checked(command: Sequence[str], *, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    record = {
        "timestamp": utc_timestamp(),
        "command": list(command),
        "returncode": completed.returncode,
        "seconds": time.monotonic() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }
    write_json(output, record)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return record


def collection_names(exp, base_url: str) -> list[str]:
    rows = exp.request_json(base_url, "GET", "/collections")["result"]["collections"]
    return sorted(str(row["name"]) for row in rows)


def set_full_qdrant_resources(output: Path) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                hashall.update_container_resources,
                node,
                cpuset=hashall.QDRANT_CPUSET,
                cpus=16.0,
            )
            for node in hashall.NODES
        ]
        for future in futures:
            future.result()
    containers = hashall.inspect_all_containers()
    for row in containers:
        if row["cpuset"] != hashall.QDRANT_CPUSET:
            raise RuntimeError(f"Qdrant cpuset mismatch: {row}")
        if abs(row["nano_cpus"] / 1_000_000_000.0 - 16.0) > 1e-6:
            raise RuntimeError(f"Qdrant CPU quota mismatch: {row}")
    record = {
        "timestamp": utc_timestamp(),
        "status": "PASS",
        "physical_machine_count": 4,
        "logical_shard_count": EXPECTED_SHARD_COUNT,
        "qdrant_cpu_cores_per_machine": 16,
        "qdrant_cpu_cores_total": 64,
        "qdrant_cpuset": hashall.QDRANT_CPUSET,
        "benchmark_cpuset": hashall.BENCHMARK_CPUSET,
        "containers": containers,
    }
    write_json(output, record)
    return record


def prepare_collection(args: argparse.Namespace, exp, layout: dict[str, Any]) -> dict[str, Any]:
    output = args.output_root / "prepare"
    manifest_path = output / "preparation_manifest.json"
    names = collection_names(exp, args.base_url)
    unexpected = [name for name in names if name != args.collection]
    if unexpected:
        raise RuntimeError(f"unexpected live collections before A/B test: {unexpected}")
    if args.collection in names:
        if not manifest_path.is_file():
            raise RuntimeError(
                "collection exists but this experiment has no preparation manifest; "
                "refusing ambiguous reuse"
            )
        return {
            "status": "REUSED",
            "preparation_manifest": str(manifest_path),
            "preparation_manifest_sha256": sha256_path(manifest_path),
        }

    command = [
        sys.executable,
        str(REPO_ROOT / "tools/native_auto_shard_prepare.py"),
        "--method",
        "orion",
        "--topology",
        str(args.topology),
        "--run-id",
        args.run_id,
        "--collection",
        args.collection,
        "--base-url",
        args.base_url,
        "--output-dir",
        str(output),
        "--layout-dir",
        str(args.layout_dir),
        "--allow-orion-scaling-layout",
        "--hnsw-m",
        "32",
        "--ef-construct",
        "200",
        "--max-indexing-threads",
        "1",
        "--full-scan-threshold",
        "10",
        "--indexing-threshold",
        "10",
        "--max-optimization-threads",
        "1",
        "--max-segment-size-kb",
        str(hashall.MAX_SEGMENT_SIZE_KB),
        "--batch-size",
        "2000",
        "--request-timeout-secs",
        "300",
        "--smoke-limit",
        "10",
        "--transfer-timeout-secs",
        str(args.transfer_timeout),
        "--transfer-poll-interval-secs",
        "1",
        "--transfer-method",
        "snapshot",
        "--placement-strategy",
        "round_robin",
        "--placement-peers",
        "all_peers",
        "--cargo-runner",
        str(REPO_ROOT / "tools/cargo_in_docker.sh"),
        "--cargo-target-dir",
        str(args.cargo_target_dir),
        "--importer-binary",
        str(args.importer_binary),
        "--preserve-import-checkpoint",
        "--defer-artifact-install",
    ]
    run_checked(command, output=args.output_root / "prepare-command.json")
    if not manifest_path.is_file():
        raise RuntimeError("native preparation did not emit its manifest")
    return {
        "status": "CREATED",
        "preparation_manifest": str(manifest_path),
        "preparation_manifest_sha256": sha256_path(manifest_path),
        "layout_generation": layout["parameters"]["generation"],
    }


def activate_artifact(args: argparse.Namespace, layout: dict[str, Any]) -> dict[str, Any]:
    artifact = args.layout_dir / layout["outputs"]["production_artifact"]
    if sha256_path(artifact) != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError("production artifact checksum changed")
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/method4_distributed_cluster.py"),
        "--topology",
        str(args.topology),
        "--run-id",
        args.run_id,
        "install-orion-artifact",
        "--collection",
        args.collection,
        "--generation",
        str(EXPECTED_GENERATION),
        "--artifact",
        str(artifact),
        "--expected-sha256",
        EXPECTED_ARTIFACT_SHA256,
        "--restart",
        "workers-first",
    ]
    return run_checked(command, output=args.output_root / "artifact-activation.json")


def verify_collection_contract(exp, args: argparse.Namespace) -> dict[str, Any]:
    info = exp.collection_info(args.base_url, args.collection)
    config = info.get("config") or {}
    params = config.get("params") or {}
    hnsw = config.get("hnsw_config") or {}
    policy = config.get("auto_shard_policy") or params.get("auto_shard_policy") or {}
    errors: list[str] = []
    if int(params.get("shard_number") or 0) != EXPECTED_SHARD_COUNT:
        errors.append(f"shard_number={params.get('shard_number')!r}")
    if int(params.get("replication_factor") or 0) != 1:
        errors.append(f"replication_factor={params.get('replication_factor')!r}")
    if int(info.get("points_count") or 0) != EXPECTED_POINT_COUNT:
        errors.append(f"points_count={info.get('points_count')!r}")
    indexed_vectors = int(info.get("indexed_vectors_count") or 0)
    unindexed_tail = EXPECTED_POINT_COUNT - indexed_vectors
    # Qdrant may leave a final segment below full_scan_threshold outside HNSW
    # after restart.  Those vectors remain searchable by exact full scan and the
    # collection is fully ready when the deficit is strictly below the configured
    # threshold and optimizer health is green/ok.
    if unindexed_tail < 0 or unindexed_tail >= 10:
        errors.append(
            f"indexed_vectors_count={info.get('indexed_vectors_count')!r}, "
            f"unindexed_tail={unindexed_tail}"
        )
    if hnsw.get("m") != 32 or hnsw.get("ef_construct") != 200:
        errors.append(f"hnsw={hnsw!r}")
    expected_policy = {
        "type": "orion",
        "generation": EXPECTED_GENERATION,
        "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
    }
    if policy != expected_policy:
        errors.append(f"policy={policy!r}")
    if info.get("status") != "green" or info.get("optimizer_status") != "ok":
        errors.append(
            f"health status={info.get('status')!r}, optimizer={info.get('optimizer_status')!r}"
        )
    if errors:
        raise RuntimeError("collection contract mismatch: " + "; ".join(errors))
    return {
        "status": "PASS",
        "collection": args.collection,
        "logical_shard_count": EXPECTED_SHARD_COUNT,
        "physical_point_copies": EXPECTED_POINT_COUNT,
        "indexed_vector_count": indexed_vectors,
        "full_scan_tail_vector_count": unindexed_tail,
        "replication_factor": 1,
        "hnsw": hnsw,
        "policy": policy,
        "collection_info": info,
    }


def normalize_placements(payload: dict[str, Any]) -> tuple[list[int], dict[str, dict[int, int]]]:
    peers = [int(value) for value in payload["peer_order"]]
    placements = {
        name: {int(shard_id): int(peer_id) for shard_id, peer_id in mapping.items()}
        for name, mapping in payload["placements"].items()
    }
    for name, placement in placements.items():
        if set(placement) != set(range(EXPECTED_SHARD_COUNT)):
            raise ValueError(f"placement {name} does not cover shards 0..31")
        if set(placement.values()) != set(peers):
            raise ValueError(f"placement {name} does not use the planned four peers")
    return peers, placements


def apply_placement(
    exp,
    args: argparse.Namespace,
    phase_id: str,
    strategy: str,
    peers: Sequence[int],
    placement: dict[int, int],
) -> dict[str, Any]:
    phase_root = args.output_root / "phases" / phase_id
    started = time.monotonic()
    proof = exp.move_numeric_shards_explicit(
        args.base_url,
        args.collection,
        list(peers),
        placement,
        expected_shard_count=EXPECTED_SHARD_COUNT,
        include_controller=True,
        transfer_method="snapshot",
        timeout_sec=args.transfer_timeout,
        poll_interval_sec=1.0,
    )
    record = {
        "timestamp": utc_timestamp(),
        "status": "PASS",
        "phase_id": phase_id,
        "strategy": strategy,
        "seconds": time.monotonic() - started,
        "proof": proof,
    }
    write_json(phase_root / "placement.json", record)
    return record


def aggregate_repeat_metrics(repeats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    qps = [float(row["qps"]) for row in repeats]
    hosts = sorted(repeats[0]["cpu_average_cores"])
    cpu = {
        host: statistics.fmean(float(row["cpu_average_cores"][host]) for row in repeats)
        for host in hosts
    }
    latency = {
        key: statistics.fmean(float(row["batch_latency_ms"][key]) for row in repeats)
        for key in ("p50", "p95", "p99", "max")
    }
    return {
        "qps_mean": statistics.fmean(qps),
        "qps_stdev": statistics.stdev(qps),
        "qps_cv": statistics.stdev(qps) / statistics.fmean(qps),
        "batch_latency_ms_mean_across_repeats": latency,
        "cpu_average_cores_mean_across_repeats": cpu,
        "cpu_average_cores_total_mean": sum(cpu.values()),
    }


def measure_phase(
    exp,
    args: argparse.Namespace,
    phase_id: str,
    strategy: str,
    placement_record: dict[str, Any],
    collection_contract: dict[str, Any],
    queries,
    neighbors,
) -> dict[str, Any]:
    phase_root = args.output_root / "phases" / phase_id
    benchmark_path = phase_root / "benchmark.json"
    if benchmark_path.is_file():
        existing = load_json(benchmark_path)
        if existing.get("status") != "PASS":
            raise RuntimeError(f"existing phase result is not PASS: {benchmark_path}")
        return existing
    parsed = urllib.parse.urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6333
    endpoint = (
        f"/collections/{urllib.parse.quote(args.collection, safe='')}/points/search/batch"
    )
    tuning_queries, tuning_neighbors = queries[:1000], neighbors[:1000]
    heldout_queries, heldout_neighbors = queries[1000:10000], neighbors[1000:10000]
    tuning = simple.recall_probe(
        host,
        port,
        endpoint,
        tuning_queries,
        tuning_neighbors,
        batch_size=args.batch_size,
        top_k=10,
    )
    heldout = simple.recall_probe(
        host,
        port,
        endpoint,
        heldout_queries,
        heldout_neighbors,
        batch_size=args.batch_size,
        top_k=10,
    )
    if float(heldout["recall_at_10"]) < args.target_recall:
        raise RuntimeError(f"{phase_id} misses Recall@10 gate: {heldout}")
    bodies = simple.make_simple_bodies(heldout_queries, batch_size=args.batch_size, top_k=10)
    sweep = [
        hashall.timed_run(
            host,
            port,
            endpoint,
            bodies,
            concurrency=concurrency,
            duration_s=args.sweep_seconds,
            batch_size=args.batch_size,
        )
        for concurrency in args.concurrency_candidates
    ]
    selection = hashall.select_saturation(sweep)
    if not selection["knee_observed"]:
        raise RuntimeError(f"{phase_id}: saturation knee not observed")
    selected = int(selection["selected_concurrency"])
    warmup = hashall.timed_run(
        host,
        port,
        endpoint,
        bodies,
        concurrency=selected,
        duration_s=args.warmup_seconds,
        batch_size=args.batch_size,
    )
    repeats: list[dict[str, Any]] = []
    while len(repeats) < args.min_repeats:
        repeats.append(
            hashall.timed_run(
                host,
                port,
                endpoint,
                bodies,
                concurrency=selected,
                duration_s=args.measure_seconds,
                batch_size=args.batch_size,
            )
        )
    while len(repeats) < args.max_repeats:
        aggregate = aggregate_repeat_metrics(repeats)
        if aggregate["qps_cv"] <= args.max_cv:
            break
        repeats.append(
            hashall.timed_run(
                host,
                port,
                endpoint,
                bodies,
                concurrency=selected,
                duration_s=args.measure_seconds,
                batch_size=args.batch_size,
            )
        )
    aggregate = aggregate_repeat_metrics(repeats)
    if aggregate["qps_cv"] > args.max_cv:
        raise RuntimeError(f"{phase_id}: QPS CV remains high: {aggregate['qps_cv']:.4f}")
    ending_contract = verify_collection_contract(exp, args)
    ending_cluster = exp.collection_cluster_info(args.base_url, args.collection)
    ending_placement = exp.validate_numeric_shard_explicit_placement(
        ending_contract["collection_info"],
        ending_cluster,
        sorted(set(placement_record["proof"]["expected_placement"].values())),
        EXPECTED_SHARD_COUNT,
        {int(k): int(v) for k, v in placement_record["proof"]["expected_placement"].items()},
        include_controller=True,
    )
    result = {
        "timestamp": utc_timestamp(),
        "record_type": "orion_physical_placement_online_ab",
        "status": "PASS",
        "phase_id": phase_id,
        "strategy": strategy,
        "physical_machine_count": 4,
        "logical_shard_count": EXPECTED_SHARD_COUNT,
        "collection": args.collection,
        "request_contract": {
            "endpoint": endpoint,
            "standard_coordinator_request": True,
            "client_side_fanout": False,
            "shard_selector_present": False,
            "server_router": "native_orion",
        },
        "fixed_contract": {
            "dataset": "GloVe-200-angular",
            "distance": "Cosine",
            "top_k": 10,
            "batch_size": args.batch_size,
            "tuning_query_range": [0, 1000],
            "heldout_query_range": [1000, 10000],
            "target_recall_at_10": args.target_recall,
            "artifact_generation": EXPECTED_GENERATION,
            "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
            "upper_k": 48,
            "upper_search_ef": 48,
            "dynamic_ef_base": 50,
            "dynamic_ef_factor": 14,
            "qdrant_cpu_cores_per_machine": 16,
            "qdrant_cpu_cores_total": 64,
        },
        "placement": placement_record,
        "collection_contract_start": collection_contract,
        "tuning_recall": tuning,
        "heldout_recall": heldout,
        "concurrency_sweep": sweep,
        "saturation_selection": selection,
        "warmup": warmup,
        "repeats": repeats,
        "selected_concurrency": selected,
        "repeat_count": len(repeats),
        **aggregate,
        "collection_contract_end": ending_contract,
        "ending_placement": ending_placement,
        "per_node_collection_cluster": hashall.per_node_collection_cluster(
            exp, args.collection
        ),
    }
    write_json(benchmark_path, result)
    return result


def summarize(args: argparse.Namespace, results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        latency = result["batch_latency_ms_mean_across_repeats"]
        rows.append(
            {
                "phase_id": result["phase_id"],
                "strategy": result["strategy"],
                "recall_at_10": result["heldout_recall"]["recall_at_10"],
                "selected_concurrency": result["selected_concurrency"],
                "repeat_count": result["repeat_count"],
                "qps_mean": result["qps_mean"],
                "qps_stdev": result["qps_stdev"],
                "qps_cv": result["qps_cv"],
                "batch_latency_p50_ms": latency["p50"],
                "batch_latency_p95_ms": latency["p95"],
                "batch_latency_p99_ms": latency["p99"],
            }
        )
    summary_csv = args.output_root / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(row["strategy"], []).append(row)
    strategy_summary = {
        strategy: {
            "phase_count": len(strategy_rows),
            "qps_mean_across_phases": statistics.fmean(
                float(row["qps_mean"]) for row in strategy_rows
            ),
            "qps_min": min(float(row["qps_mean"]) for row in strategy_rows),
            "qps_max": max(float(row["qps_mean"]) for row in strategy_rows),
            "recall_min": min(float(row["recall_at_10"]) for row in strategy_rows),
            "selected_concurrencies": [
                int(row["selected_concurrency"]) for row in strategy_rows
            ],
        }
        for strategy, strategy_rows in by_strategy.items()
    }
    ranking = sorted(
        strategy_summary,
        key=lambda strategy: strategy_summary[strategy]["qps_mean_across_phases"],
        reverse=True,
    )
    winner = ranking[0]
    round_robin = strategy_summary["round_robin"]
    candidate = strategy_summary["controller_aware_tail"]
    candidate_confirmed = (
        candidate["phase_count"] == 2
        and round_robin["phase_count"] == 2
        and candidate["qps_min"] > round_robin["qps_max"]
        and candidate["recall_min"] >= args.target_recall
    )
    summary = {
        "timestamp": utc_timestamp(),
        "status": "PASS",
        "physical_machine_count": 4,
        "logical_shard_count": EXPECTED_SHARD_COUNT,
        "rows": rows,
        "strategy_summary": strategy_summary,
        "ranking_by_online_qps_mean": ranking,
        "online_winner": winner,
        "controller_aware_tail_confirmed_over_round_robin": candidate_confirmed,
        "confirmation_rule": (
            "both controller-aware phases exceed both round-robin phases and all "
            "held-out Recall@10 values satisfy the target"
        ),
        "summary_csv": str(summary_csv),
        "summary_csv_sha256": sha256_path(summary_csv),
    }
    write_json(args.output_root / "summary.json", summary)
    return summary


def completion_audit(
    args: argparse.Namespace,
    results: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "five_interleaved_phases": [row["phase_id"] for row in results]
        == [phase_id for phase_id, _strategy in PHASES],
        "round_robin_repeated": sum(row["strategy"] == "round_robin" for row in results)
        == 2,
        "controller_candidate_repeated": sum(
            row["strategy"] == "controller_aware_tail" for row in results
        )
        == 2,
        "size_balanced_measured": sum(
            row["strategy"] == "size_balanced" for row in results
        )
        == 1,
        "all_recall_pass": all(
            float(row["heldout_recall"]["recall_at_10"]) >= args.target_recall
            for row in results
        ),
        "all_have_five_repeats": all(int(row["repeat_count"]) >= 5 for row in results),
        "all_cv_pass": all(float(row["qps_cv"]) <= args.max_cv for row in results),
        "all_have_knee": all(
            bool(row["saturation_selection"]["knee_observed"]) for row in results
        ),
        "all_physical_machine_count_four": all(
            int(row["physical_machine_count"]) == 4 for row in results
        ),
        "all_logical_shard_count_32": all(
            int(row["logical_shard_count"]) == 32 for row in results
        ),
        "summary_has_online_winner": bool(summary.get("online_winner")),
    }
    audit = {
        "timestamp": utc_timestamp(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
    }
    write_json(args.output_root / "completion-audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"completion audit failed: {checks}")
    return audit


def execute(args: argparse.Namespace) -> int:
    if not args.plan_dir.is_dir():
        raise FileNotFoundError(f"placement plan is missing: {args.plan_dir}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    exp = hashall.load_experiment_module(REPO_ROOT)
    placements_payload = load_json(args.plan_dir / "placements.json")
    peers, placements = normalize_placements(placements_payload)
    plan_manifest = load_json(args.plan_dir / "manifest.json")
    if plan_manifest.get("status") != "OFFLINE_PLAN_ONLY":
        raise RuntimeError("placement plan has an unexpected status")
    layout = load_json(args.layout_dir / "build-manifest.json")
    if int(layout["parameters"]["generation"]) != EXPECTED_GENERATION:
        raise RuntimeError("layout generation changed")
    selected_phases = [
        phase for phase in PHASES if not args.phase_ids or phase[0] in args.phase_ids
    ]
    unknown = set(args.phase_ids) - {phase_id for phase_id, _strategy in PHASES}
    if unknown:
        raise ValueError(f"unknown phase IDs: {sorted(unknown)}")
    original_resources_path = args.output_root / "original-resources.json"
    if not original_resources_path.is_file():
        write_json(
            original_resources_path,
            {"timestamp": utc_timestamp(), "containers": hashall.inspect_all_containers()},
        )
    body_error: BaseException | None = None
    completed_results: list[dict[str, Any]] = []
    try:
        hashall.wait_cluster_ready(exp, args.base_url)
        set_full_qdrant_resources(args.output_root / "resource-contract.json")
        preparation = prepare_collection(args, exp, layout)
        write_json(args.output_root / "preparation.json", preparation)
        activate_artifact(args, layout)
        hashall.wait_cluster_ready(exp, args.base_url)
        collection_contract = verify_collection_contract(exp, args)
        write_json(args.output_root / "collection-contract.json", collection_contract)
        queries, neighbors = fixed.load_dataset(args.hdf5_path, 10)
        for phase_id, strategy in selected_phases:
            placement = apply_placement(
                exp, args, phase_id, strategy, peers, placements[strategy]
            )
            hashall.wait_cluster_ready(exp, args.base_url)
            contract = verify_collection_contract(exp, args)
            result = measure_phase(
                exp,
                args,
                phase_id,
                strategy,
                placement,
                contract,
                queries,
                neighbors,
            )
            completed_results.append(result)
            print(
                json.dumps(
                    {
                        "phase_id": phase_id,
                        "strategy": strategy,
                        "recall_at_10": result["heldout_recall"]["recall_at_10"],
                        "selected_concurrency": result["selected_concurrency"],
                        "qps_mean": result["qps_mean"],
                        "qps_cv": result["qps_cv"],
                    }
                ),
                flush=True,
            )
        if tuple(selected_phases) != PHASES:
            return 0
        summary = summarize(args, completed_results)
        completion_audit(args, completed_results, summary)
        write_json(
            args.output_root / "execution-complete.json",
            {
                "timestamp": utc_timestamp(),
                "status": "ONLINE_AB_COMPLETE",
                "completed_phases": [row["phase_id"] for row in completed_results],
                "online_winner": summary["online_winner"],
            },
        )
        return 0
    except BaseException as error:
        body_error = error
        write_json(
            args.output_root / "execution-failed.json",
            {
                "timestamp": utc_timestamp(),
                "status": "FAILED",
                "completed_phases": [row["phase_id"] for row in completed_results],
                "error": repr(error),
            },
        )
        raise
    finally:
        if tuple(selected_phases) == PHASES and body_error is None and not args.keep_collection:
            exp.delete_collection_if_exists(args.base_url, args.collection)
            write_json(
                args.output_root / "collection-cleanup.json",
                {
                    "timestamp": utc_timestamp(),
                    "status": "PASS",
                    "deleted_collection": args.collection,
                    "remaining_collections": collection_names(exp, args.base_url),
                },
            )
        original = load_json(original_resources_path)["containers"]
        try:
            hashall.restore_resource_state(
                original, args.output_root / "resource-restored.json"
            )
        except Exception:
            if body_error is None:
                raise


def parse_int_csv(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or result != sorted(set(result)) or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("concurrency list must be sorted unique positives")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cargo-target-dir", type=Path, default=DEFAULT_CARGO_TARGET)
    parser.add_argument("--importer-binary", type=Path, default=DEFAULT_IMPORTER)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument(
        "--concurrency-candidates", type=parse_int_csv, default=parse_int_csv("1,2,4,8,16,32,64")
    )
    parser.add_argument("--sweep-seconds", type=float, default=8.0)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--measure-seconds", type=float, default=20.0)
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--max-repeats", type=int, default=7)
    parser.add_argument("--max-cv", type=float, default=0.05)
    parser.add_argument("--transfer-timeout", type=float, default=10_800.0)
    parser.add_argument("--phase", dest="phase_ids", action="append", default=[])
    parser.add_argument("--keep-collection", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "topology",
        "layout_dir",
        "hdf5_path",
        "plan_dir",
        "output_root",
        "cargo_target_dir",
        "importer_binary",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.batch_size != 200:
        raise ValueError("formal placement experiment requires batch-size=200")
    if args.min_repeats < 5 or args.max_repeats < args.min_repeats:
        raise ValueError("formal placement experiment requires at least five repeats")
    if args.max_cv <= 0:
        raise ValueError("max CV must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
