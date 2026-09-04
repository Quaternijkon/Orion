#!/usr/bin/env python3
"""Plan and audit physical placement for one fixed native Orion layout.

This tool deliberately stops at an offline workload prediction.  It emits exact
shard-to-peer maps that a separate live experiment can apply, but it never calls
the predicted winner an online QPS winner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_LAYOUT_DIR = Path(
    "/users/dry/orion-distributed/native-20260818-scale4-32-r080-095-v1/"
    "artifacts/scale32/orion-r090-u48-b50-f14-g3248141"
)
DEFAULT_TRACE = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "orion-load-balance-r090-20260825/trace/p24-per-query.json"
)
DEFAULT_BASELINE_BENCHMARK = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "orion-virtual-linear-cpu-glove-20260825/measurements-v1/"
    "m32/exact-s32/benchmark.json"
)
DEFAULT_BASELINE_PLACEMENT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "orion-virtual-linear-cpu-glove-20260825/measurements-v1/"
    "layouts/p24/native-prepare/placement_map.json"
)
DEFAULT_TOPOLOGY = Path(__file__).resolve().parents[1] / "c1/topology-amd-4node.json"
DEFAULT_OUTPUT_DIR = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/"
    "orion-load-balance-r090-20260825/plan"
)
DEFAULT_BASE_URL = "http://10.10.1.1:6333"
PHYSICAL_MACHINE_COUNT = 4
EXPECTED_SHARD_COUNT = 32

# This grouping was selected from the fixed 10,000-query production-router trace
# by minimizing average-node and per-query tail load together.  Peer assignment is
# still chosen at run time to minimize movement from the recorded round-robin map.
CONTROLLER_TAIL_BINS = (
    (4, 5, 8, 16, 27, 28, 29, 30),
    (1, 2, 12, 13, 17, 22, 26, 31),
    (6, 9, 10, 14, 19, 20, 21, 23),
    (0, 3, 7, 11, 15, 18, 24, 25),
)


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


def write_csv_new(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def cluster_peer_uris(base_url: str) -> tuple[int, dict[int, str]]:
    url = base_url.rstrip("/") + "/cluster"
    with urllib.request.urlopen(url, timeout=10.0) as response:
        payload = json.load(response)["result"]
    controller = int(payload["peer_id"])
    peers = {
        int(peer_id): str(row["uri"]).rstrip("/")
        for peer_id, row in payload["peers"].items()
    }
    return controller, peers


def topology_hosts(topology: dict[str, Any]) -> list[str]:
    controller = str(topology["controller"]["private_ip"])
    workers = [str(worker["private_ip"]) for worker in topology["workers"]]
    hosts = [controller, *workers]
    if len(hosts) != PHYSICAL_MACHINE_COUNT or len(set(hosts)) != len(hosts):
        raise ValueError(f"expected four unique physical hosts, got {hosts}")
    return hosts


def peer_hosts(controller_peer: int, peer_uris: dict[int, str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for peer_id, uri in peer_uris.items():
        hostname = urllib.parse.urlparse(uri).hostname
        if hostname is None:
            raise ValueError(f"peer URI has no hostname: {uri!r}")
        result[peer_id] = hostname
    if controller_peer not in result:
        raise ValueError("controller peer is absent from the cluster peer map")
    return result


def normalize_placement(raw: dict[str, Any] | dict[int, Any]) -> dict[int, int]:
    placement = {int(shard_id): int(peer_id) for shard_id, peer_id in raw.items()}
    expected = set(range(EXPECTED_SHARD_COUNT))
    if set(placement) != expected:
        raise ValueError(
            "placement must cover shard IDs 0..31: "
            f"missing={sorted(expected - set(placement))}, "
            f"extra={sorted(set(placement) - expected)}"
        )
    return placement


def baseline_peer_order(placement: dict[int, int], controller_peer: int) -> list[int]:
    order = [placement[index] for index in range(PHYSICAL_MACHINE_COUNT)]
    if len(set(order)) != PHYSICAL_MACHINE_COUNT:
        raise ValueError(f"baseline shards 0..3 do not cover four peers: {order}")
    if order[0] != controller_peer:
        raise ValueError(
            f"baseline shard 0 belongs to peer {order[0]}, expected controller {controller_peer}"
        )
    expected = {
        shard_id: order[shard_id % PHYSICAL_MACHINE_COUNT]
        for shard_id in range(EXPECTED_SHARD_COUNT)
    }
    if placement != expected:
        raise ValueError("recorded baseline placement is not exact four-peer round-robin")
    return order


def shard_average_work(trace: dict[str, Any], shard_counts: Sequence[int]) -> list[float]:
    query_count = int(trace["aggregate"]["query_count"])
    if query_count != len(trace["per_query"]):
        raise ValueError("trace aggregate query_count does not match per_query rows")
    totals = [0.0 for _ in shard_counts]
    for row in trace["per_query"]:
        for target in row["targets"]:
            shard_id = int(target["shard_id"])
            totals[shard_id] += float(target["ef"]) * math.log2(shard_counts[shard_id])
    return [total / query_count for total in totals]


def selected_sweep_row(benchmark: dict[str, Any]) -> dict[str, Any]:
    selected = int(benchmark["parameters"]["selected_concurrency"])
    matches = [
        row for row in benchmark["concurrency_sweep"]
        if int(row["concurrency"]) == selected
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one baseline sweep row for concurrency {selected}")
    return matches[0]


def controller_overhead_model(
    benchmark: dict[str, Any],
    peer_order: Sequence[int],
    peer_to_host: dict[int, str],
    average_work: Sequence[float],
) -> dict[str, Any]:
    sweep = selected_sweep_row(benchmark)
    cpu = {str(host): float(value) for host, value in sweep["cpu_average_cores"].items()}
    node_work = [
        sum(average_work[shard_id] for shard_id in range(index, EXPECTED_SHARD_COUNT, 4))
        for index in range(PHYSICAL_MACHINE_COUNT)
    ]
    worker_x: list[float] = []
    worker_y: list[float] = []
    for index in range(1, PHYSICAL_MACHINE_COUNT):
        host = peer_to_host[peer_order[index]]
        worker_x.append(node_work[index])
        worker_y.append(cpu[host])
    slope = sum(x * y for x, y in zip(worker_x, worker_y, strict=True)) / sum(
        x * x for x in worker_x
    )
    controller_host = peer_to_host[peer_order[0]]
    overhead = cpu[controller_host] / slope - node_work[0]
    predictions = [slope * work for work in worker_x]
    residuals = [actual - predicted for actual, predicted in zip(worker_y, predictions, strict=True)]
    return {
        "model": "worker CPU cores = slope * sum(EF * log2(shard_points)); fitted through origin",
        "baseline_selected_concurrency": int(sweep["concurrency"]),
        "worker_slope_cores_per_work_unit": slope,
        "worker_observed_cpu_cores": worker_y,
        "worker_predicted_cpu_cores": predictions,
        "worker_residual_cpu_cores": residuals,
        "controller_observed_cpu_cores": cpu[controller_host],
        "controller_local_work_units": node_work[0],
        "controller_overhead_work_units": overhead,
    }


def lpt_size_balanced_bins(shard_counts: Sequence[int]) -> list[list[int]]:
    bins: list[list[int]] = [[] for _ in range(PHYSICAL_MACHINE_COUNT)]
    loads = [0 for _ in bins]
    for shard_id in sorted(range(len(shard_counts)), key=lambda item: (-shard_counts[item], item)):
        choices = [index for index, shards in enumerate(bins) if len(shards) < 8]
        target = min(choices, key=lambda index: (loads[index], len(bins[index]), index))
        bins[target].append(shard_id)
        loads[target] += int(shard_counts[shard_id])
    if sorted(map(len, bins)) != [8, 8, 8, 8]:
        raise AssertionError(f"size-balanced bins are not 8/8/8/8: {bins}")
    return [sorted(shards) for shards in bins]


def assign_bins_min_movement(
    bins: Sequence[Sequence[int]],
    peers: Sequence[int],
    baseline: dict[int, int],
    shard_counts: Sequence[int],
    *,
    fixed_first_peer: int | None = None,
) -> dict[int, int]:
    if len(bins) != len(peers):
        raise ValueError("bin and peer counts differ")
    permutations = itertools.permutations(peers)
    if fixed_first_peer is not None:
        permutations = (
            permutation for permutation in permutations if permutation[0] == fixed_first_peer
        )

    def score(permutation: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
        moved = 0
        moved_weight = 0
        for bin_index, shards in enumerate(bins):
            peer_id = permutation[bin_index]
            for shard_id in shards:
                if baseline[shard_id] != peer_id:
                    moved += 1
                    moved_weight += int(shard_counts[shard_id])
        return moved, moved_weight, permutation

    selected = min(permutations, key=score)
    return {
        int(shard_id): int(selected[bin_index])
        for bin_index, shards in enumerate(bins)
        for shard_id in shards
    }


def placement_metrics(
    name: str,
    placement: dict[int, int],
    peer_order: Sequence[int],
    controller_peer: int,
    shard_counts: Sequence[int],
    trace: dict[str, Any],
    controller_overhead: float,
    baseline: dict[int, int],
) -> dict[str, Any]:
    counts = Counter(placement.values())
    if set(placement.values()) != set(peer_order):
        raise ValueError(f"{name} does not use exactly the four expected peers")
    if any(counts[peer_id] != 8 for peer_id in peer_order):
        raise ValueError(f"{name} is not strict 8-shards-per-peer: {counts}")
    node_sums = {peer_id: 0.0 for peer_id in peer_order}
    per_query_peaks: list[float] = []
    for query in trace["per_query"]:
        work = {peer_id: 0.0 for peer_id in peer_order}
        work[controller_peer] = controller_overhead
        for target in query["targets"]:
            shard_id = int(target["shard_id"])
            work[placement[shard_id]] += float(target["ef"]) * math.log2(
                shard_counts[shard_id]
            )
        for peer_id, value in work.items():
            node_sums[peer_id] += value
        per_query_peaks.append(max(work.values()))
    query_count = len(trace["per_query"])
    averages = {peer_id: value / query_count for peer_id, value in node_sums.items()}
    moved_shards = sorted(
        shard_id for shard_id in placement if placement[shard_id] != baseline[shard_id]
    )
    return {
        "strategy": name,
        "placement": placement,
        "shards_by_peer": {
            peer_id: sorted(shard_id for shard_id, owner in placement.items() if owner == peer_id)
            for peer_id in peer_order
        },
        "physical_shards_per_peer": {peer_id: counts[peer_id] for peer_id in peer_order},
        "average_work_by_peer": averages,
        "average_node_load_peak": max(averages.values()),
        "per_query_node_peak_mean": statistics.fmean(per_query_peaks),
        "per_query_node_peak_p95": percentile(per_query_peaks, 0.95),
        "per_query_node_peak_p99": percentile(per_query_peaks, 0.99),
        "moved_shard_count_from_baseline": len(moved_shards),
        "moved_shards_from_baseline": moved_shards,
        "moved_physical_vector_copies_from_baseline": sum(
            int(shard_counts[shard_id]) for shard_id in moved_shards
        ),
    }


def plan(args: argparse.Namespace) -> Path:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    layout_dir = args.layout_dir.expanduser().resolve()
    build_manifest_path = layout_dir / "build-manifest.json"
    build_manifest = load_json(build_manifest_path)
    trace = load_json(args.trace)
    benchmark = load_json(args.baseline_benchmark)
    baseline_payload = load_json(args.baseline_placement)
    baseline = normalize_placement(baseline_payload["target_placement"])
    shard_counts = [int(value) for value in build_manifest["routing"]["shard_counts"]]
    if len(shard_counts) != EXPECTED_SHARD_COUNT:
        raise ValueError(f"expected 32 shard counts, got {len(shard_counts)}")
    if int(trace["artifact"]["shard_count"]) != EXPECTED_SHARD_COUNT:
        raise ValueError("route trace is not for the 32-shard artifact")

    topology = load_json(args.topology)
    physical_hosts = topology_hosts(topology)
    controller_peer, peer_uris = cluster_peer_uris(args.base_url)
    peer_to_host = peer_hosts(controller_peer, peer_uris)
    if set(peer_to_host.values()) != set(physical_hosts):
        raise ValueError(
            "live cluster hosts do not match topology: "
            f"live={peer_to_host}, topology={physical_hosts}"
        )
    peer_order = baseline_peer_order(baseline, controller_peer)
    average_work = shard_average_work(trace, shard_counts)
    overhead_model = controller_overhead_model(
        benchmark, peer_order, peer_to_host, average_work
    )
    controller_overhead = float(overhead_model["controller_overhead_work_units"])

    size_bins = lpt_size_balanced_bins(shard_counts)
    size_placement = assign_bins_min_movement(
        size_bins, peer_order, baseline, shard_counts
    )
    controller_tail_placement = assign_bins_min_movement(
        CONTROLLER_TAIL_BINS,
        peer_order,
        baseline,
        shard_counts,
        fixed_first_peer=controller_peer,
    )
    placements = {
        "round_robin": baseline,
        "size_balanced": size_placement,
        "controller_aware_tail": controller_tail_placement,
    }
    metrics = {
        name: placement_metrics(
            name,
            placement,
            peer_order,
            controller_peer,
            shard_counts,
            trace,
            controller_overhead,
            baseline,
        )
        for name, placement in placements.items()
    }

    output.mkdir(parents=True)
    placement_path = output / "placements.json"
    write_json_new(
        placement_path,
        {
            "format_version": 1,
            "physical_machine_count": PHYSICAL_MACHINE_COUNT,
            "logical_shard_count": EXPECTED_SHARD_COUNT,
            "controller_peer_id": controller_peer,
            "peer_order": peer_order,
            "peer_hosts": peer_to_host,
            "placements": {
                name: {str(shard_id): peer_id for shard_id, peer_id in placement.items()}
                for name, placement in placements.items()
            },
        },
    )
    shard_rows = []
    for shard_id, count in enumerate(shard_counts):
        aggregate = trace["aggregate"]["per_shard"][shard_id]
        shard_rows.append(
            {
                "shard_id": shard_id,
                "physical_vector_copies": count,
                "query_visits": aggregate["query_visits"],
                "average_ef_when_visited": aggregate["average_ef_when_visited"],
                "average_work_per_query": average_work[shard_id],
                "round_robin_peer_id": baseline[shard_id],
                "size_balanced_peer_id": size_placement[shard_id],
                "controller_aware_tail_peer_id": controller_tail_placement[shard_id],
            }
        )
    shard_metrics_path = output / "shard-metrics.csv"
    write_csv_new(shard_metrics_path, list(shard_rows[0]), shard_rows)
    candidate_rows = []
    for name, row in metrics.items():
        candidate_rows.append(
            {
                "strategy": name,
                "average_node_load_peak": row["average_node_load_peak"],
                "per_query_node_peak_mean": row["per_query_node_peak_mean"],
                "per_query_node_peak_p95": row["per_query_node_peak_p95"],
                "per_query_node_peak_p99": row["per_query_node_peak_p99"],
                "moved_shard_count_from_baseline": row["moved_shard_count_from_baseline"],
                "moved_physical_vector_copies_from_baseline": row[
                    "moved_physical_vector_copies_from_baseline"
                ],
            }
        )
    candidate_metrics_path = output / "candidate-metrics.csv"
    write_csv_new(candidate_metrics_path, list(candidate_rows[0]), candidate_rows)
    metrics_path = output / "candidate-metrics.json"
    write_json_new(metrics_path, metrics)

    production_artifact = layout_dir / build_manifest["outputs"]["production_artifact"]
    baseline_metric = metrics["round_robin"]
    candidate_metric = metrics["controller_aware_tail"]
    manifest_path = output / "manifest.json"
    write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_PLAN_ONLY",
            "claim_boundary": (
                "The workload proxy ranks candidates but cannot establish QPS. "
                "Every placement must be measured online at Recall@10 >= 0.90."
            ),
            "experiment_contract": {
                "dataset": "GloVe-200-angular",
                "distance": "Cosine",
                "top_k": 10,
                "logical_shard_count": EXPECTED_SHARD_COUNT,
                "physical_machine_count": PHYSICAL_MACHINE_COUNT,
                "qdrant_cpu_cores_per_machine": 16,
                "qdrant_cpu_cores_total": 64,
                "artifact_generation": build_manifest["parameters"]["generation"],
                "upper_k": build_manifest["parameters"]["upper_k"],
                "upper_search_ef": build_manifest["parameters"]["upper_search_ef"],
                "dynamic_ef_base": build_manifest["parameters"]["dynamic_ef_base"],
                "dynamic_ef_factor": build_manifest["parameters"]["dynamic_ef_factor"],
            },
            "inputs": {
                "layout_build_manifest": str(build_manifest_path),
                "layout_build_manifest_sha256": sha256_path(build_manifest_path),
                "production_artifact": str(production_artifact),
                "production_artifact_sha256": sha256_path(production_artifact),
                "route_trace": str(args.trace),
                "route_trace_sha256": sha256_path(args.trace),
                "baseline_benchmark": str(args.baseline_benchmark),
                "baseline_benchmark_sha256": sha256_path(args.baseline_benchmark),
                "baseline_placement": str(args.baseline_placement),
                "baseline_placement_sha256": sha256_path(args.baseline_placement),
                "topology": str(args.topology),
                "topology_sha256": sha256_path(args.topology),
            },
            "live_peer_map": {
                "controller_peer_id": controller_peer,
                "peer_order_from_recorded_round_robin": peer_order,
                "peer_hosts": peer_to_host,
                "peer_uris": peer_uris,
            },
            "work_proxy": {
                "local_shard_work": "EF * log2(physical_vector_copies_in_shard)",
                "controller_overhead_model": overhead_model,
            },
            "predicted_controller_aware_tail_improvement_percent": {
                "average_node_load_peak": 100.0 * (
                    1.0
                    - candidate_metric["average_node_load_peak"]
                    / baseline_metric["average_node_load_peak"]
                ),
                "per_query_node_peak_mean": 100.0 * (
                    1.0
                    - candidate_metric["per_query_node_peak_mean"]
                    / baseline_metric["per_query_node_peak_mean"]
                ),
                "per_query_node_peak_p95": 100.0 * (
                    1.0
                    - candidate_metric["per_query_node_peak_p95"]
                    / baseline_metric["per_query_node_peak_p95"]
                ),
                "per_query_node_peak_p99": 100.0 * (
                    1.0
                    - candidate_metric["per_query_node_peak_p99"]
                    / baseline_metric["per_query_node_peak_p99"]
                ),
            },
            "outputs": {
                "placements": str(placement_path),
                "placements_sha256": sha256_path(placement_path),
                "shard_metrics": str(shard_metrics_path),
                "shard_metrics_sha256": sha256_path(shard_metrics_path),
                "candidate_metrics_csv": str(candidate_metrics_path),
                "candidate_metrics_csv_sha256": sha256_path(candidate_metrics_path),
                "candidate_metrics_json": str(metrics_path),
                "candidate_metrics_json_sha256": sha256_path(metrics_path),
            },
        },
    )
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--baseline-benchmark", type=Path, default=DEFAULT_BASELINE_BENCHMARK
    )
    parser.add_argument(
        "--baseline-placement", type=Path, default=DEFAULT_BASELINE_PLACEMENT
    )
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    plan(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
