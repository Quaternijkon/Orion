#!/usr/bin/env python3
"""Validate a fixed Orion peer-premerge A1/B/A2 sandwich.

The input root must contain exactly the six protocol leaves below::

    enabled-a1/orion-r090/{benchmark/,transport-probe.json}
    enabled-a1/orion-r095/{benchmark/,transport-probe.json}
    disabled-b/orion-r090/{benchmark/,transport-probe.json}
    disabled-b/orion-r095/{benchmark/,transport-probe.json}
    enabled-a2/orion-r090/{benchmark/,transport-probe.json}
    enabled-a2/orion-r095/{benchmark/,transport-probe.json}

Each benchmark is the formal 500-warmup/3,000-query protocol.  Each transport
probe is the fixed 500-warmup/200-query bit-exact proof.  The analyzer reuses
the chunk-sweep analyzer's strict single-leaf validation, then binds dataset,
query hashes, collection, artifact, commit, image, placement, request contract,
and benchmark parameters across the sandwich.  It also requires A1/A2 to use
enabled compact chunk 4, B to use disabled ordinary fan-out, exact probe IDs
and float32 score bytes, 12 compact or 46 ordinary status-0 RPCs, no legacy
RPCs, and at most 5% symmetric A1/A2 QPS drift.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

try:
    from tools import native_auto_shard_chunk_sweep as chunk
except ModuleNotFoundError:  # Direct execution from the tools directory.
    import native_auto_shard_chunk_sweep as chunk


ARM_SETTINGS = (
    ("enabled-a1", "A1", "enabled", "4"),
    ("disabled-b", "B", "disabled", "4"),
    ("enabled-a2", "A2", "enabled", "4"),
)
TARGET_SETTINGS = (
    ("r090", "orion-r090"),
    ("r095", "orion-r095"),
)
ARM_BY_LABEL = {label: name for name, label, _mode, _chunk in ARM_SETTINGS}
FORMAL_BENCHMARK_QUERY_COUNT = 3_000
A1_A2_QPS_DRIFT_LIMIT = 0.05
EXPECTED_ENABLED_COMPACT_BY_WORKER = (4, 4, 4)
EXPECTED_ENABLED_COMPACT_TOTAL = 12
EXPECTED_DISABLED_ORDINARY_BY_WORKER = chunk.EXPECTED_DISABLED_RPC_COUNTS
EXPECTED_DISABLED_ORDINARY_TOTAL = chunk.EXPECTED_SHARD_COUNT
OUTPUT_FILES = (
    "ab_summary.csv",
    "transport_equivalence.json",
    "drift_analysis.json",
    "run_manifest.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sandwich_root",
        help="Completed enabled-A1/disabled-B/enabled-A2 result root.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory to create for the four analysis artifacts.",
    )
    return parser.parse_args(argv)


@contextlib.contextmanager
def formal_benchmark_protocol() -> Iterator[None]:
    """Temporarily select the formal query count in the shared validator.

    The imported validator intentionally fixes the chunk sweep at 1,000
    queries.  Its checks read the module constant at call time, so a narrowly
    scoped override lets this analyzer reuse the same fail-closed manifest and
    stability-CSV validation without maintaining a divergent copy.
    """

    previous_query_count = chunk.BENCHMARK_QUERY_COUNT
    chunk.BENCHMARK_QUERY_COUNT = FORMAL_BENCHMARK_QUERY_COUNT
    try:
        yield
    finally:
        chunk.BENCHMARK_QUERY_COUNT = previous_query_count


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_contract(value: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(value[field] for field in fields)


def target_token_present(collection: str, target: str) -> bool:
    normalized = collection.lower().replace("-", "_")
    return f"_{target}_" in f"_{normalized}_"


def validate_leaf_binding(
    leaf_name: str,
    benchmark: dict[str, Any],
    probe: dict[str, Any],
) -> None:
    manifest = benchmark["manifest"]
    checks = (
        (benchmark["collection"], probe["collection"], "collection"),
        (benchmark["api"], probe["api"], "API"),
        (benchmark["top_k"], probe["top_k"], "top-k"),
        (benchmark["batch_size"], probe["batch_size"], "batch size"),
        (manifest["base_url"], probe["base_url"], "controller API"),
        (manifest["worker_urls"], probe["worker_urls"], "worker URLs"),
        (manifest["dataset"], probe["dataset"], "dataset"),
        (manifest["commit"], probe["deployment_commit"], "deployment commit"),
        (manifest["image_tag"], probe["image_tag"], "image tag"),
        (manifest["image_id"], probe["image_id"], "image ID"),
        (
            manifest["deployment_manifest_path"],
            probe["deployment_manifest_path"],
            "deployment manifest path",
        ),
        (
            manifest["controller_peer_id"],
            probe["controller_peer_id"],
            "controller peer ID",
        ),
    )
    for benchmark_value, probe_value, field in checks:
        if benchmark_value != probe_value:
            raise RuntimeError(f"{leaf_name} benchmark/probe {field} mismatch")


def validate_target_equivalence(
    target: str,
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference = arms[ARM_BY_LABEL["B"]]
    benchmark_fields = (
        "collection",
        "method",
        "api",
        "query_count",
        "top_k",
        "batch_size",
        "repeats",
        "run_labels",
        "recall_values",
        "recall_mean",
        "recall_stdev",
    )
    benchmark_manifest_fields = (
        "schema_version",
        "base_url",
        "dataset",
        "commit",
        "image_tag",
        "image_id",
        "deployment_manifest_path",
        "worker_urls",
        "controller_peer_id",
        "worker_peer_ids",
        "placement",
        "placement_by_worker_url",
        "artifact_sha256",
        "generation",
        "points_count",
    )
    probe_fields = (
        "schema_version",
        "run_id",
        "collection",
        "base_url",
        "worker_urls",
        "api",
        "dataset",
        "vector_distance",
        "vector_name",
        "query_dtype",
        "query_dimension",
        "query_offset",
        "query_count",
        "warmup_query_count",
        "top_k",
        "batch_size",
        "query_sha256",
        "warmup_query_sha256",
        "ids_sha256",
        "ids_scores_sha256",
        "row_lengths",
        "deployment_commit",
        "deployment_manifest_path",
        "image_tag",
        "image_id",
        "request_contract",
        "controller_peer_id",
        "cluster_contract",
        "placement_sha256",
    )
    benchmark_reference = exact_contract(reference["benchmark"], benchmark_fields)
    manifest_reference = exact_contract(
        reference["benchmark"]["manifest"], benchmark_manifest_fields
    )
    probe_reference = exact_contract(reference["probe"], probe_fields)

    for arm_name, value in arms.items():
        benchmark = value["benchmark"]
        manifest = benchmark["manifest"]
        probe = value["probe"]
        validate_leaf_binding(f"{arm_name}/{target}", benchmark, probe)
        if exact_contract(benchmark, benchmark_fields) != benchmark_reference:
            raise RuntimeError(
                f"{arm_name}/{target} benchmark contract or exact recall "
                "differs from disabled-b"
            )
        if exact_contract(manifest, benchmark_manifest_fields) != manifest_reference:
            raise RuntimeError(
                f"{arm_name}/{target} benchmark deployment/layout binding "
                "differs from disabled-b"
            )
        if exact_contract(probe, probe_fields) != probe_reference:
            raise RuntimeError(
                f"{arm_name}/{target} query/warmup/IDs/float32-score proof "
                "differs from disabled-b"
            )

    return {
        "collection": reference["benchmark"]["collection"],
        "artifact_sha256": reference["benchmark"]["manifest"]["artifact_sha256"],
        "generation": reference["benchmark"]["manifest"]["generation"],
        "recall_sequence": reference["benchmark"]["recall_values"],
        "query_sha256": reference["probe"]["query_sha256"],
        "warmup_query_sha256": reference["probe"]["warmup_query_sha256"],
        "ids_sha256": reference["probe"]["ids_sha256"],
        "ids_scores_sha256": reference["probe"]["ids_scores_sha256"],
    }


def validate_cross_target_binding(
    leaves: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    reference = leaves["r090"][ARM_BY_LABEL["A1"]]
    benchmark = reference["benchmark"]
    manifest = benchmark["manifest"]
    probe = reference["probe"]
    common_fields = {
        "dataset": manifest["dataset"],
        "commit": manifest["commit"],
        "image_tag": manifest["image_tag"],
        "image_id": manifest["image_id"],
        "deployment_manifest_path": manifest["deployment_manifest_path"],
        "base_url": manifest["base_url"],
        "worker_urls": manifest["worker_urls"],
        "controller_peer_id": manifest["controller_peer_id"],
        "worker_peer_ids": manifest["worker_peer_ids"],
        "placement": manifest["placement"],
        "placement_by_worker_url": manifest["placement_by_worker_url"],
        "benchmark_query_count": benchmark["query_count"],
        "benchmark_top_k": benchmark["top_k"],
        "benchmark_batch_size": benchmark["batch_size"],
        "stability_repeats": benchmark["repeats"],
        "run_labels": benchmark["run_labels"],
        "probe_query_count": probe["query_count"],
        "probe_warmup_query_count": probe["warmup_query_count"],
        "probe_top_k": probe["top_k"],
        "probe_batch_size": probe["batch_size"],
        "query_sha256": probe["query_sha256"],
        "warmup_query_sha256": probe["warmup_query_sha256"],
    }
    arm_manifest_sha256: dict[str, str] = {}
    collections: dict[str, str] = {}
    for target, target_arms in leaves.items():
        collections[target] = target_arms[ARM_BY_LABEL["A1"]]["benchmark"][
            "collection"
        ]
        if not target_token_present(collections[target], target):
            raise RuntimeError(
                f"{target} collection {collections[target]!r} does not identify "
                f"{target}"
            )
        for arm_name, value in target_arms.items():
            item_benchmark = value["benchmark"]
            item_manifest = item_benchmark["manifest"]
            item_probe = value["probe"]
            actual = {
                "dataset": item_manifest["dataset"],
                "commit": item_manifest["commit"],
                "image_tag": item_manifest["image_tag"],
                "image_id": item_manifest["image_id"],
                "deployment_manifest_path": item_manifest[
                    "deployment_manifest_path"
                ],
                "base_url": item_manifest["base_url"],
                "worker_urls": item_manifest["worker_urls"],
                "controller_peer_id": item_manifest["controller_peer_id"],
                "worker_peer_ids": item_manifest["worker_peer_ids"],
                "placement": item_manifest["placement"],
                "placement_by_worker_url": item_manifest[
                    "placement_by_worker_url"
                ],
                "benchmark_query_count": item_benchmark["query_count"],
                "benchmark_top_k": item_benchmark["top_k"],
                "benchmark_batch_size": item_benchmark["batch_size"],
                "stability_repeats": item_benchmark["repeats"],
                "run_labels": item_benchmark["run_labels"],
                "probe_query_count": item_probe["query_count"],
                "probe_warmup_query_count": item_probe["warmup_query_count"],
                "probe_top_k": item_probe["top_k"],
                "probe_batch_size": item_probe["batch_size"],
                "query_sha256": item_probe["query_sha256"],
                "warmup_query_sha256": item_probe["warmup_query_sha256"],
            }
            if actual != common_fields:
                raise RuntimeError(
                    f"{arm_name}/{target} common dataset/query/deployment contract "
                    "differs from enabled-a1/r090"
                )
            manifest_sha = item_probe["deployment_manifest_sha256"]
            previous = arm_manifest_sha256.setdefault(arm_name, manifest_sha)
            if previous != manifest_sha:
                raise RuntimeError(
                    f"{arm_name} deployment manifest SHA differs between targets"
                )
    if len(set(collections.values())) != len(collections):
        raise RuntimeError("r090 and r095 must use distinct collections")
    return {
        **common_fields,
        "collections": collections,
        "deployment_manifest_sha256_by_arm": arm_manifest_sha256,
    }


def validate_rpc_contract(
    target: str,
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    proof: dict[str, Any] = {}
    for arm_name, arm_label, mode, chunk_value in ARM_SETTINGS:
        probe = arms[arm_name]["probe"]
        telemetry = probe["telemetry"]
        worker_urls = tuple(probe["worker_urls"])
        ordinary_by_worker = tuple(
            telemetry[worker][chunk.CORE_SEARCH_BATCH] for worker in worker_urls
        )
        legacy_by_worker = tuple(
            telemetry[worker][chunk.LEGACY_BY_SHARD] for worker in worker_urls
        )
        compact_by_worker = tuple(
            telemetry[worker][chunk.COMPACT_BY_SHARD] for worker in worker_urls
        )
        if mode == "enabled":
            chunk.require_exact(
                compact_by_worker,
                EXPECTED_ENABLED_COMPACT_BY_WORKER,
                f"{arm_name}/{target} compact RPC counts",
            )
            chunk.require_exact(
                sum(compact_by_worker),
                EXPECTED_ENABLED_COMPACT_TOTAL,
                f"{arm_name}/{target} compact RPC total",
            )
            chunk.require_exact(
                ordinary_by_worker,
                (0, 0, 0),
                f"{arm_name}/{target} ordinary RPC counts",
            )
        else:
            chunk.require_exact(
                ordinary_by_worker,
                EXPECTED_DISABLED_ORDINARY_BY_WORKER,
                f"{arm_name}/{target} ordinary RPC counts",
            )
            chunk.require_exact(
                sum(ordinary_by_worker),
                EXPECTED_DISABLED_ORDINARY_TOTAL,
                f"{arm_name}/{target} ordinary RPC total",
            )
            chunk.require_exact(
                compact_by_worker,
                (0, 0, 0),
                f"{arm_name}/{target} compact RPC counts",
            )
        chunk.require_exact(
            legacy_by_worker,
            (0, 0, 0),
            f"{arm_name}/{target} legacy RPC counts",
        )
        active_by_worker = (
            compact_by_worker if mode == "enabled" else ordinary_by_worker
        )
        proof[arm_label] = {
            "arm": arm_name,
            "mode": mode,
            "chunk": chunk_value,
            "ordinary_by_worker": list(ordinary_by_worker),
            "ordinary_total": sum(ordinary_by_worker),
            "legacy_by_worker": list(legacy_by_worker),
            "legacy_total": sum(legacy_by_worker),
            "compact_by_worker": list(compact_by_worker),
            "compact_total": sum(compact_by_worker),
            "active_by_worker": list(active_by_worker),
            "active_total": sum(active_by_worker),
            "status_zero_only": True,
        }
    return proof


def analyze_drift(
    target: str,
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    a1 = arms[ARM_BY_LABEL["A1"]]["benchmark"]["qps_mean"]
    b = arms[ARM_BY_LABEL["B"]]["benchmark"]["qps_mean"]
    a2 = arms[ARM_BY_LABEL["A2"]]["benchmark"]["qps_mean"]
    enabled_mean = (a1 + a2) / 2.0
    symmetric_drift = abs(a2 - a1) / enabled_mean
    if symmetric_drift > A1_A2_QPS_DRIFT_LIMIT:
        raise RuntimeError(
            f"{target} A1/A2 QPS drift {symmetric_drift:.6%} exceeds "
            f"the {A1_A2_QPS_DRIFT_LIMIT:.0%} limit"
        )
    return {
        "target": target,
        "drift_definition": "abs(A2-A1) / mean(A1,A2)",
        "drift_limit": A1_A2_QPS_DRIFT_LIMIT,
        "a1_qps": a1,
        "b_qps": b,
        "a2_qps": a2,
        "enabled_qps_mean": enabled_mean,
        "a2_delta_pct_vs_a1": (a2 / a1 - 1.0) * 100.0,
        "a1_a2_symmetric_drift": symmetric_drift,
        "a1_a2_symmetric_drift_pct": symmetric_drift * 100.0,
        "within_limit": True,
        "b_qps_relative_to_enabled_mean": b / enabled_mean,
        "b_qps_delta_pct_vs_enabled_mean": (b / enabled_mean - 1.0) * 100.0,
    }


def build_transport_equivalence(
    root: Path,
    leaves: dict[str, dict[str, dict[str, Any]]],
    target_proofs: dict[str, dict[str, Any]],
    rpc_proofs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for target, arms in leaves.items():
        reference = arms[ARM_BY_LABEL["B"]]
        arm_values: dict[str, Any] = {}
        for arm_name, arm_label, mode, chunk_value in ARM_SETTINGS:
            value = arms[arm_name]
            benchmark = value["benchmark"]
            probe = value["probe"]
            arm_values[arm_label] = {
                "arm": arm_name,
                "mode": mode,
                "chunk": chunk_value,
                "query_equal": probe["query_sha256"]
                == reference["probe"]["query_sha256"],
                "warmup_equal": probe["warmup_query_sha256"]
                == reference["probe"]["warmup_query_sha256"],
                "ids_equal": probe["ids_sha256"]
                == reference["probe"]["ids_sha256"],
                "ids_scores_equal": probe["ids_scores_sha256"]
                == reference["probe"]["ids_scores_sha256"],
                "benchmark_recall_equal": benchmark["recall_values"]
                == reference["benchmark"]["recall_values"],
                "status_zero_only": True,
                "benchmark_manifest_path": benchmark["manifest"]["manifest_path"],
                "benchmark_summary_path": benchmark["summary_path"],
                "transport_probe_path": probe["probe_path"],
                "deployment_manifest_sha256": probe[
                    "deployment_manifest_sha256"
                ],
                "method_totals": probe["method_totals"],
            }
        targets[target] = {
            "reference_arm": "B",
            "equivalent": True,
            **target_proofs[target],
            "rpc_contract": rpc_proofs[target],
            "arms": arm_values,
        }
    return {
        "schema_version": 1,
        "sandwich_root": str(root),
        "equivalent": True,
        "sequence": [label for _name, label, _mode, _chunk in ARM_SETTINGS],
        "targets": targets,
    }


def summary_rows(
    leaves: dict[str, dict[str, dict[str, Any]]],
    drift: dict[str, dict[str, Any]],
    rpc_proofs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target, _directory in TARGET_SETTINGS:
        target_drift = drift[target]
        for sequence_position, (arm_name, arm_label, mode, chunk_value) in enumerate(
            ARM_SETTINGS, start=1
        ):
            value = leaves[target][arm_name]
            benchmark = value["benchmark"]
            manifest = benchmark["manifest"]
            probe = value["probe"]
            rpc = rpc_proofs[target][arm_label]
            rows.append(
                {
                    "target": target,
                    "sequence_position": sequence_position,
                    "arm": arm_label,
                    "arm_directory": arm_name,
                    "peer_premerge_mode": mode,
                    "shards_per_rpc": chunk_value,
                    "collection": benchmark["collection"],
                    "benchmark_query_count": benchmark["query_count"],
                    "benchmark_warmup_query_count": chunk.BENCHMARK_WARMUP_QUERY_COUNT,
                    "benchmark_top_k": benchmark["top_k"],
                    "benchmark_batch_size": benchmark["batch_size"],
                    "stability_repeats": benchmark["repeats"],
                    "recall_at_k": benchmark["recall_mean"],
                    "qps_mean": benchmark["qps_mean"],
                    "qps_stdev": benchmark["qps_stdev"],
                    "qps_cv": benchmark["qps_cv"],
                    "latency_p95_ms": benchmark["latency_p95_ms"],
                    "latency_p99_ms": benchmark["latency_p99_ms"],
                    "probe_query_count": probe["query_count"],
                    "probe_warmup_query_count": probe["warmup_query_count"],
                    "query_sha256": probe["query_sha256"],
                    "warmup_query_sha256": probe["warmup_query_sha256"],
                    "ids_sha256": probe["ids_sha256"],
                    "ids_scores_sha256": probe["ids_scores_sha256"],
                    "ordinary_rpc_count": rpc["ordinary_total"],
                    "legacy_rpc_count": rpc["legacy_total"],
                    "compact_rpc_count": rpc["compact_total"],
                    "active_rpc_count": rpc["active_total"],
                    "deployment_commit": manifest["commit"],
                    "image_tag": manifest["image_tag"],
                    "image_id": manifest["image_id"],
                    "placement_sha256": probe["placement_sha256"],
                    "deployment_manifest_sha256": probe[
                        "deployment_manifest_sha256"
                    ],
                    "a1_a2_symmetric_drift_pct": target_drift[
                        "a1_a2_symmetric_drift_pct"
                    ],
                    "enabled_qps_mean": target_drift["enabled_qps_mean"],
                    "b_qps_delta_pct_vs_enabled_mean": target_drift[
                        "b_qps_delta_pct_vs_enabled_mean"
                    ],
                    "transport_equivalent": True,
                }
            )
    return rows


def input_manifest(
    root: Path,
    leaves: dict[str, dict[str, dict[str, Any]]],
    common_binding: dict[str, Any],
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for arm_name, arm_label, mode, chunk_value in ARM_SETTINGS:
        targets: dict[str, Any] = {}
        for target, _directory in TARGET_SETTINGS:
            leaf = leaves[target][arm_name]
            paths = {
                "benchmark_summary": Path(leaf["benchmark"]["summary_path"]),
                "benchmark_stability": Path(leaf["benchmark"]["stability_path"]),
                "benchmark_manifest": Path(
                    leaf["benchmark"]["manifest"]["manifest_path"]
                ),
                "transport_probe": Path(leaf["probe"]["probe_path"]),
            }
            targets[target] = {
                key: {"path": str(path), "sha256": file_sha256(path)}
                for key, path in paths.items()
            }
        inputs[arm_label] = {
            "directory": arm_name,
            "mode": mode,
            "chunk": chunk_value,
            "targets": targets,
        }
    return {
        "schema_version": 1,
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "sandwich_root": str(root),
        "sequence": [label for _name, label, _mode, _chunk in ARM_SETTINGS],
        "protocol": {
            "benchmark_warmup_query_count": chunk.BENCHMARK_WARMUP_QUERY_COUNT,
            "benchmark_query_count": FORMAL_BENCHMARK_QUERY_COUNT,
            "probe_warmup_query_count": chunk.PROBE_WARMUP_QUERY_COUNT,
            "probe_query_count": chunk.PROBE_QUERY_COUNT,
            "top_k": chunk.TOP_K,
            "batch_size": chunk.BATCH_SIZE,
            "stability_repeats": chunk.STABILITY_REPEATS,
            "a1_a2_qps_drift_limit": A1_A2_QPS_DRIFT_LIMIT,
            "enabled_compact_rpc_total": EXPECTED_ENABLED_COMPACT_TOTAL,
            "disabled_ordinary_rpc_total": EXPECTED_DISABLED_ORDINARY_TOTAL,
            "legacy_rpc_total": 0,
        },
        "binding": common_binding,
        "inputs": inputs,
        "outputs": list(OUTPUT_FILES),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(sandwich_root: str | Path, output_dir: str | Path) -> Path:
    root = Path(sandwich_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"sandwich root not found: {root}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")

    leaves: dict[str, dict[str, dict[str, Any]]] = {
        target: {} for target, _directory in TARGET_SETTINGS
    }
    with formal_benchmark_protocol():
        for arm_name, _arm_label, mode, chunk_value in ARM_SETTINGS:
            arm_root = root / arm_name
            if not arm_root.is_dir():
                raise FileNotFoundError(f"required sandwich arm not found: {arm_root}")
            for target, target_directory in TARGET_SETTINGS:
                leaf_root = arm_root / target_directory
                if not leaf_root.is_dir():
                    raise FileNotFoundError(
                        f"required sandwich target not found: {leaf_root}"
                    )
                context = f"{arm_name}/{target}"
                leaves[target][arm_name] = {
                    "benchmark": chunk.validate_benchmark(
                        context, mode, chunk_value, leaf_root
                    ),
                    "probe": chunk.validate_probe(
                        context, mode, chunk_value, leaf_root
                    ),
                }

    target_proofs = {
        target: validate_target_equivalence(target, leaves[target])
        for target, _directory in TARGET_SETTINGS
    }
    common_binding = validate_cross_target_binding(leaves)
    rpc_proofs = {
        target: validate_rpc_contract(target, leaves[target])
        for target, _directory in TARGET_SETTINGS
    }
    drift = {
        target: analyze_drift(target, leaves[target])
        for target, _directory in TARGET_SETTINGS
    }
    transport = build_transport_equivalence(
        root, leaves, target_proofs, rpc_proofs
    )
    rows = summary_rows(leaves, drift, rpc_proofs)
    drift_output = {
        "schema_version": 1,
        "drift_definition": "abs(A2-A1) / mean(A1,A2)",
        "drift_limit": A1_A2_QPS_DRIFT_LIMIT,
        "comparison": "B QPS / mean(A1 QPS, A2 QPS)",
        "all_targets_within_limit": True,
        "targets": drift,
    }
    run_manifest = input_manifest(root, leaves, common_binding)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    write_csv(output / "ab_summary.csv", rows)
    write_json(output / "transport_equivalence.json", transport)
    write_json(output / "drift_analysis.json", drift_output)
    write_json(output / "run_manifest.json", run_manifest)
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = analyze(args.sandwich_root, args.output_dir)
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
