#!/usr/bin/env python3
"""Read-only, lock-scoped, interleaved A/B for two prepared Orion layouts.

This runner deliberately does not prepare, activate, move, create, delete, or
otherwise mutate collections or container resources.  Both Orion collections
must already exist and must be checksum-bound to their layout bundle and
``native_auto_shard_prepare`` manifest.

The formal protocol is fixed to GloVe/Cosine, Recall@10 >= 0.90, 1,000 tuning
queries, the disjoint 9,000-query held-out split, batch size 200, and five
paired 20-second repetitions.  If either arm's five-run QPS CV exceeds 5%, the
pair count is extended directly to seven.  Odd pairs execute A then B; even
pairs execute B then A.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import native_auto_shard_benchmark as native
from tools import native_auto_shard_benchmark_lock as benchmark_lock


VIRTUAL_SCALE_TOOL = Path(__file__).resolve().with_name("c1_orion_virtual_scale.py")
TUNING_QUERY_COUNT = 1_000
HELDOUT_QUERY_COUNT = 9_000
TOTAL_QUERY_COUNT = TUNING_QUERY_COUNT + HELDOUT_QUERY_COUNT
TOP_K = 10
BATCH_SIZE = 200
TARGET_RECALL = 0.90
MEASURE_SECONDS = 20.0
MIN_PAIRED_ROUNDS = 5
MAX_PAIRED_ROUNDS = 7
MAX_QPS_CV = 0.05
SATURATION_FRACTION = 0.98
ARM_LABELS = ("A", "B")
HISTORICAL_BUDGET = {
    "upper_k": 48,
    "upper_search_ef": 48,
    "dynamic_ef_base": 50,
    "dynamic_ef_factor": 14,
}
TUNING_RECALL_UPPER_EXCLUSIVE = 0.93
ADJUSTMENT_BEFORE_GENERATION = 3_248_141
ADJUSTMENT_BEFORE_ARTIFACT_SHA256 = (
    "8936c14a242af48ccb675825f5a193b3307585a2a588087622bac3276577708e"
)
ADJUSTMENT_BEFORE_BUILD_MANIFEST_SHA256 = (
    "bbaf6913dbb6720777cb35e5d758ed68456153a8781407e2734f735c32cd6641"
)
ADJUSTMENT_BEFORE_INITIAL_SHARDS = 24
FORMAL_LOGICAL_SHARDS = 32
FORMAL_PHYSICAL_HOSTS = 4
FORMAL_SERVER_CPUS_PER_HOST = 16.0
FORMAL_SERVER_CPUS_TOTAL = 64.0
FORBIDDEN_MUTATIONS = (
    "collection create/delete/update",
    "routing-artifact install/activation",
    "numeric-shard movement",
    "container CPU/cpuset update",
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


orion_scale = load_module(VIRTUAL_SCALE_TOOL, "orion_vscale_for_balance_ab")
hashall = orion_scale.hashall
simple = orion_scale.simple


@dataclass(frozen=True)
class ArmSpec:
    label: str
    name: str
    collection: str
    artifact: Path
    layout_dir: Path
    prepare_manifest: Path
    allow_balance_layout: bool
    allow_scaling_layout: bool
    allow_l1_partition_layout: bool
    allow_historical_prepare_deployment: bool
    allow_single_assignment_layout: bool = False


@dataclass(frozen=True)
class ProtocolDataset:
    path: Path
    dataset_sha256: str
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    neighbors_shape: tuple[int, int]
    tuning_queries: np.ndarray
    tuning_neighbors: np.ndarray
    heldout_queries: np.ndarray
    heldout_neighbors: np.ndarray
    tuning_query_sha256: str
    heldout_query_sha256: str
    tuning_neighbors_sha256: str
    heldout_neighbors_sha256: str


@dataclass(frozen=True)
class InterleavedABContract:
    """Comparison-specific hooks for the shared read-only A/B engine.

    The historical Orion runner remains the default contract.  A separate
    N-versus-C runner can reuse the measurement/statistics path while supplying
    stricter L1-owner fairness and identity validation without pretending that
    arm A is the historical P24->P32 baseline.
    """

    arm_specs_factory: Callable[[argparse.Namespace], Mapping[str, ArmSpec]]
    fairness_validator: Callable[
        [Mapping[str, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]],
        dict[str, Any],
    ]
    comparison_validator: Callable[
        [
            Mapping[str, ArmSpec],
            Mapping[str, Mapping[str, Any]],
            Mapping[str, Mapping[str, Any]],
        ],
        dict[str, Any],
    ]
    comparison_manifest_key: str
    comparison_completion_key: str
    fairness_budget_check_key: str
    fairness_budget_completion_key: str
    summary_record_type: str
    run_record_type: str
    budget_mode: str
    budget_subject: str
    adoption_pass_decision: str
    adoption_fail_decision: str
    adoption_scope: str
    decision_gate_key: str
    semantic_ratio_key: str
    completion_error_subject: str
    adoption_prerequisite_validator: Callable[
        [
            argparse.Namespace,
            Mapping[str, Mapping[str, Any]],
            Mapping[str, Any],
        ],
        dict[str, Any],
    ] | None = None
    prepare_binding_validator: Callable[..., dict[str, Any]] | None = None
    multi_assignment_retained_check_key: str | None = (
        "multi_assignment_enabled_in_both"
    )


def utc_timestamp() -> str:
    return hashall.utc_timestamp()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_int_csv(value: str) -> list[int]:
    result = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not result or any(candidate <= 0 for candidate in result):
        raise ValueError("concurrency candidates must be positive integers")
    if result != sorted(set(result)):
        raise ValueError("concurrency candidates must be sorted and unique")
    return result


def normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def paired_order(pair_number: int) -> tuple[str, str]:
    if pair_number <= 0:
        raise ValueError("pair number must be positive")
    return ARM_LABELS if pair_number % 2 else tuple(reversed(ARM_LABELS))


def qps_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        raise ValueError("QPS statistics require at least one value")
    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
        raise ValueError(f"QPS values must be finite and positive: {numeric}")
    mean = statistics.fmean(numeric)
    stdev = statistics.stdev(numeric) if len(numeric) > 1 else None
    return {
        "count": len(numeric),
        "mean": mean,
        "stdev": stdev,
        "cv": (stdev / mean) if stdev is not None else None,
        "min": min(numeric),
        "max": max(numeric),
    }


T_975_BY_DF = {
    4: 2.7764451051977987,
    5: 2.570581835636305,
    6: 2.4469118511449692,
}


def paired_ratio_confidence_interval(
    ratios: Sequence[float],
) -> dict[str, float | int | str]:
    """Return a two-sided 95% paired t interval on log(B/A)."""
    numeric = [float(value) for value in ratios]
    if len(numeric) not in {MIN_PAIRED_ROUNDS, MAX_PAIRED_ROUNDS}:
        raise ValueError("paired ratio CI requires exactly five or seven pairs")
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
        raise ValueError("paired ratios must be finite and positive")
    logs = [math.log(value) for value in numeric]
    mean_log = statistics.fmean(logs)
    stdev_log = statistics.stdev(logs)
    critical = T_975_BY_DF[len(logs) - 1]
    half_width = critical * stdev_log / math.sqrt(len(logs))
    return {
        "method": "two_sided_95pct_paired_t_interval_on_log_ratio",
        "confidence_level": 0.95,
        "pair_count": len(logs),
        "geometric_mean": math.exp(mean_log),
        "lower": math.exp(mean_log - half_width),
        "upper": math.exp(mean_log + half_width),
        "log_ratio_stdev": stdev_log,
        "t_critical": critical,
    }


def adoption_gate_passes(
    *,
    stable: bool,
    recall_failures: Mapping[str, Any],
    ratio_ci_lower: float,
    adoption_prerequisites: Mapping[str, Any] | None,
) -> bool:
    """Require comparison-specific prerequisites in every positive decision."""

    return bool(
        stable
        and not recall_failures
        and float(ratio_ci_lower) > 1.0
        and (
            adoption_prerequisites is None
            or adoption_prerequisites.get("status") == "PASS"
        )
    )


def select_independent_saturation(
    sweep_pairs: Sequence[Mapping[str, Any]],
    *,
    saturation_fraction: float = SATURATION_FRACTION,
) -> dict[str, Any]:
    """Select each arm's knee from one symmetric, counterbalanced sweep."""
    if not 0.0 < saturation_fraction <= 1.0:
        raise ValueError("saturation fraction must be in (0, 1]")
    if not sweep_pairs:
        raise ValueError("empty paired concurrency sweep")
    candidates = [int(row["concurrency"]) for row in sweep_pairs]
    if candidates != sorted(set(candidates)):
        raise ValueError("paired sweep concurrency values must be sorted and unique")
    by_arm: dict[str, list[dict[str, Any]]] = {label: [] for label in ARM_LABELS}
    for row in sweep_pairs:
        for label in ARM_LABELS:
            result = row.get(label)
            if not isinstance(result, Mapping):
                raise ValueError(f"sweep row lacks arm {label}: {row}")
            by_arm[label].append(
                {
                    "concurrency": int(row["concurrency"]),
                    "qps": float(result["qps"]),
                }
            )
    independent = {
        label: hashall.select_saturation(rows) for label, rows in by_arm.items()
    }
    best = {
        label: max(float(row["qps"]) for row in rows)
        for label, rows in by_arm.items()
    }
    common_eligible = [
        int(row["concurrency"])
        for row in sweep_pairs
        if all(
            float(row[label]["qps"]) >= best[label] * saturation_fraction
            for label in ARM_LABELS
        )
    ]
    maximum = max(candidates)
    knee_observed = {
        label: bool(independent[label]["knee_observed"]) for label in ARM_LABELS
    }
    return {
        "selection_rule": (
            "for each arm independently, minimum concurrency within 98% of that "
            "arm's best QPS on the same counterbalanced candidate sweep"
        ),
        "saturation_fraction": saturation_fraction,
        "independent_arm_selections": independent,
        "best_qps_by_arm": best,
        "common_eligible_concurrencies": common_eligible,
        "selected_concurrency_by_arm": {
            label: int(independent[label]["selected_concurrency"])
            for label in ARM_LABELS
        },
        "max_candidate": maximum,
        "knee_observed_by_arm": knee_observed,
        "all_knees_observed": all(knee_observed.values()),
    }


def run_ordered_pair(
    pair_number: int,
    runner: Callable[[str, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    order = paired_order(pair_number)
    results: dict[str, dict[str, Any]] = {}
    for position, label in enumerate(order, start=1):
        result = dict(runner(label, pair_number, position))
        qps = float(result.get("qps") or 0.0)
        if not math.isfinite(qps) or qps <= 0.0:
            raise RuntimeError(f"pair {pair_number} arm {label} returned invalid QPS")
        result["execution_position"] = position
        results[label] = result
    return {
        "pair_number": pair_number,
        "execution_order": list(order),
        "A": results["A"],
        "B": results["B"],
        "paired_ratio_b_over_a": float(results["B"]["qps"])
        / float(results["A"]["qps"]),
    }


def annotate_running_statistics(
    pairs: Sequence[dict[str, Any]],
    recall_by_arm: Mapping[str, float],
) -> None:
    qps_by_arm: dict[str, list[float]] = {label: [] for label in ARM_LABELS}
    ratios: list[float] = []
    for pair in pairs:
        ratios.append(float(pair["paired_ratio_b_over_a"]))
        for label in ARM_LABELS:
            qps_by_arm[label].append(float(pair[label]["qps"]))
            pair[label]["heldout_recall_at_10"] = float(recall_by_arm[label])
            pair[label]["recall_source"] = "single_disjoint_9000_query_probe"
            pair[label]["running_qps"] = qps_statistics(qps_by_arm[label])
        pair["running_paired_ratio_b_over_a"] = qps_statistics(ratios)


def run_formal_pairs(
    runner: Callable[[str, int, int], Mapping[str, Any]],
    recall_by_arm: Mapping[str, float],
    *,
    min_pairs: int = MIN_PAIRED_ROUNDS,
    max_pairs: int = MAX_PAIRED_ROUNDS,
    max_cv: float = MAX_QPS_CV,
) -> dict[str, Any]:
    if min_pairs != 5 or max_pairs != 7:
        raise ValueError("the formal Orion balance A/B protocol is fixed to 5 -> 7 pairs")
    pairs = [run_ordered_pair(number, runner) for number in range(1, min_pairs + 1)]
    annotate_running_statistics(pairs, recall_by_arm)
    five_pair_stats = {
        label: qps_statistics([float(pair[label]["qps"]) for pair in pairs])
        for label in ARM_LABELS
    }
    extended_to_seven = any(
        float(five_pair_stats[label]["cv"] or 0.0) > max_cv
        for label in ARM_LABELS
    )
    if extended_to_seven:
        pairs.extend(
            run_ordered_pair(number, runner)
            for number in range(min_pairs + 1, max_pairs + 1)
        )
        annotate_running_statistics(pairs, recall_by_arm)
    final_stats = {
        label: qps_statistics([float(pair[label]["qps"]) for pair in pairs])
        for label in ARM_LABELS
    }
    stable = all(float(final_stats[label]["cv"] or 0.0) <= max_cv for label in ARM_LABELS)
    return {
        "pairs": pairs,
        "five_pair_stats": five_pair_stats,
        "extended_to_seven": extended_to_seven,
        "final_stats": final_stats,
        "stable": stable,
        "max_cv": max_cv,
    }


def paired_round_csv_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        row: dict[str, Any] = {
            "pair_number": int(pair["pair_number"]),
            "execution_order": "->".join(str(value) for value in pair["execution_order"]),
            "a_qps": float(pair["A"]["qps"]),
            "a_recall_at_10": float(pair["A"]["heldout_recall_at_10"]),
            "a_running_qps_mean": float(pair["A"]["running_qps"]["mean"]),
            "a_running_qps_cv": pair["A"]["running_qps"]["cv"],
            "b_qps": float(pair["B"]["qps"]),
            "b_recall_at_10": float(pair["B"]["heldout_recall_at_10"]),
            "b_running_qps_mean": float(pair["B"]["running_qps"]["mean"]),
            "b_running_qps_cv": pair["B"]["running_qps"]["cv"],
            "paired_ratio_b_over_a": float(pair["paired_ratio_b_over_a"]),
            "running_paired_ratio_mean": float(
                pair["running_paired_ratio_b_over_a"]["mean"]
            ),
            "running_paired_ratio_cv": pair["running_paired_ratio_b_over_a"]["cv"],
        }
        for label, prefix in (("A", "a"), ("B", "b")):
            result = pair[label]
            row[f"{prefix}_concurrency"] = result.get("concurrency")
            row[f"{prefix}_wall_s"] = result.get("wall_s")
            row[f"{prefix}_query_count"] = result.get("query_count")
            row[f"{prefix}_cpu_average_cores_total"] = result.get(
                "cpu_average_cores_total"
            )
        rows.append(row)
    return rows


def concurrency_sweep_csv_rows(
    sweep_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in sweep_pairs:
        rows.append(
            {
                "candidate_number": int(pair["candidate_number"]),
                "concurrency": int(pair["concurrency"]),
                "execution_order": "->".join(pair["execution_order"]),
                "a_qps": float(pair["A"]["qps"]),
                "a_wall_s": pair["A"].get("wall_s"),
                "b_qps": float(pair["B"]["qps"]),
                "b_wall_s": pair["B"].get("wall_s"),
                "paired_ratio_b_over_a": float(pair["B"]["qps"])
                / float(pair["A"]["qps"]),
            }
        )
    return rows


def validate_prepare_binding(
    arm: ArmSpec,
    *,
    base_url: str,
    deployment_manifest: Path,
    topology: Path,
) -> dict[str, Any]:
    """Bind one declared arm to immutable prepare/layout/artifact files."""
    artifact = arm.artifact.expanduser().resolve(strict=True)
    layout_dir = arm.layout_dir.expanduser().resolve(strict=True)
    prepare_path = arm.prepare_manifest.expanduser().resolve(strict=True)
    deployment_path = deployment_manifest.expanduser().resolve(strict=True)
    topology_path = topology.expanduser().resolve(strict=True)
    if not layout_dir.is_dir():
        raise FileNotFoundError(f"arm {arm.label} layout is not a directory: {layout_dir}")
    if artifact.parent != layout_dir:
        raise RuntimeError(f"arm {arm.label} artifact is not inside its declared layout")
    prepare = load_json_object(prepare_path)
    build_manifest_path = layout_dir / "build-manifest.json"
    build_manifest = load_json_object(build_manifest_path)
    parameters = build_manifest.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise RuntimeError(f"arm {arm.label} build parameters are missing")
    use_multi_assign = parameters.get("use_multi_assign")
    single_assignment_explicitly_allowed = (
        arm.allow_single_assignment_layout and use_multi_assign is False
    )
    if use_multi_assign is not True and not single_assignment_explicitly_allowed:
        raise RuntimeError(
            f"arm {arm.label} disables multi-assignment; this A/B only accepts "
            "load balancing orthogonal to Orion multi-assignment"
        )
    outputs = build_manifest.get("outputs") or {}
    if outputs.get("production_artifact") != artifact.name:
        raise RuntimeError(f"arm {arm.label} build manifest selects another artifact")
    if prepare.get("method") != "orion":
        raise RuntimeError(f"arm {arm.label} prepare manifest is not Orion")
    if prepare.get("collection") != arm.collection:
        raise RuntimeError(f"arm {arm.label} prepare/collection mismatch")
    if normalize_base_url(str(prepare.get("base_url") or "")) != normalize_base_url(
        base_url
    ):
        raise RuntimeError(f"arm {arm.label} prepare/base-url mismatch")
    deployment = prepare.get("deployment_manifest") or {}
    recorded_deployment_path = Path(
        str(deployment.get("path") or "")
    ).expanduser().resolve()
    same_deployment_path = recorded_deployment_path == deployment_path
    deployment_sha256 = sha256_path(deployment_path)
    recorded_deployment_sha256 = str(deployment.get("sha256") or "").lower()
    recorded_deployment_exists = recorded_deployment_path.is_file()
    recorded_deployment_observed_sha256 = (
        sha256_path(recorded_deployment_path) if recorded_deployment_exists else None
    )
    recorded_deployment_digest_reverified = (
        recorded_deployment_observed_sha256 == recorded_deployment_sha256
    )
    current_deployment_match = (
        same_deployment_path and recorded_deployment_sha256 == deployment_sha256
    )
    if not current_deployment_match and not arm.allow_historical_prepare_deployment:
        raise RuntimeError(
            f"arm {arm.label} prepare/deployment binding is historical; pass the "
            "explicit historical-prepare flag only for the frozen adjustment-before "
            "baseline"
        )
    prepare_topology = prepare.get("topology") or {}
    recorded_topology_path = Path(
        str(prepare_topology.get("path") or "")
    ).expanduser().resolve()
    recorded_topology_sha256 = str(prepare_topology.get("sha256") or "").lower()
    if not recorded_topology_path.is_file():
        raise FileNotFoundError(
            f"arm {arm.label} recorded prepare topology is unavailable: "
            f"{recorded_topology_path}"
        )
    if sha256_path(recorded_topology_path) != recorded_topology_sha256:
        raise RuntimeError(f"arm {arm.label} recorded prepare topology changed")
    topology_sha256 = sha256_path(topology_path)
    current_topology_match = (
        recorded_topology_path == topology_path
        and recorded_topology_sha256 == topology_sha256
    )
    if not current_topology_match and not arm.allow_historical_prepare_deployment:
        raise RuntimeError(
            f"arm {arm.label} prepare/topology binding is historical; use the "
            "explicit historical prepare provenance flag only for a verified clone"
        )
    artifact_sha256 = sha256_path(artifact)
    build_manifest_sha256 = sha256_path(build_manifest_path)
    checksums = prepare.get("checksums") or {}
    expected_checksums = {
        "routing_artifact_sha256": artifact_sha256,
        "layout_build_manifest_sha256": build_manifest_sha256,
        "topology_sha256": recorded_topology_sha256,
        "deployment_manifest_sha256": recorded_deployment_sha256,
    }
    for key, expected in expected_checksums.items():
        if checksums.get(key) != expected:
            raise RuntimeError(
                f"arm {arm.label} prepare {key} mismatch: "
                f"{checksums.get(key)!r} != {expected!r}"
            )
    prepare_layout = prepare.get("layout") or {}
    path_checks = {
        "layout_dir": layout_dir,
        "build_manifest_path": build_manifest_path,
        "artifact_path": artifact,
    }
    for key, expected in path_checks.items():
        actual = Path(str(prepare_layout.get(key) or "")).expanduser().resolve()
        if actual != expected:
            raise RuntimeError(
                f"arm {arm.label} prepare layout {key} mismatch: {actual} != {expected}"
            )
    if prepare_layout.get("artifact_sha256") != artifact_sha256:
        raise RuntimeError(f"arm {arm.label} prepare layout artifact checksum mismatch")
    if prepare_layout.get("build_manifest_sha256") != build_manifest_sha256:
        raise RuntimeError(f"arm {arm.label} prepare layout manifest checksum mismatch")
    provenance_metadata = prepare.get("provenance_metadata")
    if not isinstance(provenance_metadata, dict):
        raise RuntimeError(f"arm {arm.label} prepare provenance metadata is missing")
    return {
        "status": "PASS",
        "label": arm.label,
        "name": arm.name,
        "collection": arm.collection,
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_sha256,
        "layout_dir": str(layout_dir),
        "build_manifest_path": str(build_manifest_path),
        "build_manifest_sha256": build_manifest_sha256,
        "prepare_manifest_path": str(prepare_path),
        "prepare_manifest_sha256": sha256_path(prepare_path),
        "deployment_manifest_path": str(deployment_path),
        "deployment_manifest_sha256": deployment_sha256,
        "prepare_deployment_binding": {
            "mode": "current" if current_deployment_match else "historical",
            "recorded_path": str(recorded_deployment_path),
            "recorded_sha256": recorded_deployment_sha256,
            "current_path": str(deployment_path),
            "current_sha256": deployment_sha256,
            "explicitly_allowed": bool(arm.allow_historical_prepare_deployment),
            "recorded_path_exists": recorded_deployment_exists,
            "recorded_observed_sha256": recorded_deployment_observed_sha256,
            "recorded_digest_reverified": recorded_deployment_digest_reverified,
            "historical_digest_authority": (
                "recorded_file_reverified"
                if recorded_deployment_digest_reverified
                else "prepare_manifest_attestation_only"
            ),
        },
        "topology_path": str(topology_path),
        "topology_sha256": topology_sha256,
        "prepare_topology_binding": {
            "mode": "current" if current_topology_match else "historical",
            "recorded_path": str(recorded_topology_path),
            "recorded_sha256": recorded_topology_sha256,
            "current_path": str(topology_path),
            "current_sha256": topology_sha256,
            "explicitly_allowed": bool(arm.allow_historical_prepare_deployment),
            "recorded_digest_reverified": True,
            "historical_digest_authority": "recorded_file_reverified",
        },
        "parameters": parameters,
        "dataset": build_manifest.get("dataset") or {},
        "routing": build_manifest.get("routing") or {},
        "prepare": prepare,
        "build_manifest": build_manifest,
    }


def validate_dataset_binding(
    binding: Mapping[str, Any], dataset: ProtocolDataset
) -> dict[str, Any]:
    declared = binding.get("dataset")
    if not isinstance(declared, Mapping):
        raise RuntimeError(f"arm {binding.get('label')} build manifest lacks dataset proof")
    expected_sha256 = str(declared.get("sha256") or "").lower()
    if expected_sha256 != dataset.dataset_sha256:
        raise RuntimeError(
            f"arm {binding.get('label')} layout/dataset checksum mismatch: "
            f"{expected_sha256!r} != {dataset.dataset_sha256!r}"
        )
    dimension = declared.get("dimension")
    if dimension is not None and int(dimension) != dataset.train_shape[1]:
        raise RuntimeError(f"arm {binding.get('label')} layout/dataset dimension mismatch")
    train_rows_total = declared.get("train_rows_total")
    if train_rows_total is not None and int(train_rows_total) != dataset.train_shape[0]:
        raise RuntimeError(f"arm {binding.get('label')} layout/dataset row-count mismatch")
    return {
        "status": "PASS",
        "dataset_sha256": dataset.dataset_sha256,
        "dimension": dataset.train_shape[1],
        "train_rows_total": dataset.train_shape[0],
        "declared_dataset": dict(declared),
    }


def expected_placement_from_prepare(prepare: Mapping[str, Any]) -> dict[int, int]:
    for key in ("final_placement_proof", "placement"):
        proof = prepare.get(key)
        if not isinstance(proof, Mapping):
            continue
        raw = proof.get("expected_placement")
        if isinstance(raw, Mapping):
            placement = {int(shard): int(peer) for shard, peer in raw.items()}
            if placement:
                return placement
    raise RuntimeError("prepare manifest lacks an expected final numeric-shard placement")


def upper_navigator_sha256(artifact: Mapping[str, Any]) -> str:
    nodes = artifact.get("upper_nodes")
    graph = artifact.get("upper_graph")
    if not isinstance(nodes, list) or not isinstance(graph, dict):
        raise RuntimeError("Orion artifact lacks upper navigator data")
    navigation_nodes = []
    for node in nodes:
        if not isinstance(node, Mapping):
            raise RuntimeError("Orion upper node must be an object")
        navigation_nodes.append(
            {key: value for key, value in node.items() if key != "shard_membership"}
        )
    return canonical_sha256(
        {
            "vector_schema": artifact.get("vector_schema"),
            "upper_nodes": navigation_nodes,
            "upper_graph": graph,
        }
    )


def upper_graph_semantic_sha256(artifact: Mapping[str, Any]) -> str:
    graph = artifact.get("upper_graph")
    if not isinstance(graph, dict):
        raise RuntimeError("Orion artifact lacks upper graph data")
    return canonical_sha256(graph)


def live_collection_identity(
    info: Mapping[str, Any],
    placement: Mapping[int, int],
) -> dict[str, Any]:
    config = info.get("config") or {}
    return {
        "config": config,
        "points_count": info.get("points_count"),
        "indexed_vectors_count": info.get("indexed_vectors_count"),
        "segments_count": info.get("segments_count"),
        "status": info.get("status"),
        "optimizer_status": info.get("optimizer_status"),
        "update_queue": info.get("update_queue"),
        "placement": {str(key): placement[key] for key in sorted(placement)},
    }


def snapshot_quiescent_background_collections(
    base_url: str,
    collection_names: Sequence[str],
) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    for collection in sorted(collection_names):
        info = native.experiment.collection_info(base_url, collection)
        update_queue = info.get("update_queue") or {}
        if int(update_queue.get("length") or 0) != 0:
            raise RuntimeError(
                f"background collection {collection!r} has a non-empty update queue"
            )
        cluster = native.experiment.collection_cluster_info(base_url, collection)
        if cluster is None:
            raise RuntimeError(
                f"background collection {collection!r} lacks cluster information"
            )
        if cluster.get("shard_transfers"):
            raise RuntimeError(
                f"background collection {collection!r} has active shard transfers"
            )
        snapshots[collection] = {
            "config": info.get("config"),
            "points_count": info.get("points_count"),
            "indexed_vectors_count": info.get("indexed_vectors_count"),
            "segments_count": info.get("segments_count"),
            "status": info.get("status"),
            "optimizer_status": info.get("optimizer_status"),
            "update_queue": update_queue,
            "cluster": cluster,
        }
    return snapshots


def validate_artifact_bundle_for_arm(
    arm: ArmSpec,
    binding: Mapping[str, Any],
    artifact_proof: Mapping[str, Any],
) -> dict[str, Any]:
    build_tool = binding["build_manifest"].get("tool")
    is_l1_partition = (
        build_tool in native.prepare.ORION_L1_PARTITION_LAYOUT_TOOLS
    )
    if is_l1_partition != arm.allow_l1_partition_layout:
        raise RuntimeError(
            f"arm {arm.label} L1 partition layout authorization mismatch: "
            f"tool={build_tool!r}, allowed={arm.allow_l1_partition_layout}"
        )
    if not is_l1_partition:
        bundle = native.validate_artifact_bundle(
            "orion",
            arm.artifact,
            dict(artifact_proof),
            allow_orion_scaling_layout=arm.allow_scaling_layout,
            allow_orion_balance_layout=arm.allow_balance_layout,
        )
        assert bundle is not None
        return bundle

    layout = native.prepare.load_routed_layout(
        "orion",
        arm.layout_dir,
        allow_orion_l1_partition_layout=True,
    )
    if Path(layout["artifact_path"]).resolve() != arm.artifact.resolve():
        raise RuntimeError("L1 finalist artifact is not the production bundle artifact")
    return {
        "status": "verified",
        "layout_dir": layout["layout_dir"],
        "build_manifest_path": layout["build_manifest_path"],
        "build_manifest_sha256": layout["build_manifest_sha256"],
        "import_manifest_path": layout["import_manifest_path"],
        "import_manifest_sha256": layout["import_manifest_sha256"],
        "l1_partition_layout_proof": layout["l1_partition_layout_proof"],
        "attachment_search_ef": layout["attachment_search_ef"],
        "logical_point_count": layout["logical_point_count"],
        "physical_point_count": layout["physical_point_count"],
        "shard_count": layout["shard_count"],
    }


def validate_live_arm(
    arm: ArmSpec,
    binding: Mapping[str, Any],
    *,
    base_url: str,
    train_count: int,
    vector_dimension: int,
) -> dict[str, Any]:
    info = native.experiment.collection_info(base_url, arm.collection)
    points_count = info.get("points_count")
    if isinstance(points_count, bool) or not isinstance(points_count, int):
        raise RuntimeError(f"arm {arm.label} collection has invalid points_count")
    info = native.experiment.wait_collection_indexed(
        base_url, arm.collection, points_count
    )
    status = str(info.get("status") or "").lower()
    if status != "green" or not native.prepare.optimizer_ok(info.get("optimizer_status")):
        raise RuntimeError(f"arm {arm.label} collection is not green/ready")
    update_queue = info.get("update_queue") or {}
    if int(update_queue.get("length") or 0) != 0:
        raise RuntimeError(f"arm {arm.label} collection update queue is not empty")
    cluster = native.experiment.collection_cluster_info(base_url, arm.collection)
    if cluster is None:
        raise RuntimeError(f"arm {arm.label} collection cluster info is unavailable")
    if cluster.get("shard_transfers"):
        raise RuntimeError(f"arm {arm.label} has active shard transfers")
    config = info.get("config") or {}
    params = config.get("params") or {}
    shard_count = params.get("shard_number")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count <= 0:
        raise RuntimeError(f"arm {arm.label} has invalid shard_number")
    if params.get("replication_factor") != 1:
        raise RuntimeError(f"arm {arm.label} must use replication_factor=1")
    schema = native.collection_vector_schema(info, "")
    if schema["dimension"] != vector_dimension or schema["distance"].lower() != "cosine":
        raise RuntimeError(f"arm {arm.label} collection/dataset vector schema mismatch")
    policy = native.live_policy_for_method(info, "orion")
    artifact_payload, artifact_proof = native.validate_artifact(
        "orion",
        arm.artifact,
        policy,
        schema,
        shard_count,
        points_count,
        train_count,
    )
    assert artifact_payload is not None
    bundle = validate_artifact_bundle_for_arm(arm, binding, artifact_proof)
    native.prepare.validate_collection_provenance(
        info, binding["prepare"]["provenance_metadata"]
    )
    expected_placement = expected_placement_from_prepare(binding["prepare"])
    actual_placement = native.experiment.numeric_shard_placement_from_cluster(
        cluster, expected_shard_count=shard_count
    )
    if actual_placement != expected_placement:
        raise RuntimeError(
            f"arm {arm.label} live placement differs from its prepare manifest"
        )
    return {
        "status": "PASS",
        "label": arm.label,
        "collection": arm.collection,
        "collection_info": info,
        "collection_cluster": cluster,
        "collection_identity": live_collection_identity(info, actual_placement),
        "policy": policy,
        "vector_schema": schema,
        "shard_count": shard_count,
        "logical_point_count": artifact_proof["logical_point_count"],
        "physical_point_count": artifact_proof["physical_point_count"],
        "expected_placement": {
            str(key): expected_placement[key] for key in sorted(expected_placement)
        },
        "artifact": artifact_payload,
        "artifact_proof": artifact_proof,
        "artifact_bundle": bundle,
        "upper_navigator_sha256": upper_navigator_sha256(artifact_payload),
        "upper_graph_semantic_sha256": upper_graph_semantic_sha256(
            artifact_payload
        ),
        "hnsw_config": config.get("hnsw_config") or {},
        "optimizer_config": config.get("optimizer_config")
        or config.get("optimizers_config")
        or {},
    }


COMMON_BUILD_PARAMETER_KEYS = (
    "sample_denominator",
    "upper_sample_seed",
    "upper_m",
    "upper_ef_construction",
    "attachment_search_ef",
    "upper_graph_seed",
    "k_overlap",
    "use_multi_assign",
    "multi_assign_min_max_vote",
    "multi_assign_vote_delta",
    "multi_assign_max_shards",
    "upper_k",
    "upper_search_ef",
    "dynamic_ef_base",
    "dynamic_ef_factor",
)


def validate_cross_arm_fairness(
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    a_binding, b_binding = bindings["A"], bindings["B"]
    a_live, b_live = live["A"], live["B"]
    common_parameters = {
        key: a_binding["parameters"].get(key) for key in COMMON_BUILD_PARAMETER_KEYS
    }
    parameter_mismatches = {
        key: {
            "A": a_binding["parameters"].get(key),
            "B": b_binding["parameters"].get(key),
        }
        for key in COMMON_BUILD_PARAMETER_KEYS
        if a_binding["parameters"].get(key) != b_binding["parameters"].get(key)
    }
    equality_checks = {
        "same_upper_navigator": a_live["upper_navigator_sha256"]
        == b_live["upper_navigator_sha256"],
        "same_upper_graph_semantics": a_live["upper_graph_semantic_sha256"]
        == b_live["upper_graph_semantic_sha256"],
        "same_vector_schema": a_live["vector_schema"] == b_live["vector_schema"],
        "same_logical_shard_count": a_live["shard_count"] == b_live["shard_count"],
        "same_logical_point_count": a_live["logical_point_count"]
        == b_live["logical_point_count"],
        "same_hnsw_config": a_live["hnsw_config"] == b_live["hnsw_config"],
        "same_optimizer_config": a_live["optimizer_config"]
        == b_live["optimizer_config"],
        "same_numeric_shard_placement": a_live["expected_placement"]
        == b_live["expected_placement"],
        "same_common_build_parameters": not parameter_mismatches,
        "multi_assignment_enabled_in_both": all(
            bindings[label]["parameters"].get("use_multi_assign") is True
            for label in ARM_LABELS
        ),
        "same_frozen_historical_budget": all(
            all(
                bindings[label]["parameters"].get(key) == expected
                for key, expected in HISTORICAL_BUDGET.items()
            )
            for label in ARM_LABELS
        ),
    }
    if not all(equality_checks.values()):
        raise RuntimeError(
            "A/B fairness contract failed: "
            + json.dumps(
                {
                    "checks": equality_checks,
                    "parameter_mismatches": parameter_mismatches,
                },
                sort_keys=True,
            )
        )
    return {
        "status": "PASS",
        "checks": equality_checks,
        "common_build_parameters": common_parameters,
        "upper_navigator_sha256": a_live["upper_navigator_sha256"],
        "upper_graph_semantic_sha256": a_live[
            "upper_graph_semantic_sha256"
        ],
        "logical_shard_count": a_live["shard_count"],
        "logical_point_count": a_live["logical_point_count"],
        "physical_point_count_by_arm": {
            label: live[label]["physical_point_count"] for label in ARM_LABELS
        },
        "numeric_shard_placement": a_live["expected_placement"],
    }


def validate_adjustment_before_baseline(
    arm: ArmSpec,
    binding: Mapping[str, Any],
    live: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = binding["parameters"]
    routing = binding["routing"]
    prepare = binding["prepare"]
    placement_plan = prepare.get("placement_plan") or {}
    checks = {
        "artifact_sha256": binding["artifact_sha256"]
        == ADJUSTMENT_BEFORE_ARTIFACT_SHA256,
        "layout_build_manifest_sha256": binding["build_manifest_sha256"]
        == ADJUSTMENT_BEFORE_BUILD_MANIFEST_SHA256,
        "generation": live["policy"]["generation"]
        == ADJUSTMENT_BEFORE_GENERATION,
        "initial_p24": parameters.get("initial_num_shards")
        == ADJUSTMENT_BEFORE_INITIAL_SHARDS,
        "fission_enabled": parameters.get("enable_fission") is True,
        "final_p32": live["shard_count"] == FORMAL_LOGICAL_SHARDS
        and routing.get("effective_num_shards") == FORMAL_LOGICAL_SHARDS,
        "accepted_fission_present": any(
            event.get("accepted") is True
            for event in (routing.get("fission_events") or [])
            if isinstance(event, Mapping)
        ),
        "multi_assignment_enabled": parameters.get("use_multi_assign") is True,
        "round_robin_prepare": placement_plan.get("strategy") == "round_robin",
    }
    if not all(checks.values()):
        raise RuntimeError(
            "arm A is not the frozen 2026-08-25 adjustment-before baseline: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "PASS",
        "identity": "2026-08-25 GloVe figure adjustment-before Orion baseline",
        "description": (
            f"live collection {arm.collection!r} is a clone bound to the exact "
            "generation-3248141 artifact and build-manifest used by the figure; "
            "initial P=24; fission to P=32; multi-assignment enabled; "
            "round-robin physical placement"
        ),
        "live_clone_collection": arm.collection,
        "historical_figure_collection": "orion_vscale_glove_p24_20260825",
        "checks": checks,
    }


def validate_round_robin_four_host_contract(
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
    cluster_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    placement_by_arm: dict[str, dict[str, int]] = {}
    controller_peer_id = cluster_preflight.get("controller_peer_id")
    worker_peer_ids = cluster_preflight.get("worker_peer_ids")
    if (
        isinstance(controller_peer_id, bool)
        or not isinstance(controller_peer_id, int)
        or not isinstance(worker_peer_ids, list)
        or any(
            isinstance(peer, bool) or not isinstance(peer, int)
            for peer in worker_peer_ids
        )
    ):
        raise RuntimeError("cluster preflight lacks numeric controller/worker peer IDs")
    canonical_peer_order = [controller_peer_id, *worker_peer_ids]
    checks["cluster_has_four_distinct_peers"] = (
        len(canonical_peer_order) == FORMAL_PHYSICAL_HOSTS
        and len(set(canonical_peer_order)) == FORMAL_PHYSICAL_HOSTS
    )
    canonical_round_robin = {
        str(shard): canonical_peer_order[shard % FORMAL_PHYSICAL_HOSTS]
        for shard in range(FORMAL_LOGICAL_SHARDS)
    }
    for label in ARM_LABELS:
        prepare = bindings[label]["prepare"]
        placement_plan = prepare.get("placement_plan") or {}
        placement_peers = prepare.get("placement_peers") or {}
        proof = prepare.get("final_placement_proof") or prepare.get("placement") or {}
        placement = {
            str(shard): int(peer)
            for shard, peer in live[label]["expected_placement"].items()
        }
        plan_target = {
            str(shard): int(peer)
            for shard, peer in (placement_plan.get("target_placement") or {}).items()
        }
        declared_peer_order = [
            int(peer) for peer in (placement_peers.get("peer_ids") or [])
        ]
        placement_by_arm[label] = placement
        counts: dict[int, int] = {}
        for peer in placement.values():
            counts[peer] = counts.get(peer, 0) + 1
        checks[f"{label}_p32"] = (
            live[label]["shard_count"] == FORMAL_LOGICAL_SHARDS
            and set(int(shard) for shard in placement)
            == set(range(FORMAL_LOGICAL_SHARDS))
        )
        checks[f"{label}_round_robin_declared"] = (
            placement_plan.get("strategy") == "round_robin"
            and proof.get("placement_mode") == "round_robin"
            and placement_peers.get("mode") == "all_peers"
        )
        checks[f"{label}_placement_peers_match_cluster"] = (
            declared_peer_order == canonical_peer_order
        )
        checks[f"{label}_plan_matches_live"] = plan_target == placement
        checks[f"{label}_exact_round_robin_sequence"] = (
            placement == canonical_round_robin
        )
        checks[f"{label}_four_hosts_even"] = (
            len(counts) == FORMAL_PHYSICAL_HOSTS
            and sorted(counts.values()) == [8, 8, 8, 8]
        )
    checks["same_placement"] = placement_by_arm["A"] == placement_by_arm["B"]
    if not all(checks.values()):
        raise RuntimeError(
            "four-host P=32 round-robin placement contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "PASS",
        "checks": checks,
        "physical_host_count": FORMAL_PHYSICAL_HOSTS,
        "logical_shard_count": FORMAL_LOGICAL_SHARDS,
        "shards_per_host": 8,
        "canonical_peer_order": canonical_peer_order,
        "placement": placement_by_arm["A"],
    }


def load_protocol_dataset(path: Path) -> ProtocolDataset:
    source = path.expanduser().resolve(strict=True)
    with native.experiment.h5py.File(source, "r") as handle:
        missing = [key for key in ("train", "test", "neighbors") if key not in handle]
        if missing:
            raise ValueError(f"dataset is missing arrays: {missing}")
        train_shape = tuple(int(value) for value in handle["train"].shape)
        test_shape = tuple(int(value) for value in handle["test"].shape)
        neighbors_shape = tuple(int(value) for value in handle["neighbors"].shape)
        if len(train_shape) != 2 or len(test_shape) != 2 or len(neighbors_shape) != 2:
            raise ValueError("train, test, and neighbors must be two-dimensional")
        if test_shape[0] < TOTAL_QUERY_COUNT or neighbors_shape[0] < TOTAL_QUERY_COUNT:
            raise ValueError("fixed A/B protocol requires at least 10,000 test queries")
        if neighbors_shape[1] < TOP_K:
            raise ValueError("ground truth contains fewer than 10 neighbors")
        queries = handle["test"][:TOTAL_QUERY_COUNT].astype(np.float32, copy=True)
        neighbors = handle["neighbors"][:TOTAL_QUERY_COUNT, :TOP_K].astype(
            np.int64, copy=True
        )
        distance = str(handle.attrs.get("distance") or "")
    if train_shape[1] != test_shape[1]:
        raise ValueError("train and test dimensions differ")
    if distance.lower() not in {"angular", "cosine"}:
        raise ValueError(f"Orion balance A/B requires angular/cosine data, found {distance!r}")
    queries = native.experiment.prepare_vectors_for_distance(queries, "cosine")
    tuning_queries = queries[:TUNING_QUERY_COUNT]
    heldout_queries = queries[TUNING_QUERY_COUNT:TOTAL_QUERY_COUNT]
    tuning_neighbors = neighbors[:TUNING_QUERY_COUNT]
    heldout_neighbors = neighbors[TUNING_QUERY_COUNT:TOTAL_QUERY_COUNT]
    return ProtocolDataset(
        path=source,
        dataset_sha256=sha256_path(source),
        train_shape=(train_shape[0], train_shape[1]),
        test_shape=(test_shape[0], test_shape[1]),
        neighbors_shape=(neighbors_shape[0], neighbors_shape[1]),
        tuning_queries=tuning_queries,
        tuning_neighbors=tuning_neighbors,
        heldout_queries=heldout_queries,
        heldout_neighbors=heldout_neighbors,
        tuning_query_sha256=array_sha256(tuning_queries),
        heldout_query_sha256=array_sha256(heldout_queries),
        tuning_neighbors_sha256=array_sha256(tuning_neighbors),
        heldout_neighbors_sha256=array_sha256(heldout_neighbors),
    )


def endpoint_for(base_url: str, collection: str) -> tuple[str, int, str]:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported base URL: {base_url!r}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 6333)
    if parsed.scheme != "http":
        raise ValueError("the reused native duration runner currently requires HTTP")
    endpoint = (
        f"/collections/{urllib.parse.quote(collection, safe='')}/points/search/batch"
    )
    return host, port, endpoint


def measure_recall_split(
    arm: ArmSpec,
    queries: np.ndarray,
    neighbors: np.ndarray,
    *,
    base_url: str,
    split: str,
) -> dict[str, Any]:
    host, port, endpoint = endpoint_for(base_url, arm.collection)
    result = simple.recall_probe(
        host,
        port,
        endpoint,
        queries,
        neighbors,
        batch_size=BATCH_SIZE,
        top_k=TOP_K,
    )
    return {
        "label": arm.label,
        "collection": arm.collection,
        "endpoint": endpoint,
        "split": split,
        **result,
    }


def repository_snapshot() -> dict[str, Any]:
    provenance = native.experiment.repository_provenance(REPO_ROOT)
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    source_bundle = repo_local_execution_source_bundle()
    return {
        **provenance,
        "status_porcelain_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "execution_source_bundle": source_bundle,
        "execution_source_bundle_sha256": source_bundle["sha256"],
    }


def _python_source_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix in {".pyc", ".pyo"}:
        try:
            source = Path(importlib.util.source_from_cache(str(resolved))).resolve()
        except (ValueError, NotImplementedError):
            return resolved
        if source.is_file():
            return source
    return resolved


def repo_local_execution_source_paths() -> list[Path]:
    """Return every currently loaded repository-local Python source dependency.

    The formal runners and their shared engine may be untracked files, so a Git
    porcelain digest alone cannot detect edits made while an A/B is running.
    Validation is allowed to load additional modules between the preflight and
    measurement snapshots.  From the measurement snapshot onward, hashing the
    complete loaded-source set makes the integrity gate independent of Git
    tracking state and rejects further additions as well as byte changes.
    """

    paths: set[Path] = {_python_source_path(Path(__file__))}
    entry = Path(sys.argv[0]).expanduser()
    if entry.is_file():
        paths.add(_python_source_path(entry))
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path:
            continue
        paths.add(_python_source_path(Path(raw_path)))

    local: list[Path] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            path.relative_to(REPO_ROOT)
        except ValueError:
            continue
        local.append(path)
    return sorted(local, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def source_bundle_for_paths(paths: Sequence[Path]) -> dict[str, Any]:
    entries = []
    for raw_path in sorted(
        {_python_source_path(Path(path)) for path in paths},
        key=lambda path: path.as_posix(),
    ):
        path = raw_path.resolve(strict=True)
        try:
            display_path = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            display_path = str(path)
        entries.append(
            {
                "path": display_path,
                "sha256": sha256_path(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "algorithm": "sha256",
        "scope": "loaded_repo_local_python_sources",
        "files": entries,
        "sha256": canonical_sha256(entries),
    }


def repo_local_execution_source_bundle() -> dict[str, Any]:
    return source_bundle_for_paths(repo_local_execution_source_paths())


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_repository_snapshot(
    snapshot: Mapping[str, Any], *, label: str
) -> tuple[str, str, dict[str, tuple[str, int]]]:
    if not isinstance(snapshot, Mapping):
        raise RuntimeError(f"{label} repository snapshot must be an object")

    commit = snapshot.get("commit")
    if not isinstance(commit, str) or not commit:
        raise RuntimeError(f"{label} repository snapshot has no Git commit")
    status_sha256 = snapshot.get("status_porcelain_sha256")
    if not _is_sha256(status_sha256):
        raise RuntimeError(
            f"{label} repository snapshot has an invalid Git status digest"
        )

    bundle = snapshot.get("execution_source_bundle")
    if not isinstance(bundle, Mapping):
        raise RuntimeError(f"{label} execution source bundle must be an object")
    if bundle.get("algorithm") != "sha256":
        raise RuntimeError(f"{label} execution source bundle algorithm is invalid")
    if bundle.get("scope") != "loaded_repo_local_python_sources":
        raise RuntimeError(f"{label} execution source bundle scope is invalid")
    files = bundle.get("files")
    if not isinstance(files, list):
        raise RuntimeError(f"{label} execution source bundle files must be a list")

    entries: dict[str, tuple[str, int]] = {}
    observed_paths: list[str] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeError(
                f"{label} execution source entry {index} has an invalid schema"
            )
        path = entry.get("path")
        if not isinstance(path, str) or not path or "\\" in path:
            raise RuntimeError(
                f"{label} execution source entry {index} has an invalid path"
            )
        parsed = PurePosixPath(path)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != path
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise RuntimeError(
                f"{label} execution source entry {index} has a non-canonical path"
            )
        if path in entries:
            raise RuntimeError(
                f"{label} execution source bundle repeats path {path!r}"
            )
        sha256 = entry.get("sha256")
        size_bytes = entry.get("size_bytes")
        if not _is_sha256(sha256):
            raise RuntimeError(
                f"{label} execution source entry {path!r} has an invalid digest"
            )
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise RuntimeError(
                f"{label} execution source entry {path!r} has an invalid size"
            )
        entries[path] = (sha256, size_bytes)
        observed_paths.append(path)

    if observed_paths != sorted(observed_paths):
        raise RuntimeError(f"{label} execution source bundle paths are not sorted")
    bundle_sha256 = bundle.get("sha256")
    if not _is_sha256(bundle_sha256) or bundle_sha256 != canonical_sha256(files):
        raise RuntimeError(f"{label} execution source bundle digest is invalid")
    if snapshot.get("execution_source_bundle_sha256") != bundle_sha256:
        raise RuntimeError(
            f"{label} repository snapshot source-bundle digest is inconsistent"
        )
    return commit, status_sha256, entries


def validate_repository_preflight_transition(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Allow validation-time module loads while freezing existing source bytes."""

    before_commit, before_status, before_entries = _validated_repository_snapshot(
        before, label="preflight-start"
    )
    after_commit, after_status, after_entries = _validated_repository_snapshot(
        after, label="measurement-start"
    )
    if after_commit != before_commit:
        raise RuntimeError("Git commit changed during preflight validation")
    if after_status != before_status:
        raise RuntimeError("Git status changed during preflight validation")

    removed_paths = sorted(set(before_entries) - set(after_entries))
    if removed_paths:
        raise RuntimeError(
            "loaded repository source paths disappeared during preflight validation: "
            f"{removed_paths}"
        )
    changed_paths = sorted(
        path
        for path, identity in before_entries.items()
        if after_entries[path] != identity
    )
    if changed_paths:
        raise RuntimeError(
            "loaded repository source files changed during preflight validation: "
            f"{changed_paths}"
        )

    added_paths = sorted(set(after_entries) - set(before_entries))
    return {
        "status": "PASS",
        "mode": "allow_validation_time_source_additions_only",
        "commit_unchanged": True,
        "status_porcelain_unchanged": True,
        "existing_source_files_unchanged": True,
        "preflight_source_path_count": len(before_entries),
        "measurement_source_path_count": len(after_entries),
        "common_source_path_count": len(before_entries),
        "added_source_paths": added_paths,
    }


def validate_repository_measurement_transition(
    start: Mapping[str, Any], end: Mapping[str, Any]
) -> dict[str, Any]:
    """Require exact repository identity from first query through A/B end."""

    start_commit, start_status, start_entries = _validated_repository_snapshot(
        start, label="measurement-start"
    )
    end_commit, end_status, end_entries = _validated_repository_snapshot(
        end, label="measurement-end"
    )
    if end_commit != start_commit:
        raise RuntimeError("Git commit changed during A/B measurement")
    if end_status != start_status:
        raise RuntimeError("Git status changed during A/B measurement")

    added_paths = sorted(set(end_entries) - set(start_entries))
    removed_paths = sorted(set(start_entries) - set(end_entries))
    changed_paths = sorted(
        path
        for path in set(start_entries) & set(end_entries)
        if end_entries[path] != start_entries[path]
    )
    if added_paths or removed_paths or changed_paths:
        raise RuntimeError(
            "repository execution source bundle changed during A/B measurement: "
            f"added={added_paths}, removed={removed_paths}, changed={changed_paths}"
        )

    return {
        "status": "PASS",
        "mode": "exact_measurement_source_bundle",
        "commit_unchanged": True,
        "status_porcelain_unchanged": True,
        "source_files_unchanged": True,
        "source_path_count": len(start_entries),
        "added_source_paths": [],
        "removed_source_paths": [],
        "changed_source_paths": [],
    }


RESOURCE_FIELDS = (
    "ssh_host",
    "private_ip",
    "container",
    "container_id",
    "image",
    "image_id",
    "cpuset",
    "nano_cpus",
    "cpu_quota",
    "cpu_period",
    "cpu_max",
    "runtime_affinity",
    "environment",
)


def resource_identity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {field: row.get(field) for field in RESOURCE_FIELDS}
        for row in sorted(rows, key=lambda value: str(value.get("ssh_host") or ""))
    ]


def _container_environment(row: Mapping[str, Any], label: str) -> dict[str, str]:
    raw = row.get("environment")
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise RuntimeError(f"{label} container environment is unavailable")
    environment: dict[str, str] = {}
    for value in raw:
        key, separator, setting = value.partition("=")
        if not separator or not key or key in environment:
            raise RuntimeError(f"{label} container environment is malformed")
        environment[key] = setting
    return environment


def validate_live_deployment_attestation(
    rows: Sequence[Mapping[str, Any]],
    topology: Mapping[str, Any],
    deployment_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the running containers to the frozen deployment manifest."""
    image = deployment_evidence.get("image")
    nodes = deployment_evidence.get("nodes")
    if not isinstance(image, Mapping) or not isinstance(nodes, list):
        raise RuntimeError("deployment evidence lacks image/node identities")
    expected_image_id = str(image.get("id") or image.get("digest") or "")
    expected_image_tag = str(image.get("tag") or "")
    if not expected_image_id or not expected_image_tag:
        raise RuntimeError("deployment evidence lacks an exact image ID/tag")
    if len(nodes) != FORMAL_PHYSICAL_HOSTS or any(
        not isinstance(node, Mapping) for node in nodes
    ):
        raise RuntimeError("deployment evidence does not describe exactly four nodes")

    expected_by_ip = {str(node.get("private_ip") or ""): node for node in nodes}
    observed_by_ip = {str(row.get("private_ip") or ""): row for row in rows}
    topology_nodes = [topology.get("controller"), *(topology.get("workers") or [])]
    if any(not isinstance(node, Mapping) for node in topology_nodes):
        raise RuntimeError("formal topology lacks one controller and three workers")
    topology_by_ip = {
        str(node.get("private_ip") or ""): node for node in topology_nodes
    }
    expected_ips = set(expected_by_ip)
    checks: dict[str, bool] = {
        "four_manifest_nodes": len(expected_by_ip) == FORMAL_PHYSICAL_HOSTS
        and "" not in expected_by_ip,
        "four_live_nodes": len(observed_by_ip) == FORMAL_PHYSICAL_HOSTS
        and "" not in observed_by_ip,
        "manifest_live_topology_ips_equal": (
            expected_ips == set(observed_by_ip) == set(topology_by_ip)
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "live deployment attestation failed: " + json.dumps(failed)
        )
    seed = topology.get("hnsw_graph_build_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise RuntimeError("formal topology lacks a positive HNSW build seed")
    if topology.get("hardware_reporting") is not True:
        raise RuntimeError("formal topology must require hardware reporting")

    node_proofs: list[dict[str, Any]] = []
    for private_ip in sorted(expected_ips):
        expected = expected_by_ip[private_ip]
        observed = observed_by_ip.get(private_ip) or {}
        topology_node = topology_by_ip.get(private_ip) or {}
        environment = _container_environment(observed, f"node {private_ip}")
        container_name = str(expected.get("container_name") or "")
        node_image_id = str(expected.get("image_id") or "")
        max_search_threads = topology_node.get("max_search_threads")
        required_environment = {
            "QDRANT_HNSW_GRAPH_BUILD_SEED": str(seed),
            "QDRANT__SERVICE__HARDWARE_REPORTING": "true",
            "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS": str(
                max_search_threads
            ),
        }
        node_checks = {
            "container_name_matches": bool(container_name)
            and observed.get("container") == container_name,
            "manifest_node_image_matches_global": node_image_id
            == expected_image_id,
            "live_image_id_matches": observed.get("image_id")
            == expected_image_id,
            "live_image_tag_matches": observed.get("image")
            == expected_image_tag,
            "max_search_threads_is_positive": (
                not isinstance(max_search_threads, bool)
                and isinstance(max_search_threads, int)
                and max_search_threads > 0
            ),
            "required_environment_matches": all(
                environment.get(key) == value
                for key, value in required_environment.items()
            ),
        }
        for name, passed in node_checks.items():
            checks[f"{private_ip}_{name}"] = passed
        node_proofs.append(
            {
                "private_ip": private_ip,
                "container": observed.get("container"),
                "image": observed.get("image"),
                "image_id": observed.get("image_id"),
                "required_environment": required_environment,
                "checks": node_checks,
            }
        )

    controller = topology.get("controller") or {}
    controller_ip = str(controller.get("private_ip") or "")
    controller_environment = _container_environment(
        observed_by_ip.get(controller_ip) or {}, "controller"
    )
    wire = deployment_evidence.get("orion_compact_wire")
    if not isinstance(wire, Mapping):
        raise RuntimeError("deployment evidence lacks compact-wire identity")
    wire_version = str(wire.get("current_version") or "")
    checks["controller_compact_wire_matches"] = bool(wire_version) and (
        controller_environment.get("QDRANT_ORION_COMPACT_WIRE_VERSION")
        == wire_version
    )
    peer_premerge = deployment_evidence.get("peer_premerge")
    if not isinstance(peer_premerge, Mapping):
        raise RuntimeError("deployment evidence lacks peer-premerge identity")
    premerge_mode = str(peer_premerge.get("current_mode") or "")
    disable_value = controller_environment.get(
        "QDRANT_DISABLE_SHARD_MAJOR_PEER_PREMERGE"
    )
    if premerge_mode == "enabled":
        checks["controller_peer_premerge_matches"] = disable_value is None or (
            disable_value.lower() in {"0", "false", "no", "off"}
        )
    elif premerge_mode == "disabled":
        checks["controller_peer_premerge_matches"] = disable_value is not None and (
            disable_value.lower() in {"1", "true", "yes", "on"}
        )
    else:
        checks["controller_peer_premerge_matches"] = False
    shards_per_rpc = str(peer_premerge.get("current_shards_per_rpc") or "")
    live_shards_per_rpc = controller_environment.get(
        "QDRANT_ORION_PEER_PREMERGE_SHARDS_PER_RPC"
    )
    checks["controller_peer_premerge_shards_per_rpc_matches"] = bool(
        shards_per_rpc
    ) and (
        live_shards_per_rpc == shards_per_rpc
        or (shards_per_rpc == "all" and live_shards_per_rpc is None)
    )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "live deployment attestation failed: " + json.dumps(failed)
        )
    return {
        "status": "PASS",
        "deployment_manifest_sha256": deployment_evidence.get("manifest_sha256"),
        "image_id": expected_image_id,
        "image_tag": expected_image_tag,
        "compact_wire_version": wire_version,
        "peer_premerge_mode": premerge_mode,
        "peer_premerge_shards_per_rpc": shards_per_rpc,
        "checks": checks,
        "nodes": node_proofs,
    }


def validate_64_cpu_resource_contract(
    rows: Sequence[Mapping[str, Any]],
    topology: Mapping[str, Any],
    deployment_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    deployment_attestation = validate_live_deployment_attestation(
        rows, topology, deployment_evidence
    )
    cpuset_counts = [
        hashall.cpuset_cpu_count(str(row.get("cpuset") or "")) for row in rows
    ]
    topology_nodes = [topology.get("controller"), *(topology.get("workers") or [])]
    if any(not isinstance(node, Mapping) for node in topology_nodes):
        raise RuntimeError("formal topology lacks one controller and three workers")
    expected_host_identities = {
        (str(node.get("ssh_host") or ""), str(node.get("private_ip") or ""))
        for node in topology_nodes
    }
    observed_host_identities = {
        (str(row.get("ssh_host") or ""), str(row.get("private_ip") or ""))
        for row in rows
    }
    ssh_hosts = [str(row.get("ssh_host") or "") for row in rows]
    private_ips = [str(row.get("private_ip") or "") for row in rows]
    container_ids = [str(row.get("container_id") or "") for row in rows]
    checks = {
        "four_physical_hosts": len(rows) == FORMAL_PHYSICAL_HOSTS,
        "four_distinct_ssh_hosts": len(set(ssh_hosts)) == FORMAL_PHYSICAL_HOSTS
        and all(ssh_hosts),
        "four_distinct_private_ips": len(set(private_ips)) == FORMAL_PHYSICAL_HOSTS
        and all(private_ips),
        "four_distinct_container_ids": len(set(container_ids))
        == FORMAL_PHYSICAL_HOSTS
        and all(container_ids),
        "hosts_match_current_topology": (
            len(expected_host_identities) == FORMAL_PHYSICAL_HOSTS
            and observed_host_identities == expected_host_identities
        ),
        "qdrant_cpuset": all(
            row.get("cpuset") == hashall.QDRANT_CPUSET for row in rows
        ),
        "runtime_affinity": all(
            row.get("runtime_affinity") == hashall.QDRANT_CPUSET for row in rows
        ),
        "sixteen_cpuset_logical_cpus_per_host": all(
            count == int(FORMAL_SERVER_CPUS_PER_HOST) for count in cpuset_counts
        ),
    }
    total = float(sum(cpuset_counts))
    checks["sixty_four_server_cpus_total"] = math.isclose(
        total, FORMAL_SERVER_CPUS_TOTAL, rel_tol=0.0, abs_tol=1e-9
    )
    if not all(checks.values()):
        raise RuntimeError(
            "formal four-host/64-CPU resource contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "PASS",
        "checks": checks,
        "physical_host_count": FORMAL_PHYSICAL_HOSTS,
        "host_identities": [
            {"ssh_host": ssh_host, "private_ip": private_ip}
            for ssh_host, private_ip in sorted(observed_host_identities)
        ],
        "server_cpus_per_host": FORMAL_SERVER_CPUS_PER_HOST,
        "server_cpus_total": total,
        "capacity_source": "container cpuset logical CPU count",
        "cpu_max_by_host": {
            str(row.get("ssh_host") or index): row.get("cpu_max")
            for index, row in enumerate(rows)
        },
        "nano_cpus_by_host": {
            str(row.get("ssh_host") or index): row.get("nano_cpus")
            for index, row in enumerate(rows)
        },
        "strict_cpu_max_uniform": len(
            {str(row.get("cpu_max") or "") for row in rows}
        )
        == 1,
        "qdrant_cpuset": hashall.QDRANT_CPUSET,
        "benchmark_cpuset": hashall.BENCHMARK_CPUSET,
        "live_deployment_attestation": deployment_attestation,
    }


def verify_static_files_unchanged(bindings: Mapping[str, Mapping[str, Any]]) -> None:
    for label in ARM_LABELS:
        binding = bindings[label]
        checks = {
            Path(binding["artifact_path"]): binding["artifact_sha256"],
            Path(binding["build_manifest_path"]): binding["build_manifest_sha256"],
            Path(binding["prepare_manifest_path"]): binding["prepare_manifest_sha256"],
        }
        for path, expected in checks.items():
            actual = sha256_path(path)
            if actual != expected:
                raise RuntimeError(
                    f"arm {label} bound file changed during A/B: {path}"
                )


def validate_args(args: argparse.Namespace) -> None:
    if args.collection_a == args.collection_b:
        raise ValueError("A and B collections must be distinct")
    if args.arm_a_name == args.arm_b_name:
        raise ValueError("A and B display names must be distinct")
    if not math.isclose(
        args.target_recall, TARGET_RECALL, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("the frozen formal target Recall@10 is exactly 0.90")
    if args.sweep_seconds <= 0.0 or args.warmup_seconds <= 0.0:
        raise ValueError("sweep and warmup durations must be positive")
    if len(args.concurrency_candidates) < 2:
        raise ValueError("at least two concurrency candidates are required")
    for path_name in (
        "hdf5_path",
        "topology",
        "deployment_manifest",
        "artifact_a",
        "artifact_b",
        "layout_dir_a",
        "layout_dir_b",
        "prepare_manifest_a",
        "prepare_manifest_b",
    ):
        path = Path(getattr(args, path_name)).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)


def arm_specs(args: argparse.Namespace) -> dict[str, ArmSpec]:
    return {
        "A": ArmSpec(
            label="A",
            name=args.arm_a_name,
            collection=args.collection_a,
            artifact=args.artifact_a.expanduser().resolve(),
            layout_dir=args.layout_dir_a.expanduser().resolve(),
            prepare_manifest=args.prepare_manifest_a.expanduser().resolve(),
            allow_balance_layout=args.allow_balance_layout_a,
            allow_scaling_layout=True,
            allow_l1_partition_layout=False,
            allow_historical_prepare_deployment=(
                args.allow_historical_prepare_deployment_a
            ),
        ),
        "B": ArmSpec(
            label="B",
            name=args.arm_b_name,
            collection=args.collection_b,
            artifact=args.artifact_b.expanduser().resolve(),
            layout_dir=args.layout_dir_b.expanduser().resolve(),
            prepare_manifest=args.prepare_manifest_b.expanduser().resolve(),
            allow_balance_layout=args.allow_balance_layout_b,
            allow_scaling_layout=args.allow_scaling_layout_b,
            allow_l1_partition_layout=args.allow_l1_partition_layout_b,
            allow_historical_prepare_deployment=False,
        ),
    }


def validate_historical_comparison(
    arms: Mapping[str, ArmSpec],
    bindings: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return validate_adjustment_before_baseline(
        arms["A"], bindings["A"], live["A"]
    )


def historical_interleaved_contract() -> InterleavedABContract:
    """Return the unchanged historical-H-versus-candidate comparison contract."""

    return InterleavedABContract(
        arm_specs_factory=arm_specs,
        fairness_validator=validate_cross_arm_fairness,
        comparison_validator=validate_historical_comparison,
        comparison_manifest_key="adjustment_before_baseline",
        comparison_completion_key="adjustment_before_baseline_bound",
        fairness_budget_check_key="same_frozen_historical_budget",
        fairness_budget_completion_key="same_frozen_historical_budget",
        summary_record_type="c1_orion_balance_interleaved_ab_summary",
        run_record_type="c1_orion_balance_interleaved_ab",
        budget_mode="frozen_historical_budget_main_comparison",
        budget_subject="the frozen historical Orion search budget",
        adoption_pass_decision="ONLINE_QPS_GATE_PASS",
        adoption_fail_decision="RETAIN_ORIGINAL_ON_ONLINE_EVIDENCE",
        adoption_scope=(
            "This is the online QPS gate only; final adoption additionally "
            "requires the frozen offline topology/identity gates."
        ),
        decision_gate_key="online_qps_adoption_gate",
        semantic_ratio_key="paired_ratio_b_over_a",
        completion_error_subject="Orion balance interleaved A/B",
    )


def _run_locked(
    args: argparse.Namespace,
    held_lock: benchmark_lock.HeldBenchmarkLock,
    *,
    contract: InterleavedABContract | None = None,
) -> Path:
    comparison = contract or historical_interleaved_contract()
    output_dir = native.create_output_directory(args.output_dir)
    arms = dict(comparison.arm_specs_factory(args))
    if set(arms) != set(ARM_LABELS):
        raise ValueError(
            f"comparison arm factory must return exactly {ARM_LABELS}, "
            f"found {tuple(sorted(arms))}"
        )
    deployment_path = args.deployment_manifest.expanduser().resolve(strict=True)
    topology = native.experiment.load_cluster_topology(args.topology)
    deployment_manifest = native.experiment.load_optional_json(deployment_path)
    assert deployment_manifest is not None
    deployment_evidence = native.build_deployment_evidence(
        deployment_path, deployment_manifest
    )
    native.validate_deployment_topology_transport_binding(
        topology, deployment_evidence
    )
    cluster_preflight = native.experiment.validate_cluster_preflight(
        args.base_url, topology
    )
    repository_preflight_start = repository_snapshot()
    client_affinity_start = hashall.verify_client_affinity()
    resources_start = hashall.inspect_all_containers()
    dataset = load_protocol_dataset(args.hdf5_path)

    collection_rows = native.experiment.request_json(
        args.base_url, "GET", "/collections"
    )["result"]["collections"]
    observed_collections = {str(row.get("name") or "") for row in collection_rows}
    if "" in observed_collections:
        raise RuntimeError(f"invalid live collection inventory: {collection_rows}")
    expected_collections = {arm.collection for arm in arms.values()}
    if not expected_collections.issubset(observed_collections):
        raise RuntimeError(
            "read-only balance A/B requires both declared prepared collections: "
            f"collections: observed={sorted(observed_collections)}, "
            f"expected={sorted(expected_collections)}"
        )
    background_collections = sorted(observed_collections - expected_collections)
    background_start = snapshot_quiescent_background_collections(
        args.base_url, background_collections
    )

    prepare_binding_validator = (
        comparison.prepare_binding_validator or validate_prepare_binding
    )
    bindings = {
        label: prepare_binding_validator(
            arm,
            base_url=args.base_url,
            deployment_manifest=deployment_path,
            topology=args.topology,
        )
        for label, arm in arms.items()
    }
    dataset_bindings = {
        label: validate_dataset_binding(bindings[label], dataset)
        for label in ARM_LABELS
    }
    live_start = {
        label: validate_live_arm(
            arm,
            bindings[label],
            base_url=args.base_url,
            train_count=dataset.train_shape[0],
            vector_dimension=dataset.train_shape[1],
        )
        for label, arm in arms.items()
    }
    fairness = comparison.fairness_validator(bindings, live_start)
    comparison_validation = comparison.comparison_validator(
        arms, bindings, live_start
    )
    adoption_prerequisites: dict[str, Any] | None = None
    if comparison.adoption_prerequisite_validator is not None:
        adoption_prerequisites = comparison.adoption_prerequisite_validator(
            args, bindings, comparison_validation
        )
        if (
            not isinstance(adoption_prerequisites, dict)
            or adoption_prerequisites.get("status") != "PASS"
        ):
            raise RuntimeError(
                "comparison-specific adoption prerequisites did not pass"
            )
    placement_contract = validate_round_robin_four_host_contract(
        bindings, live_start, cluster_preflight
    )
    resource_contract = validate_64_cpu_resource_contract(
        resources_start, topology, deployment_evidence
    )
    repository_measurement_start = repository_snapshot()
    repository_preflight_transition = validate_repository_preflight_transition(
        repository_preflight_start, repository_measurement_start
    )

    recall: dict[str, dict[str, Any]] = {}
    recall_order: list[dict[str, Any]] = []
    # The historical u48/EF48/base50/factor14 budget is eligible for the main
    # comparison only while tuning Recall remains in the frozen [0.90, 0.93)
    # band.  Outside that band, an external checksum-bound runtime-profile
    # sweep is required because this read-only runner cannot activate artifacts.
    for label in ARM_LABELS:
        tuning = measure_recall_split(
            arms[label],
            dataset.tuning_queries,
            dataset.tuning_neighbors,
            base_url=args.base_url,
            split="tuning",
        )
        recall[label] = {"tuning": tuning}
        recall_order.append(
            {
                "position": len(recall_order) + 1,
                "arm": label,
                "split": "tuning",
                "query_count": TUNING_QUERY_COUNT,
            }
        )
    tuning_budget_violations = {
        label: float(recall[label]["tuning"]["recall_at_10"])
        for label in ARM_LABELS
        if not (
            args.target_recall
            <= float(recall[label]["tuning"]["recall_at_10"])
            < TUNING_RECALL_UPPER_EXCLUSIVE
        )
    }
    if tuning_budget_violations:
        raise RuntimeError(
            f"{comparison.budget_subject} is outside the tuning "
            f"Recall band [{args.target_recall:.2f}, "
            f"{TUNING_RECALL_UPPER_EXCLUSIVE:.2f}); run a separate "
            "checksum-bound symmetric runtime-profile sweep: "
            f"{tuning_budget_violations}"
        )
    budget_selection = {
        "mode": comparison.budget_mode,
        "budget": HISTORICAL_BUDGET,
        "selection_rule": (
            "retain u48/upperEF48/base50/factor14 only when each arm's 1000-query "
            "tuning Recall@10 lies in [0.90, 0.93); otherwise fail closed and "
            "require a checksum-bound symmetric runtime-profile sweep"
        ),
        "tuning_recall_at_10": {
            label: float(recall[label]["tuning"]["recall_at_10"])
            for label in ARM_LABELS
        },
        "passed": True,
    }

    tuning_bodies = simple.make_simple_bodies(
        dataset.tuning_queries,
        batch_size=BATCH_SIZE,
        top_k=TOP_K,
    )
    endpoints = {
        label: endpoint_for(args.base_url, arm.collection) for label, arm in arms.items()
    }

    sweep_pairs: list[dict[str, Any]] = []
    for candidate_number, concurrency in enumerate(
        args.concurrency_candidates, start=1
    ):
        order = paired_order(candidate_number)
        results: dict[str, dict[str, Any]] = {}
        for label in order:
            host, port, endpoint = endpoints[label]
            results[label] = hashall.timed_run(
                host,
                port,
                endpoint,
                tuning_bodies,
                concurrency=concurrency,
                duration_s=args.sweep_seconds,
                batch_size=BATCH_SIZE,
            )
        sweep_pairs.append(
            {
                "candidate_number": candidate_number,
                "concurrency": concurrency,
                "execution_order": list(order),
                "A": results["A"],
                "B": results["B"],
            }
        )
    saturation = select_independent_saturation(sweep_pairs)
    if not saturation["all_knees_observed"]:
        raise RuntimeError(
            "one or both arm-specific saturation knees were not observed on the "
            "shared tuning-only concurrency grid; extend the grid before formal timing"
        )
    selected_concurrency = {
        label: int(saturation["selected_concurrency_by_arm"][label])
        for label in ARM_LABELS
    }

    # Held-out queries are first issued only after both the frozen search budget
    # and each arm's saturation concurrency have been selected from tuning.
    for label in ARM_LABELS:
        heldout = measure_recall_split(
            arms[label],
            dataset.heldout_queries,
            dataset.heldout_neighbors,
            base_url=args.base_url,
            split="heldout",
        )
        recall[label]["heldout"] = heldout
        recall_order.append(
            {
                "position": len(recall_order) + 1,
                "arm": label,
                "split": "heldout",
                "query_count": HELDOUT_QUERY_COUNT,
            }
        )
    recall_failures = {
        label: float(recall[label]["heldout"]["recall_at_10"])
        for label in ARM_LABELS
        if float(recall[label]["heldout"]["recall_at_10"])
        < args.target_recall
    }
    if recall_failures:
        raise RuntimeError(f"held-out Recall@10 gate failed: {recall_failures}")

    heldout_bodies = simple.make_simple_bodies(
        dataset.heldout_queries,
        batch_size=BATCH_SIZE,
        top_k=TOP_K,
    )

    # Two half-duration warmup pairs give each arm one first and one second
    # position while preserving equal total warmup time per arm.
    warmup_legs: list[dict[str, Any]] = []
    for warmup_pair in (1, 2):
        order = paired_order(warmup_pair)
        for position, label in enumerate(order, start=1):
            host, port, endpoint = endpoints[label]
            warmup_legs.append(
                {
                    "warmup_pair": warmup_pair,
                    "execution_order": list(order),
                    "execution_position": position,
                    "arm": label,
                    "result": hashall.timed_run(
                        host,
                        port,
                        endpoint,
                        heldout_bodies,
                        concurrency=selected_concurrency[label],
                        duration_s=args.warmup_seconds / 2.0,
                        batch_size=BATCH_SIZE,
                    ),
                }
            )

    def formal_runner(label: str, pair_number: int, position: int) -> Mapping[str, Any]:
        del pair_number, position
        host, port, endpoint = endpoints[label]
        return hashall.timed_run(
            host,
            port,
            endpoint,
            heldout_bodies,
            concurrency=selected_concurrency[label],
            duration_s=MEASURE_SECONDS,
            batch_size=BATCH_SIZE,
        )

    formal = run_formal_pairs(
        formal_runner,
        {
            label: float(recall[label]["heldout"]["recall_at_10"])
            for label in ARM_LABELS
        },
    )

    live_end = {
        label: validate_live_arm(
            arm,
            bindings[label],
            base_url=args.base_url,
            train_count=dataset.train_shape[0],
            vector_dimension=dataset.train_shape[1],
        )
        for label, arm in arms.items()
    }
    for label in ARM_LABELS:
        if live_end[label]["collection_identity"] != live_start[label]["collection_identity"]:
            raise RuntimeError(f"arm {label} live collection changed during A/B")
    resources_end = hashall.inspect_all_containers()
    if resource_identity(resources_end) != resource_identity(resources_start):
        raise RuntimeError("container resource identity changed during read-only A/B")
    client_affinity_end = hashall.verify_client_affinity()
    if client_affinity_end != client_affinity_start:
        raise RuntimeError("benchmark client affinity changed during A/B")
    native.verify_deployment_evidence_unchanged(deployment_evidence)
    verify_static_files_unchanged(bindings)
    ending_collection_rows = native.experiment.request_json(
        args.base_url, "GET", "/collections"
    )["result"]["collections"]
    ending_collections = {str(row.get("name") or "") for row in ending_collection_rows}
    if ending_collections != observed_collections:
        raise RuntimeError(
            "live collection inventory changed during read-only A/B: "
            f"start={sorted(observed_collections)}, end={sorted(ending_collections)}"
        )
    background_end = snapshot_quiescent_background_collections(
        args.base_url, background_collections
    )

    adoption_prerequisites_unchanged = True
    if comparison.adoption_prerequisite_validator is not None:
        adoption_prerequisites_end = (
            comparison.adoption_prerequisite_validator(
                args, bindings, comparison_validation
            )
        )
        adoption_prerequisites_unchanged = (
            adoption_prerequisites_end == adoption_prerequisites
        )
        if not adoption_prerequisites_unchanged:
            raise RuntimeError(
                "comparison-specific adoption prerequisites changed during A/B"
            )
    repository_end = repository_snapshot()
    repository_measurement_unchanged = validate_repository_measurement_transition(
        repository_measurement_start, repository_end
    )
    repository_unchanged = (
        repository_preflight_transition.get("status") == "PASS"
        and repository_measurement_unchanged.get("status") == "PASS"
    )

    ratio_values = [
        float(pair["paired_ratio_b_over_a"]) for pair in formal["pairs"]
    ]
    ratio_stats = qps_statistics(ratio_values)
    ratio_ci = paired_ratio_confidence_interval(ratio_values)
    online_qps_adoption_gate = adoption_gate_passes(
        stable=formal["stable"],
        recall_failures=recall_failures,
        ratio_ci_lower=float(ratio_ci["lower"]),
        adoption_prerequisites=adoption_prerequisites,
    )
    ratio_summary = {
        **ratio_stats,
        "confidence_interval": ratio_ci,
        "b_faster_pair_count": sum(value > 1.0 for value in ratio_values),
        "a_faster_pair_count": sum(value < 1.0 for value in ratio_values),
        "tie_pair_count": sum(value == 1.0 for value in ratio_values),
    }
    summary = {
        "timestamp": utc_timestamp(),
        "record_type": comparison.summary_record_type,
        "status": "PASS" if formal["stable"] else "UNSTABLE",
        "arm_names": {label: arms[label].name for label in ARM_LABELS},
        "collections": {label: arms[label].collection for label in ARM_LABELS},
        "target_recall_at_10": args.target_recall,
        "heldout_recall_at_10": {
            label: float(recall[label]["heldout"]["recall_at_10"])
            for label in ARM_LABELS
        },
        "frozen_search_budget": budget_selection,
        "selected_saturation_concurrency_by_arm": selected_concurrency,
        "pair_count": len(formal["pairs"]),
        "extended_to_seven": formal["extended_to_seven"],
        "qps": formal["final_stats"],
        "paired_ratio_b_over_a": ratio_summary,
        "stability_gate": {
            "max_qps_cv": MAX_QPS_CV,
            "passed": formal["stable"],
        },
        comparison.decision_gate_key: {
            "rule": (
                "held-out Recall@10 >= 0.90 for both arms, QPS CV <= 5%, and "
                "the two-sided 95% paired B/A QPS-ratio confidence interval is "
                "entirely above 1.0"
                + (
                    ", with every checksum-bound comparison-specific adoption "
                    "prerequisite passing"
                    if adoption_prerequisites is not None
                    else ""
                )
            ),
            "passed": online_qps_adoption_gate,
            "decision": (
                comparison.adoption_pass_decision
                if online_qps_adoption_gate
                else comparison.adoption_fail_decision
            ),
            "scope": comparison.adoption_scope,
        },
    }
    if adoption_prerequisites is not None:
        summary[comparison.decision_gate_key]["adoption_prerequisites"] = (
            adoption_prerequisites
        )
        summary[comparison.decision_gate_key][
            "adoption_prerequisites_unchanged_during_ab"
        ] = adoption_prerequisites_unchanged
    if comparison.semantic_ratio_key != "paired_ratio_b_over_a":
        summary[comparison.semantic_ratio_key] = ratio_summary
    roles = comparison_validation.get("roles")
    if isinstance(roles, Mapping):
        summary["arm_roles"] = dict(roles)
    measurements = {
        "recall_probe_order": recall_order,
        "recall": recall,
        "budget_selection": budget_selection,
        "concurrency_sweep": sweep_pairs,
        "saturation_selection": saturation,
        "warmup_legs": warmup_legs,
        "formal": formal,
    }
    run_manifest = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "record_type": comparison.run_record_type,
        "status": summary["status"],
        "command": [
            sys.executable,
            *benchmark_lock.strip_cli_arguments(sys.argv),
        ],
        "read_only_contract": {
            "collection_set_must_preexist": True,
            "declared_arm_collections": sorted(expected_collections),
            "full_collection_inventory_start": sorted(observed_collections),
            "full_collection_inventory_end": sorted(ending_collections),
            "background_collections": background_collections,
            "background_collections_start": background_start,
            "background_collections_end": background_end,
            "forbidden_mutations": list(FORBIDDEN_MUTATIONS),
            "resource_configuration_action": "inspect_and_compare_only",
            "routing_artifact_action": "validate_only",
        },
        "protocol": {
            "dataset": "GloVe-200-angular",
            "distance": "Cosine",
            "top_k": TOP_K,
            "batch_size": BATCH_SIZE,
            "tuning_query_range": [0, TUNING_QUERY_COUNT],
            "heldout_query_range": [TUNING_QUERY_COUNT, TOTAL_QUERY_COUNT],
            "target_recall_at_10": args.target_recall,
            "concurrency_candidates": args.concurrency_candidates,
            "arm_specific_concurrency_rule": saturation["selection_rule"],
            "concurrency_selection_split": "tuning_only",
            "sweep_seconds": args.sweep_seconds,
            "warmup_seconds_per_arm": args.warmup_seconds,
            "measure_seconds": MEASURE_SECONDS,
            "minimum_paired_rounds": MIN_PAIRED_ROUNDS,
            "maximum_paired_rounds": MAX_PAIRED_ROUNDS,
            "extension_rule": "if either five-run QPS CV exceeds 5%, run pairs 6 and 7",
            "pair_order_rule": "odd A->B; even B->A",
        },
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.dataset_sha256,
            "train_shape": list(dataset.train_shape),
            "test_shape": list(dataset.test_shape),
            "neighbors_shape": list(dataset.neighbors_shape),
            "tuning_query_sha256": dataset.tuning_query_sha256,
            "heldout_query_sha256": dataset.heldout_query_sha256,
            "tuning_neighbors_sha256": dataset.tuning_neighbors_sha256,
            "heldout_neighbors_sha256": dataset.heldout_neighbors_sha256,
        },
        "request_contract": {
            "standard_coordinator_batch_search": True,
            "client_side_fanout": False,
            "shard_selector_present": False,
            "client_hnsw_ef_present": False,
            "tuning_request_bodies_sha256": canonical_sha256(
                [hashlib.sha256(body).hexdigest() for body in tuning_bodies]
            ),
            "heldout_request_bodies_sha256": canonical_sha256(
                [hashlib.sha256(body).hexdigest() for body in heldout_bodies]
            ),
        },
        "benchmark_lock": held_lock.evidence(),
        "topology": topology,
        "cluster_preflight": cluster_preflight,
        "deployment": deployment_evidence,
        # Compatibility aliases retain the historical fields while the
        # explicit three-snapshot proof distinguishes validation-time imports
        # from the exact measurement interval.
        "repository_start": repository_preflight_start,
        "repository_preflight_start": repository_preflight_start,
        "repository_measurement_start": repository_measurement_start,
        "repository_end": repository_end,
        "repository_preflight_transition": repository_preflight_transition,
        "repository_measurement_unchanged": repository_measurement_unchanged,
        "repository_unchanged": repository_unchanged,
        "resources": {
            "client_affinity_start": client_affinity_start,
            "client_affinity_end": client_affinity_end,
            "containers_start": resources_start,
            "containers_end": resources_end,
            "unchanged": True,
        },
        "arm_bindings": {
            label: {
                key: value
                for key, value in bindings[label].items()
                if key not in {"prepare", "build_manifest"}
            }
            for label in ARM_LABELS
        },
        "dataset_bindings": dataset_bindings,
        "live_arm_start": {
            label: {
                key: value
                for key, value in live_start[label].items()
                if key not in {"artifact", "collection_info", "collection_cluster"}
            }
            for label in ARM_LABELS
        },
        "live_arm_end": {
            label: {
                key: value
                for key, value in live_end[label].items()
                if key not in {"artifact", "collection_info", "collection_cluster"}
            }
            for label in ARM_LABELS
        },
        "cross_arm_fairness": fairness,
        "round_robin_four_host_contract": placement_contract,
        "resource_contract": resource_contract,
        "files": {
            "summary": "summary.json",
            "measurements": "measurements.json",
            "paired_rounds": "paired_rounds.csv",
            "concurrency_sweep": "concurrency_sweep.csv",
            "completion_audit": "completion-audit.json",
        },
    }
    run_manifest[comparison.comparison_manifest_key] = comparison_validation
    if adoption_prerequisites is not None:
        run_manifest["adoption_prerequisites"] = adoption_prerequisites
    completion_checks = {
        "single_common_lock": run_manifest["benchmark_lock"]["mode"]
        in {"acquired", "inherited"},
        "declared_arms_are_distinct_observed_subset": expected_collections.issubset(
            observed_collections
        )
        and len(expected_collections) == 2,
        "full_collection_inventory_unchanged": ending_collections
        == observed_collections,
        "background_collections_quiescent_at_start_and_end": (
            set(background_start) == set(background_collections)
            and set(background_end) == set(background_collections)
            and background_start == background_end
        ),
        "frozen_budget_tuning_band_pass": not tuning_budget_violations,
        "both_recall_pass": not recall_failures,
        "arm_specific_concurrencies_selected": all(
            selected_concurrency[label]
            == saturation["independent_arm_selections"][label][
                "selected_concurrency"
            ]
            for label in ARM_LABELS
        ),
        "both_saturation_knees_observed": saturation["all_knees_observed"] is True,
        "odd_even_pair_order": all(
            tuple(pair["execution_order"]) == paired_order(int(pair["pair_number"]))
            for pair in formal["pairs"]
        ),
        "formal_pair_count_valid": len(formal["pairs"])
        in {MIN_PAIRED_ROUNDS, MAX_PAIRED_ROUNDS},
        "twenty_second_formal_contract": MEASURE_SECONDS == 20.0,
        "qps_cv_pass": formal["stable"],
        "four_host_64_cpu_contract": resource_contract["status"] == "PASS",
        "p32_round_robin_contract": placement_contract["status"] == "PASS",
        comparison.comparison_completion_key: comparison_validation["status"]
        == "PASS",
        "adoption_prerequisites_pass": (
            adoption_prerequisites is None
            or adoption_prerequisites.get("status") == "PASS"
        ),
        "adoption_prerequisites_unchanged": adoption_prerequisites_unchanged,
        "resources_unchanged": True,
        "affinity_unchanged": client_affinity_start == client_affinity_end,
        "collections_unchanged": all(
            live_start[label]["collection_identity"]
            == live_end[label]["collection_identity"]
            for label in ARM_LABELS
        ),
        "bound_files_unchanged": True,
        "same_upper_navigator": fairness["checks"]["same_upper_navigator"],
        "same_upper_graph_semantics": fairness["checks"][
            "same_upper_graph_semantics"
        ],
        comparison.fairness_budget_completion_key: fairness["checks"][
            comparison.fairness_budget_check_key
        ],
    }
    if comparison.multi_assignment_retained_check_key is None:
        completion_checks["multi_assignment_retained_not_required_by_contract"] = True
    else:
        completion_checks["multi_assignment_retained"] = fairness["checks"][
            comparison.multi_assignment_retained_check_key
        ]
    completion = {
        "timestamp": utc_timestamp(),
        "status": "PASS" if all(completion_checks.values()) else "FAIL",
        "checks": completion_checks,
        "passed": sum(bool(value) for value in completion_checks.values()),
        "total": len(completion_checks),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "measurements.json", measurements)
    write_csv(
        output_dir / "paired_rounds.csv",
        paired_round_csv_rows(formal["pairs"]),
    )
    write_csv(
        output_dir / "concurrency_sweep.csv",
        concurrency_sweep_csv_rows(sweep_pairs),
    )
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "completion-audit.json", completion)
    if completion["status"] != "PASS":
        raise RuntimeError(
            f"{comparison.completion_error_subject} completion audit failed: "
            f"{completion_checks}"
        )
    return output_dir


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    output = Path(args.output_dir).expanduser().resolve()
    output_preexisted = output.exists()
    with benchmark_lock.hold_from_args(
        args,
        args.deployment_manifest,
        owner={
            "kind": "c1_orion_balance_interleaved_ab",
            "collection_a": args.collection_a,
            "collection_b": args.collection_b,
            "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        },
    ) as held_lock:
        try:
            return _run_locked(args, held_lock)
        except BaseException as exc:
            if not output_preexisted and output.is_dir():
                try:
                    write_json(
                        output / "execution-failed.json",
                        {
                            "timestamp": utc_timestamp(),
                            "record_type": "c1_orion_balance_interleaved_ab_failure",
                            "status": "FAIL",
                            "error": repr(exc),
                            "benchmark_lock": held_lock.evidence(),
                        },
                    )
                except Exception:
                    pass
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--hdf5-path", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm-a-name", default="adjustment_before_p24_fission32")
    parser.add_argument("--arm-b-name", default="balanced")
    parser.add_argument("--collection-a", required=True)
    parser.add_argument("--collection-b", required=True)
    parser.add_argument("--artifact-a", type=Path, required=True)
    parser.add_argument("--artifact-b", type=Path, required=True)
    parser.add_argument("--layout-dir-a", type=Path, required=True)
    parser.add_argument("--layout-dir-b", type=Path, required=True)
    parser.add_argument("--prepare-manifest-a", type=Path, required=True)
    parser.add_argument("--prepare-manifest-b", type=Path, required=True)
    parser.add_argument("--allow-balance-layout-a", action="store_true")
    parser.add_argument("--allow-balance-layout-b", action="store_true")
    parser.add_argument("--allow-scaling-layout-b", action="store_true")
    parser.add_argument(
        "--allow-l1-partition-layout-b",
        action="store_true",
        help=(
            "Explicitly authorize the frozen experiments/l1_balance finalist "
            "bundle for arm B."
        ),
    )
    parser.add_argument(
        "--allow-historical-prepare-deployment-a",
        action="store_true",
        help=(
            "Allow only the frozen adjustment-before Arm A prepare manifest to "
            "bind historical deployment "
            "and topology digests. The live run remains bound to the current "
            "--deployment-manifest and --topology."
        ),
    )
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    parser.add_argument(
        "--concurrency-candidates", default="1,2,4,8,16,32,64"
    )
    parser.add_argument("--sweep-seconds", type=float, default=8.0)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    benchmark_lock.add_cli_arguments(parser)
    args = parser.parse_args(argv)
    args.concurrency_candidates = parse_int_csv(args.concurrency_candidates)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(json.dumps({"output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
