#!/usr/bin/env python3
"""Build the fail-closed cross-dataset v2 offline confirmation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


METHOD = "mass-balanced-kmeans"
MASS_MODE = "raw-regularized-v2"
MASS_SOURCE = "production_upper_navigation_top10_hit_frequency_regularized_v2"
MASS_TRANSFORM = "raw_count_with_unit_l1_prior"
ESTIMATOR_VERSION = 2
AMENDMENT_ID = "post-exploratory-raw-regularized-v2-confirmation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def checked_manifest(path: Path, *, sidecar: bool) -> tuple[dict[str, Any], str]:
    digest = sha256_path(path)
    if sidecar:
        checksum_path = path.with_name(path.name + ".sha256")
        if checksum_path.read_text(encoding="ascii").strip() != digest:
            raise ValueError(f"manifest sidecar mismatch: {path}")
    return load_json(path), digest


def one_candidate(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    matches = [
        value
        for value in manifest.get("candidates", [])
        if isinstance(value, dict) and value.get("method") == METHOD
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {METHOD} candidate: {path}")
    return matches[0]


def validate_amendment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("v2 protocol amendment is missing")
    expected = {
        "format_version": 1,
        "id": AMENDMENT_ID,
        "status": "post_exploratory_promotion_not_preregistered",
        "mass_mode": MASS_MODE,
        "source": MASS_SOURCE,
        "transform": MASS_TRANSFORM,
        "estimator_version": ESTIMATOR_VERSION,
        "method": METHOD,
        "topology_gates_unchanged": True,
        "old_raw_owner_byte_parity_required": True,
        "online_qps_only_confirmation": True,
    }
    errors = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if errors:
        raise ValueError(f"v2 amendment mismatch: {errors}")
    amendment_path = Path(value["amendment_path"]).resolve()
    protocol_path = Path(value["protocol_path"]).resolve()
    if sha256_path(amendment_path) != value["amendment_sha256"]:
        raise ValueError("amendment checksum drifted")
    if sha256_path(protocol_path) != value["protocol_sha256"]:
        raise ValueError("protocol checksum drifted")
    return value


def dataset_record(root: Path, dataset: str) -> dict[str, Any]:
    dataset_root = root / f"{dataset}-p32"
    candidate_path = (
        dataset_root / "candidates-mass-regularized-v2/candidate-manifest.json"
    ).resolve()
    screen_path = (
        dataset_root / "screen-mass-regularized-v2/screen-manifest.json"
    ).resolve()
    frozen, frozen_sha256 = checked_manifest(candidate_path, sidecar=True)
    screen, screen_sha256 = checked_manifest(screen_path, sidecar=True)
    if screen.get("stage") != "post_freeze_evaluation":
        raise ValueError(f"unexpected phase-B stage for {dataset}")
    if screen.get("frozen_candidate_manifest") != str(candidate_path):
        raise ValueError(f"phase B does not bind the expected phase A for {dataset}")
    if screen.get("frozen_candidate_manifest_sha256") != frozen_sha256:
        raise ValueError(f"phase A SHA differs in phase B for {dataset}")

    mass_info = screen.get("mass_estimator") or {}
    expected_mass = {
        "mass_mode": MASS_MODE,
        "source": MASS_SOURCE,
        "transform": MASS_TRANSFORM,
        "estimator_version": ESTIMATOR_VERSION,
        "eligible_as_finalist": True,
    }
    for key, expected in expected_mass.items():
        if mass_info.get(key) != expected:
            raise ValueError(f"{dataset} mass estimator {key} mismatch")
    amendment = validate_amendment(screen.get("protocol_amendment"))
    if frozen.get("protocol_amendment") != amendment:
        raise ValueError(f"{dataset} phase A/B amendment mismatch")

    frozen_candidate = one_candidate(frozen, candidate_path)
    candidate = one_candidate(screen, screen_path)
    if candidate.get("owner_sha256") != frozen_candidate.get("owner_sha256"):
        raise ValueError(f"{dataset} owner SHA changed after freeze")
    if candidate.get("protocol_amendment") != amendment:
        raise ValueError(f"{dataset} candidate amendment mismatch")
    required_true = (
        "graph_topology_all_pass",
        "topology_all_pass",
        "identity_all_pass",
        "owner_parity_all_pass",
        "materialization_eligible",
    )
    for field in required_true:
        if candidate.get(field) is not True:
            raise ValueError(f"{dataset} candidate failed {field}")
    failed_gates = {
        key: value
        for key, value in candidate.get("topology_gates", {}).items()
        if value.get("pass") is not True
    }
    if failed_gates:
        raise ValueError(f"{dataset} has failed topology gates: {failed_gates}")

    owner_path = Path(candidate["owner_path"]).resolve()
    owner_sha256 = sha256_path(owner_path)
    if owner_sha256 != candidate["owner_sha256"]:
        raise ValueError(f"{dataset} v2 owner bytes drifted")
    parity = candidate.get("owner_parity") or {}
    raw_manifest_path = Path(parity["reference_candidate_manifest_path"]).resolve()
    raw_manifest, raw_manifest_sha256 = checked_manifest(
        raw_manifest_path, sidecar=True
    )
    if raw_manifest_sha256 != parity["reference_candidate_manifest_sha256"]:
        raise ValueError(f"{dataset} raw manifest SHA differs from parity record")
    raw_candidate = one_candidate(raw_manifest, raw_manifest_path)
    raw_owner_path = Path(raw_candidate["owner_path"]).resolve()
    raw_owner_sha256 = sha256_path(raw_owner_path)
    if (
        parity.get("owner_bytes_identical") is not True
        or parity.get("reference_owner_path") != str(raw_owner_path)
        or parity.get("reference_owner_sha256") != raw_owner_sha256
        or owner_sha256 != raw_owner_sha256
        or owner_path.read_bytes() != raw_owner_path.read_bytes()
    ):
        raise ValueError(f"{dataset} v2/raw owner parity failed")

    construction = screen["construction_inputs"]
    artifact_path = Path(construction["artifact"]).resolve()
    attachments = screen["post_freeze_evaluation_inputs"]
    actual = candidate["actual_metrics"]
    return {
        "dataset": dataset,
        "logical_shards": int(frozen["parameters"]["num_partitions"]),
        "logical_point_count": int(construction["logical_point_count"]),
        "phase_a_candidate_manifest_path": str(candidate_path),
        "phase_a_candidate_manifest_sha256": frozen_sha256,
        "phase_b_screen_manifest_path": str(screen_path),
        "phase_b_screen_manifest_sha256": screen_sha256,
        "source_artifact_path": str(artifact_path),
        "source_artifact_sha256": construction["artifact_sha256"],
        "owner_path": str(owner_path),
        "owner_sha256": owner_sha256,
        "raw_reference_candidate_manifest_path": str(raw_manifest_path),
        "raw_reference_candidate_manifest_sha256": raw_manifest_sha256,
        "raw_reference_owner_path": str(raw_owner_path),
        "raw_reference_owner_sha256": raw_owner_sha256,
        "owner_bytes_identical": True,
        "identity_all_pass": True,
        "graph_topology_all_pass": True,
        "topology_all_pass": True,
        "owner_parity_all_pass": True,
        "materialization_eligible": True,
        "topology_gates": candidate["topology_gates"],
        "metrics": {
            "upper_edge_cut_ratio": candidate["topology_metrics"][
                "upper_edge_cut_ratio"
            ],
            "l0_load_max_over_mean": actual["load_max_over_mean"],
            "l0_load_cv": actual["load_cv"],
            "expansion_ratio": actual["expansion_ratio"],
            "routed_shards_mean": actual["routed_shards_mean"],
            "gt_routing_coverage_mean": actual["gt_routing_coverage_mean"],
        },
        "materialization_parity": candidate["materialization_parity"],
        "materializer_inputs": {
            "source_artifact": str(artifact_path),
            "source_artifact_sha256": construction["artifact_sha256"],
            "attachments": attachments["attachments"],
            "attachments_sha256": attachments["attachments_sha256"],
            "attachments_manifest": attachments["attachments_manifest"],
            "attachments_manifest_sha256": attachments[
                "attachments_manifest_sha256"
            ],
            "owner": str(owner_path),
            "owner_sha256": owner_sha256,
            "screen_manifest": str(screen_path),
            "screen_manifest_sha256": screen_sha256,
            "method": METHOD,
        },
    }


def v1_heldout_failure(root: Path) -> dict[str, Any]:
    path = (root / "glove-p32/screen-mass-floor1-v1/screen-manifest.json").resolve()
    manifest, digest = checked_manifest(path, sidecar=False)
    candidate = one_candidate(manifest, path)
    failed = {
        key: value
        for key, value in candidate.get("topology_gates", {}).items()
        if value.get("pass") is not True
    }
    if candidate.get("topology_all_pass") is not False:
        raise ValueError("v1 GloVe floor1 result no longer records a failure")
    if set(failed) != {"routed_shards_mean"}:
        raise ValueError(f"unexpected v1 GloVe failed gates: {failed}")
    return {
        "dataset": "glove",
        "mass_mode": "self-debiased-floor1",
        "method": METHOD,
        "screen_manifest_path": str(path),
        "screen_manifest_sha256": digest,
        "topology_all_pass": False,
        "materialization_eligible": False,
        "failed_gates": failed,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    datasets = {
        dataset: dataset_record(root, dataset) for dataset in ("sift", "glove")
    }
    amendments = {
        json.dumps(
            load_json(Path(record["phase_b_screen_manifest_path"]))[
                "protocol_amendment"
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in datasets.values()
    }
    if len(amendments) != 1:
        raise ValueError("SIFT and GloVe use different protocol amendments")
    protocol_amendment = json.loads(next(iter(amendments)))
    v1_failure = v1_heldout_failure(root)
    builder_path = Path(__file__).resolve()
    manifest = {
        "format_version": 1,
        "stage": "raw_regularized_v2_cross_dataset_offline_confirmation",
        "status": "materialization_eligible_online_qps_pending",
        "method_contract": {
            "method": METHOD,
            "mass_mode": MASS_MODE,
            "source": MASS_SOURCE,
            "transform": MASS_TRANSFORM,
            "estimator_version": ESTIMATOR_VERSION,
            "upper_only_phase_a": True,
            "l0_evaluation_only_after_owner_freeze": True,
            "multi_assignment_unchanged": True,
        },
        "protocol_amendment": protocol_amendment,
        "confirmation_contract": {
            "post_exploratory_not_preregistered": True,
            "v1_heldout_failure_retained": True,
            "topology_thresholds_unchanged": True,
            "old_raw_owner_byte_parity_required": True,
            "alternate_method_or_parameter_scan": False,
            "online_qps_only_confirmation": True,
            "online_qps_confirmation_required": True,
            "online_qps_confirmation_completed": False,
            "online_cluster_entered_by_this_stage": False,
        },
        "datasets": datasets,
        "v1_heldout_failure": v1_failure,
        "cross_dataset_identity_all_pass": all(
            record["identity_all_pass"] for record in datasets.values()
        ),
        "cross_dataset_topology_all_pass": all(
            record["topology_all_pass"] for record in datasets.values()
        ),
        "owner_parity_all_pass": all(
            record["owner_parity_all_pass"] for record in datasets.values()
        ),
        "materialization_eligible": all(
            record["materialization_eligible"] for record in datasets.values()
        ),
        "online_qps_confirmation_required": True,
        "builder": {
            "path": str(builder_path),
            "sha256": sha256_path(builder_path),
        },
    }
    if not all(
        manifest[key]
        for key in (
            "cross_dataset_identity_all_pass",
            "cross_dataset_topology_all_pass",
            "owner_parity_all_pass",
            "materialization_eligible",
        )
    ):
        raise ValueError("cross-dataset confirmation did not pass every offline gate")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "confirmation-manifest.json"
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    digest = sha256_path(output_path)
    checksum_path = output_path.with_name(output_path.name + ".sha256")
    checksum_path.write_text(digest + "\n", encoding="ascii")
    os.chmod(output_path, 0o444)
    os.chmod(checksum_path, 0o444)
    print(f"confirmation_manifest={output_path}")
    print(f"confirmation_manifest_sha256={digest}")


if __name__ == "__main__":
    main()
