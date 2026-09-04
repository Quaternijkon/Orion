#!/usr/bin/env python3
"""Freeze the post-exploratory BMR_10 dual-dataset selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance.run_offline_screen import sha256_path  # noqa: E402
from experiments.multi_assignment import budgeted_policy  # noqa: E402


EXPECTED_OWNERS = {
    "sift1m": "f00ccb35f6dba918c214a64e62449776ccd6a261362a7c6cbb07e69a0b19099b",
    "glove-200-angular": "0e42469f1fb30b76788f556834109562eddc1ba7cac12996b02d28938c5ea11d",
}


def load_frozen(path: Path, label: str) -> tuple[dict[str, Any], str]:
    digest = sha256_path(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != digest:
        raise ValueError(f"{label} sidecar mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sift-evaluation", required=True)
    parser.add_argument("--glove-evaluation", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def validate_evaluation(
    path: Path, dataset: str, query_scope: str
) -> tuple[dict[str, Any], str]:
    value, digest = load_frozen(path, f"{dataset} formal evaluation")
    expected = {
        "format_version": 1,
        "record_type": "fixed_cnbr_bmr10_formal_offline_evaluation",
        "status": "PASS",
        "dataset": dataset,
        "query_scope": query_scope,
        "all_gates_pass": True,
    }
    drift = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if drift:
        raise ValueError(f"{dataset} formal evaluation drifted: {drift}")
    boundary = value.get("causal_boundary")
    policy = value.get("policy")
    if not isinstance(boundary, dict) or not isinstance(policy, dict):
        raise ValueError(f"{dataset} formal evaluation lacks contracts")
    if boundary.get("owner_sha256") != EXPECTED_OWNERS[dataset]:
        raise ValueError(f"{dataset} C_CNBR owner checksum drifted")
    if boundary.get("owner_changed_between_arms") is not False:
        raise ValueError(f"{dataset} changed the frozen load-balancing owner")
    policy_expected = {
        "candidate_id": budgeted_policy.CANDIDATE_ID,
        "version": budgeted_policy.POLICY_VERSION,
        "extra_copy_budget_numerator": budgeted_policy.EXTRA_COPY_BUDGET_NUMERATOR,
        "extra_copy_budget_denominator": budgeted_policy.EXTRA_COPY_BUDGET_DENOMINATOR,
        "score": budgeted_policy.SCORE,
        "at_most_two_copies": True,
        "navigation_evidence_only": True,
    }
    if any(policy.get(key) != expected_value for key, expected_value in policy_expected.items()):
        raise ValueError(f"{dataset} BMR_10 policy contract drifted")
    candidate = value.get("candidate")
    baseline = value.get("baseline")
    if not isinstance(candidate, dict) or not isinstance(baseline, dict):
        raise ValueError(f"{dataset} metrics are missing")
    if not (
        float(candidate["expansion_ratio"]) <= 1.10
        and float(candidate["expansion_ratio"]) < float(baseline["expansion_ratio"])
    ):
        raise ValueError(f"{dataset} low-expansion gate no longer passes")
    source_code = value.get("source_code")
    current_sources = {
        "evaluator": REPO_ROOT
        / "experiments/multi_assignment/evaluate_budgeted_candidate.py",
        "policy": Path(budgeted_policy.__file__).resolve(),
        "protocol": REPO_ROOT / "experiments/multi_assignment/PROTOCOL.md",
        "addendum": REPO_ROOT
        / "experiments/multi_assignment/BUDGETED_REPLICATION_ADDENDUM.md",
    }
    if not isinstance(source_code, dict) or set(source_code) != set(current_sources):
        raise ValueError(f"{dataset} source-code set drifted")
    for name, current in current_sources.items():
        record = source_code[name]
        if (
            record.get("sha256") != sha256_path(current)
            or record.get("size_bytes") != current.stat().st_size
        ):
            raise ValueError(f"{dataset} source code drifted: {name}")
    return value, digest


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    sift_path = Path(args.sift_evaluation).expanduser().resolve()
    glove_path = Path(args.glove_evaluation).expanduser().resolve()
    sift, sift_sha = validate_evaluation(sift_path, "sift1m", "full")
    glove, glove_sha = validate_evaluation(
        glove_path, "glove-200-angular", "prefix_tuning_only"
    )
    if int(glove.get("query_count", 0)) != 1000:
        raise ValueError("GloVe selection must use exactly 1,000 tuning queries")

    exploration_path = Path(
        glove["inputs"]["exploration_manifest"]
    ).expanduser().resolve()
    if sha256_path(exploration_path) != glove["inputs"]["exploration_manifest_sha256"]:
        raise ValueError("GloVe exploration checksum drifted")
    exploration = json.loads(exploration_path.read_text(encoding="utf-8"))
    old_gate_pass = [
        row["policy"]
        for row in exploration["rows"][1:]
        if row.get("all_offline_gates_pass") is True
    ]
    exploratory_name = "budget_0.10_secondary_proxy_mass_max"
    if old_gate_pass != [exploratory_name]:
        raise ValueError(
            "BMR_10 is no longer the unique GloVe tuning pass in the frozen family"
        )

    source_files = {
        "synthesizer": Path(__file__).resolve(),
        "policy": Path(budgeted_policy.__file__).resolve(),
        "addendum": REPO_ROOT
        / "experiments/multi_assignment/BUDGETED_REPLICATION_ADDENDUM.md",
    }
    result = {
        "format_version": 1,
        "record_type": "fixed_cnbr_bmr10_selection",
        "status": "PASS",
        "selection_status": "post_exploratory_sift_and_glove_tuning_selection",
        "strict_glove_holdout_claim": False,
        "glove_online_measurement_queries_seen": False,
        "selected_candidate": budgeted_policy.CANDIDATE_ID,
        "fallback": "C_CNBR_current_all_max",
        "no_fallback_retuning": True,
        "selection_basis": (
            "unique_glove_tuning_pass_in_fixed_budget_score_family_and_"
            "absolute_expansion_at_most_1p10_on_both_datasets"
        ),
        "inputs": {
            "sift1m": {
                "formal_evaluation": str(sift_path),
                "formal_evaluation_sha256": sift_sha,
            },
            "glove-200-angular": {
                "formal_evaluation": str(glove_path),
                "formal_evaluation_sha256": glove_sha,
                "query_count": 1000,
                "exploration_manifest": str(exploration_path),
                "exploration_manifest_sha256": glove["inputs"][
                    "exploration_manifest_sha256"
                ],
                "old_gate_unique_pass": old_gate_pass,
            },
        },
        "rows": {
            "sift1m": {
                "owner_sha256": sift["causal_boundary"]["owner_sha256"],
                "baseline": sift["baseline"],
                "candidate": sift["candidate"],
                "gates": sift["gates"],
            },
            "glove-200-angular": {
                "owner_sha256": glove["causal_boundary"]["owner_sha256"],
                "baseline": glove["baseline"],
                "candidate": glove["candidate"],
                "gates": glove["gates"],
            },
        },
        "source_code": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_path(path.resolve()),
                "size_bytes": path.resolve().stat().st_size,
            }
            for name, path in source_files.items()
        },
    }
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "selection-manifest.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256_path(manifest_path)
    sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    sidecar.write_text(digest + "\n", encoding="ascii")
    os.chmod(manifest_path, 0o444)
    os.chmod(sidecar, 0o444)
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
