from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/method4_distributed_submission_report.py"
    spec = importlib.util.spec_from_file_location(
        "method4_distributed_submission_report", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_rows():
    rows = []
    for target in (0.95, 0.97):
        for method in ("Orion", "Simple KMeans", "Naive"):
            rows.append(
                {
                    "target_recall": target,
                    "method": method,
                    "recall_match_status": "strict",
                    "placement_valid": True,
                    "runtime_health_valid": True,
                    "transport_log_clean": True,
                    "transport_resources_valid": True,
                    "resource_contract_valid": True,
                    "stability_repeats": 3,
                    "eval_query_count": 3000,
                    "warmup_query_count": 500,
                    "batch_size": 200,
                    "cluster_peer_count": 4,
                    "cluster_shard_count": 46,
                    "cluster_active_shards": 46,
                    "controller_local_lower_shards": 0,
                    "fixed_ef_shard_chunk_size": 0,
                    "expected_steady_state_p2p_connections": 6,
                    "p2p_connections_start": 6,
                    "p2p_connections_end": 6,
                    "p2p_connections_delta": 0,
                    "worker_shard_counts": json.dumps(
                        {"worker-1": 16, "worker-2": 15, "worker-3": 15}
                    ),
                    "image_digest": "sha256:image",
                    "repository_commit": "commit",
                    "dataset_sha256": "dataset",
                    "resource_contract": "contract",
                    "source_main_run_dir": "/results/final-run",
                }
            )
    return rows


def test_submission_acceptance_gate_accepts_clean_strict_matrix():
    module = load_module()

    module.validate_submission_rows(valid_rows())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("recall_match_status", "nearest", "strict target match"),
        ("runtime_health_valid", False, "runtime_health_valid"),
        ("cluster_active_shards", 45, "cluster_active_shards"),
        ("p2p_connections_end", 7, "p2p_connections_end"),
        ("worker_shard_counts", '{"worker-1": 46}', "15/15/16"),
    ],
)
def test_submission_acceptance_gate_rejects_invalid_case(field, value, message):
    module = load_module()
    rows = valid_rows()
    rows[0][field] = value

    with pytest.raises(ValueError, match=message):
        module.validate_submission_rows(rows)


def test_submission_acceptance_gate_rejects_cross_case_provenance_drift():
    module = load_module()
    rows = copy.deepcopy(valid_rows())
    rows[-1]["image_digest"] = "sha256:different"

    with pytest.raises(ValueError, match="image_digest is missing or inconsistent"):
        module.validate_submission_rows(rows)
