from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/native_auto_shard_transport_probe.py"
    spec = importlib.util.spec_from_file_location(
        "native_auto_shard_transport_probe", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_telemetry_count_and_delta_support_wrapped_response():
    module = load_module()
    method = "/qdrant.PointsInternal/CoreSearchBatchByShardCompact"
    payload = {
        "result": {
            "requests": {
                "grpc": {
                    "responses": {
                        method: {
                            "0": {"count": 7},
                            "13": {"count": 2},
                        }
                    }
                }
            }
        }
    }

    assert module.telemetry_method_count(payload, method) == 9
    assert module.telemetry_method_status_counts(payload, method) == {
        "0": 7,
        "13": 2,
    }
    before = {"http://worker": {method: {"0": 7, "13": 2}}}
    after = {"http://worker": {method: {"0": 10, "13": 2}}}
    assert module.telemetry_delta(before, after) == {
        "http://worker": {method: {"0": 3, "13": 0}}
    }


def test_telemetry_delta_rejects_worker_restart_counter_reset():
    module = load_module()
    method = "/qdrant.PointsInternal/CoreSearchBatch"

    with pytest.raises(RuntimeError, match="likely restarted"):
        module.telemetry_delta(
            {"http://worker": {method: {"0": 9}}},
            {"http://worker": {method: {"0": 1}}},
        )


def test_canonical_result_proof_is_order_and_score_exact():
    module = load_module()
    first = module.canonical_result_proof(
        [[(0.5, 11), (0.25, 22)], [(0.75, 33)]]
    )
    identical = module.canonical_result_proof(
        [[(0.5, 11), (0.25, 22)], [(0.75, 33)]]
    )
    reordered = module.canonical_result_proof(
        [[(0.25, 22), (0.5, 11)], [(0.75, 33)]]
    )
    score_changed = module.canonical_result_proof(
        [[(0.5, 11), (0.25000003, 22)], [(0.75, 33)]]
    )

    assert first["ids_sha256"] == identical["ids_sha256"]
    assert first["ids_scores_sha256"] == identical["ids_scores_sha256"]
    assert first["ids_sha256"] != reordered["ids_sha256"]
    assert first["ids_scores_sha256"] != score_changed["ids_scores_sha256"]
    assert first["results"][0][0]["score_f32_le_hex"] == "0000003f"


def test_validate_result_rows_rejects_partial_duplicate_and_non_finite_rows():
    module = load_module()

    assert module.validate_result_rows(
        [[(0.5, 11), (0.25, 22)]], query_count=1, top_k=2
    ) == [2]
    with pytest.raises(RuntimeError, match="expected exactly 2"):
        module.validate_result_rows([[(0.5, 11)]], query_count=1, top_k=2)
    with pytest.raises(RuntimeError, match="duplicate point IDs"):
        module.validate_result_rows(
            [[(0.5, 11), (0.25, 11)]], query_count=1, top_k=2
        )
    with pytest.raises(RuntimeError, match="non-finite score"):
        module.validate_result_rows(
            [[(float("nan"), 11), (0.25, 22)]], query_count=1, top_k=2
        )


@pytest.mark.parametrize(
    "value",
    [
        ["--query-count", "0"],
        ["--query-offset", "-1"],
        ["--warmup-query-count", "-1"],
        ["--telemetry-method", "not-an-absolute-method"],
    ],
)
def test_validate_args_rejects_invalid_probe_ranges(tmp_path, value):
    module = load_module()
    args = module.parse_args(
        [
            "--base-url",
            "http://10.10.1.1:6333",
            "--topology",
            str(tmp_path / "topology.json"),
            "--deployment-manifest",
            str(tmp_path / "manifest.json"),
            "--run-id",
            "run-1",
            "--collection",
            "orion",
            "--hdf5-path",
            str(tmp_path / "dataset.hdf5"),
            "--output",
            str(tmp_path / "probe.json"),
            "--worker-url",
            "http://10.10.1.2:6333",
            *value,
        ]
    )

    with pytest.raises(ValueError):
        module.validate_args(args)


def test_output_must_be_new_and_outside_repository(tmp_path):
    module = load_module()

    with pytest.raises(ValueError, match="outside the repository"):
        module.output_path(REPO_ROOT / "results" / "probe.json")

    output = tmp_path / "probe.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        module.output_path(output)
