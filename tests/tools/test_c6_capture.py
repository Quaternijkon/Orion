from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "experiments/c6/scripts/c6_capture.py"


def load_module():
    spec = importlib.util.spec_from_file_location("c6_capture", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_entry_point_encoding_and_result_decoding_are_inverse_for_copy_blocks():
    module = load_module()
    encoded = module.encode_entry_point(17, 3, mode="copy_block", block_size=101)
    assert encoded == 321
    assert module.decode_result_id(
        {"id": encoded}, mode="copy_block", block_size=101
    ) == 17
    assert module.encode_entry_point(17, 3, mode="raw", block_size=None) == 17
    assert module.encode_entry_point(17, 3, mode="raw_plus_one", block_size=None) == 18


def test_payload_source_id_decoding_fails_closed_when_payload_is_missing():
    module = load_module()
    assert module.decode_result_id(
        {"id": 500, "payload": {"source_id": 42}},
        mode="payload_source_id",
        block_size=None,
    ) == 42
    with pytest.raises(RuntimeError, match="payload.source_id"):
        module.decode_result_id(
            {"id": 500}, mode="payload_source_id", block_size=None
        )


def test_search_response_requires_exact_hardware_counters():
    module = load_module()
    payload = {
        "usage": {
            "hardware": {
                "cpu": 200 * 4 * 123,
                "graph_nodes_visited": 45,
                "cpu_time_us": 67,
                "cpu_wall_time_us": 70,
            }
        },
        "result": [
            {"id": 1, "score": 0.1, "payload": {"source_id": 0}},
            {"id": 2, "score": 0.2, "payload": {"source_id": 1}},
        ],
    }
    trace = module.parse_search_response(
        payload,
        query_id=4,
        shard_id=2,
        ef_search=32,
        dimension=200,
        latency_us=80.0,
        response_bytes=90,
        result_id_mode="payload_source_id",
        block_size=None,
    )
    assert trace.distance_computations == 123
    assert trace.nodes_visited == 45
    assert trace.result_ids == (0, 1)

    del payload["usage"]["hardware"]["cpu_time_us"]
    with pytest.raises(RuntimeError, match="cpu_time_us"):
        module.parse_search_response(
            payload,
            query_id=4,
            shard_id=2,
            ef_search=32,
            dimension=200,
            latency_us=80.0,
            response_bytes=90,
            result_id_mode="payload_source_id",
            block_size=None,
        )


def test_query_checksum_and_shard_key_map_are_strict(tmp_path):
    module = load_module()
    queries = np.asarray([[1.0, 2.0]], dtype=np.float32)
    assert module.query_bytes_sha256(queries) == module.hashlib.sha256(
        np.asarray(queries, dtype="<f4").tobytes()
    ).hexdigest()

    path = tmp_path / "shards.json"
    path.write_text(json.dumps({"0": "orion_0", "1": "orion_1"}))
    assert module.load_shard_key_map(path, {0, 1}) == {
        0: "orion_0",
        1: "orion_1",
    }
    with pytest.raises(ValueError, match="differ"):
        module.load_shard_key_map(path, {0, 1, 2})

    urls = tmp_path / "urls.json"
    urls.write_text(json.dumps({"0": "http://10.0.0.1:6333", "1": "http://10.0.0.2:6333/"}))
    assert module.load_shard_base_url_map(urls, {0, 1}) == {
        0: "http://10.0.0.1:6333",
        1: "http://10.0.0.2:6333",
    }


def test_resume_key_loader_rejects_duplicate_searches(tmp_path):
    module = load_module()
    path = tmp_path / "traces.jsonl"
    row = {"query_id": 0, "shard_id": 1, "ef_search": 8}
    path.write_text(json.dumps(row) + "\n")
    assert module.load_completed_keys(path) == {(0, 1, 8)}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        module.load_completed_keys(path)


def test_required_search_keys_adds_oracle_grid_and_exact_adaptive_efs():
    module = load_module()
    protocol = sys.modules["c6_protocol"]
    route = protocol.RouteQuery(
        query_id=0,
        navigation_candidate_count=3,
        candidate_shard_count=2,
        ranked_candidate_shards=(0, 1),
        adaptive_selected_shards=(0, 1),
        evidence=(
            protocol.ShardEvidence(0, 2, 1, 0.1, 0.2, (10, 11)),
            protocol.ShardEvidence(1, 1, 3, 0.3, 0.3, (20,)),
        ),
    )
    config = protocol.PolicyConfig("P3", alpha=3, beta=5)
    keys = module.required_search_keys(
        [route],
        ef_grid=(8, 16),
        oracle_high_ef=64,
        policy_configs=[config],
    )
    assert keys == {
        (0, 0, 8),
        (0, 0, 11),
        (0, 0, 16),
        (0, 0, 64),
        (0, 1, 8),
        (0, 1, 16),
        (0, 1, 64),
    }
