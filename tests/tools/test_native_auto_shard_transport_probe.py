from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def compact_wire_topology():
    return {
        "controller": {
            "role": "controller",
            "private_ip": "10.10.1.1",
        },
        "workers": [
            {
                "role": "qdrant_shard_1",
                "private_ip": "10.10.1.2",
            },
            {
                "role": "qdrant_shard_2",
                "private_ip": "10.10.1.3",
            },
            {
                "role": "qdrant_shard_3",
                "private_ip": "10.10.1.4",
            },
        ],
    }


def compact_wire_inspected(module, *, active_version="2", image_max_version="2"):
    inspected = {}
    for node in module.cluster.all_nodes(compact_wire_topology()):
        labels = {
            module.cluster.ORION_COMPACT_WIRE_MAX_VERSION_LABEL: image_max_version,
        }
        environment = []
        if node["role"] == "controller" and active_version == "2":
            labels[module.cluster.ORION_COMPACT_WIRE_VERSION_LABEL] = "2"
            environment.append(
                f"{module.cluster.ORION_COMPACT_WIRE_VERSION_ENV}=2"
            )
        inspected[node["role"]] = {
            "Config": {
                "Env": environment,
                "Labels": labels,
            }
        }
    return inspected


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


@pytest.mark.parametrize(
    ("active_version", "representation", "controller_metadata_present"),
    [
        ("1", "implicit-v1", False),
        ("2", "explicit-v2", True),
    ],
)
def test_compact_wire_runtime_evidence_records_controller_and_worker_identity(
    active_version, representation, controller_metadata_present
):
    module = load_module()
    topology = compact_wire_topology()
    inspected = compact_wire_inspected(module, active_version=active_version)

    evidence = module.compact_wire_runtime_evidence(
        topology,
        inspected,
        active_version=active_version,
        image_max_version="2",
    )

    assert evidence["active_version"] == active_version
    assert evidence["image_max_version"] == "2"
    assert evidence["controller"]["representation"] == representation
    assert evidence["controller"]["matches_active_version"] is True
    assert evidence["controller"]["image_max_version"] == "2"
    assert (
        evidence["controller"]["controller_only_container_label"]["present"]
        is controller_metadata_present
    )
    assert evidence["controller"]["controller_only_environment"]["values"] == (
        ["2"] if controller_metadata_present else []
    )
    assert evidence["workers_controller_only_metadata_absent"] is True
    assert set(evidence["workers"]) == {
        "qdrant_shard_1",
        "qdrant_shard_2",
        "qdrant_shard_3",
    }
    for worker in evidence["workers"].values():
        assert worker["resolved_version"] == "not_applicable"
        assert worker["controller_only_metadata_absent"] is True
        assert worker["controller_only_environment"]["values"] == []
        assert worker["controller_only_container_label"]["present"] is False
        assert worker["image_max_version"] == "2"


def test_compact_wire_runtime_evidence_rejects_worker_controller_metadata():
    module = load_module()
    topology = compact_wire_topology()
    inspected = compact_wire_inspected(module)
    worker = inspected["qdrant_shard_1"]
    worker["Config"]["Env"].append(
        f"{module.cluster.ORION_COMPACT_WIRE_VERSION_ENV}=2"
    )

    with pytest.raises(RuntimeError, match="worker contains controller-only"):
        module.compact_wire_runtime_evidence(
            topology,
            inspected,
            active_version="2",
            image_max_version="2",
        )


def test_deployment_context_uses_current_wire_aware_lifecycle_api(
    tmp_path, monkeypatch
):
    module = load_module()
    topology = compact_wire_topology()
    topology_path = tmp_path / "topology.json"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(module.cluster, "load_topology", lambda _path: topology)
    monkeypatch.setattr(module.cluster, "validate_run_id", lambda value: value)
    monkeypatch.setattr(
        module.cluster, "manifest_path", lambda _topology, _run_id: manifest_path
    )
    monkeypatch.setattr(
        module.cluster,
        "validate_peer_premerge_transition_manifest",
        lambda _topology, _run_id, _manifest: (
            "orion:test",
            "sha256:image",
            "enabled",
            "4",
            "2",
            "a" * 40,
        ),
    )
    monkeypatch.setattr(
        module.cluster,
        "manifest_image_orion_compact_wire_max_version",
        lambda _manifest: "2",
    )
    monkeypatch.setattr(
        module.cluster,
        "http_url",
        lambda node, _topology: f"http://{node['private_ip']}:6333",
    )

    inspected = compact_wire_inspected(module)

    def inspect_runtime(
        actual_topology,
        run_id,
        image_tag,
        image_id,
        mode,
        shards_per_rpc,
        wire_version,
        runtime_args,
    ):
        calls.append(wire_version)
        assert actual_topology is topology
        assert run_id == "run-1"
        assert image_tag == "orion:test"
        assert image_id == "sha256:image"
        assert mode == "enabled"
        assert shards_per_rpc == "4"
        assert isinstance(runtime_args, SimpleNamespace)
        return inspected, mode, shards_per_rpc, wire_version

    monkeypatch.setattr(
        module.cluster,
        "inspect_peer_premerge_transition_runtime",
        inspect_runtime,
    )
    monkeypatch.setattr(
        module,
        "validate_live_cluster",
        lambda _topology, _run_id, collection: (
            {"result": {"peer_id": 1}},
            {collection: {}},
        ),
    )
    args = SimpleNamespace(
        topology=str(topology_path),
        deployment_manifest=str(manifest_path),
        run_id="run-1",
        base_url="http://10.10.1.1:6333",
        worker_url=[],
        collection="orion",
    )

    context = module.load_deployment_context(args)
    after = module.capture_unchanged_deployment(context, "orion")
    module.assert_deployment_unchanged(context, after)

    assert calls == ["2", "2"]
    assert after["manifest_sha256"] == context["manifest_sha256"]
    assert context["orion_compact_wire_version"] == "2"
    assert context["orion_compact_wire_image_max_version"] == "2"
    assert after["orion_compact_wire_version"] == "2"
    assert args.worker_url == [
        "http://10.10.1.2:6333",
        "http://10.10.1.3:6333",
        "http://10.10.1.4:6333",
    ]

    manifest_path.write_text('{"changed": true}\n', encoding="utf-8")
    changed = module.capture_unchanged_deployment(context, "orion")
    assert changed["manifest_sha256"] != context["manifest_sha256"]
    with pytest.raises(RuntimeError, match="deployment manifest changed"):
        module.assert_deployment_unchanged(context, changed)
