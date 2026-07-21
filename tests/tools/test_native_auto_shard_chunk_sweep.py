from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import struct
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/native_auto_shard_chunk_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "native_auto_shard_chunk_sweep", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def result_rows(module):
    return [
        [
            {
                "id": query_index * module.TOP_K + result_index,
                "score_f32_le_hex": struct.pack(
                    "<f", 1.0 - result_index / (module.TOP_K * 2)
                ).hex(),
            }
            for result_index in range(module.TOP_K)
        ]
        for query_index in range(module.PROBE_QUERY_COUNT)
    ]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def rewrite_summary_from_rows(arm_root: Path):
    stability_path = arm_root / "benchmark/stability_runs.csv"
    with stability_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    qps = [float(row["qps"]) for row in rows]
    recall = [float(row["recall_at_k"]) for row in rows]
    p95 = [float(row["latency_p95_ms"]) for row in rows]
    p99 = [float(row["latency_p99_ms"]) for row in rows]
    summary_path = arm_root / "benchmark/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["final_metrics"].update(
        {
            "qps": statistics.fmean(qps),
            "qps_stdev": statistics.stdev(qps),
            "recall_at_k": statistics.fmean(recall),
            "recall_stdev": statistics.stdev(recall),
            "latency_p95_ms": statistics.fmean(p95),
            "latency_p95_ms_stdev": statistics.stdev(p95),
            "latency_p99_ms": statistics.fmean(p99),
            "latency_p99_ms_stdev": statistics.stdev(p99),
        }
    )
    write_json(summary_path, summary)


def fixed_dataset(module, *, probe=False):
    value = {
        "path": "/datasets/glove-200-angular.hdf5",
        "sha256": module.EXPECTED_DATASET_SHA256,
        "size_bytes": module.EXPECTED_DATASET_SIZE_BYTES,
    }
    if probe:
        value["hdf5_shapes"] = {
            name: list(shape) for name, shape in module.EXPECTED_DATASET_SHAPES.items()
        }
    else:
        value.update(
            {
                "train_shape": list(module.EXPECTED_DATASET_SHAPES["train"]),
                "test_shape": list(module.EXPECTED_DATASET_SHAPES["test"]),
                "neighbors_shape": list(
                    module.EXPECTED_DATASET_SHAPES["neighbors"]
                ),
            }
        )
    return value


def fixed_placement(module):
    peer_ids = (201, 202, 203)
    assigned_peers = (
        [peer_ids[0]] * module.EXPECTED_DISABLED_RPC_COUNTS[0]
        + [peer_ids[1]] * module.EXPECTED_DISABLED_RPC_COUNTS[1]
        + [peer_ids[2]] * module.EXPECTED_DISABLED_RPC_COUNTS[2]
    )
    return peer_ids, {
        str(shard_id): peer_id for shard_id, peer_id in enumerate(assigned_peers)
    }


def benchmark_manifest(module, mode, chunk, collection):
    controller_peer_id = 100
    worker_peer_ids, placement = fixed_placement(module)
    artifact_sha = "f" * 64
    image_id = "sha256:" + "b" * 64
    image_tag = "orion:test"
    commit = "a" * 40
    points_count = 1_394_406
    peers = {
        str(controller_peer_id): module.EXPECTED_PEER_URIS[0],
        **{
            str(peer_id): peer_uri
            for peer_id, peer_uri in zip(
                worker_peer_ids, module.EXPECTED_PEER_URIS[1:], strict=True
            )
        },
    }
    remote_shards = [
        {"shard_id": int(shard_id), "peer_id": peer_id, "state": "Active"}
        for shard_id, peer_id in placement.items()
    ]
    shards_per_worker = {
        str(peer_id): sum(value == peer_id for value in placement.values())
        for peer_id in worker_peer_ids
    }
    nodes = [
        {
            "role": "controller",
            "private_ip": module.EXPECTED_CONTROLLER_IP,
            "cpuset": "0-7",
            "image_id": image_id,
            "peer_premerge_mode": mode,
            "peer_premerge_shards_per_rpc": chunk,
        },
        *[
            {
                "role": f"qdrant_shard_{index}",
                "private_ip": private_ip,
                "cpuset": "0-19",
                "image_id": image_id,
                "peer_premerge_mode": "not_applicable",
                "peer_premerge_shards_per_rpc": "not_applicable",
            }
            for index, private_ip in enumerate(module.EXPECTED_WORKER_IPS, start=1)
        ],
    ]
    provenance = {
        "method": "orion",
        "logical_point_count": 1_183_514,
        "physical_point_count": points_count,
        "shard_count": module.EXPECTED_SHARD_COUNT,
        "vector_schema": {
            "datatype": "float32",
            "dimension": 200,
            "distance": "Cosine",
            "vector_name": "",
        },
    }
    return {
        "schema_version": module.BENCHMARK_SCHEMA_VERSION,
        "method": module.EXPECTED_METHOD,
        "api": module.EXPECTED_API,
        "base_url": module.EXPECTED_CONTROLLER_HTTP_URL,
        "collection": collection,
        "request_contract": dict(module.EXPECTED_REQUEST_CONTRACT),
        "dataset": fixed_dataset(module),
        "repository": {"commit": commit, "tracked_dirty": False},
        "repository_binding": {
            "benchmark_commit": commit,
            "deployment_commit": commit,
            "tracked_dirty": False,
        },
        "deployment": {
            "path": "/experiment/manifest.json",
            "image": {"id": image_id, "tag": image_tag},
            "repository": {"commit": commit, "tracked_dirty": False},
            "nodes": nodes,
        },
        "process_affinity": list(module.EXPECTED_PROCESS_AFFINITY),
        "topology": {
            "benchmark_client_cpuset": "8-19",
            "controller_uri": module.EXPECTED_PEER_URIS[0],
            "worker_uris": list(module.EXPECTED_PEER_URIS[1:]),
            "workers": [
                {"private_ip": private_ip, "cpuset": "0-19"}
                for private_ip in module.EXPECTED_WORKER_IPS
            ],
        },
        "cluster_preflight": {
            "peer_count": 4,
            "consensus_thread_status": "working",
            "pending_operations": 0,
            "message_send_failures": {},
            "controller_peer_id": controller_peer_id,
            "peer_id": controller_peer_id,
            "worker_peer_ids": list(worker_peer_ids),
            "peers": peers,
        },
        "collection_cluster": {
            "peer_id": controller_peer_id,
            "shard_count": module.EXPECTED_SHARD_COUNT,
            "local_shards": [],
            "remote_shards": remote_shards,
            "shard_transfers": [],
        },
        "collection_info": {
            "config": {
                "params": {
                    "shard_number": module.EXPECTED_SHARD_COUNT,
                    "replication_factor": module.EXPECTED_REPLICATION_FACTOR,
                    "sharding_method": "auto",
                    "vectors": {"size": 200, "distance": "Cosine"},
                },
                "auto_shard_policy": {
                    "type": "orion",
                    "artifact_sha256": artifact_sha,
                    "generation": 8,
                },
                "metadata": {
                    "native_auto_shard_prepare": {
                        "schema_version": 2,
                        "provenance_sha256": "e" * 64,
                        "provenance": provenance,
                    }
                },
            },
            "status": "green",
            "optimizer_status": "ok",
            "update_queue": {"length": 0},
            "points_count": points_count,
            "indexed_vectors_count": points_count,
        },
        "placement_proof": {
            "valid": True,
            "controller_peer_id": controller_peer_id,
            "shard_count": module.EXPECTED_SHARD_COUNT,
            "replication_factor": module.EXPECTED_REPLICATION_FACTOR,
            "shard_transfers": [],
            "placement": placement,
            "expected_placement": dict(placement),
            "shards_per_worker": shards_per_worker,
        },
        "live_policy": {
            "type": "orion",
            "artifact_sha256": artifact_sha,
            "generation": 8,
        },
        "artifact": {
            "status": "verified",
            "sha256": artifact_sha,
            "generation": 8,
            "shard_count": module.EXPECTED_SHARD_COUNT,
            "physical_point_count": points_count,
        },
        "indexing_readiness": {
            "fully_indexed": True,
            "completion_mode": "fully_indexed",
            "status": "green",
            "optimizer_status": "ok",
            "points_count": points_count,
            "indexed_vectors_count": points_count,
            "shard_transfers": [],
            "update_queue": {"length": 0},
        },
        "parameters": {
            "method": module.EXPECTED_METHOD,
            "api": module.EXPECTED_API,
            "base_url": module.EXPECTED_CONTROLLER_HTTP_URL,
            "collection": collection,
            "warmup_query_count": module.BENCHMARK_WARMUP_QUERY_COUNT,
            "eval_query_count": module.BENCHMARK_QUERY_COUNT,
            "top_k": module.TOP_K,
            "batch_size": module.BATCH_SIZE,
            "stability_repeats": module.STABILITY_REPEATS,
            "vector_distance": "cosine",
            "vector_name": "",
            "hnsw_ef": None,
            "orion_route_trace": False,
            "write_per_query_metrics": True,
        },
    }


def telemetry_snapshots(module, mode, chunk):
    disabled_counts = dict(
        zip(
            module.EXPECTED_WORKER_HTTP_URLS,
            module.EXPECTED_DISABLED_RPC_COUNTS,
            strict=True,
        )
    )
    before = {}
    after = {}
    delta = {}
    for worker in module.EXPECTED_WORKER_HTTP_URLS:
        active_method = (
            module.COMPACT_BY_SHARD if mode == "enabled" else module.CORE_SEARCH_BATCH
        )
        increment = (
            module.expected_compact_calls(disabled_counts[worker], chunk)
            if mode == "enabled"
            else disabled_counts[worker]
        )
        before[worker] = {method: {} for method in module.TELEMETRY_METHODS}
        after[worker] = {method: {} for method in module.TELEMETRY_METHODS}
        delta[worker] = {method: {} for method in module.TELEMETRY_METHODS}
        before[worker][active_method] = {"0": 100}
        after[worker][active_method] = {"0": 100 + increment}
        delta[worker][active_method] = {"0": increment}
    return before, after, delta


def probe_deployment(module, mode, chunk):
    image_id = "sha256:" + "b" * 64
    image_tag = "orion:test"
    containers = {
        role: {
            "container_id": f"container-{role}",
            "cpuset": "0-7" if role == "controller" else "0-19",
            "image_id": image_id,
            "image_tag": image_tag,
            "network_mode": "host",
            "running": True,
        }
        for role in ["controller", "qdrant_shard_1", "qdrant_shard_2", "qdrant_shard_3"]
    }
    cluster = {
        "consensus_thread_status": {
            "consensus_thread_status": "working",
            "last_update": "now",
        },
        "message_send_failures": {},
        "peer_count": 4,
        "peer_id": 100,
        "peer_uris": list(module.EXPECTED_PEER_URIS),
        "pending_operations": 0,
    }
    return {
        "topology_path": "tools/distributed/cloudlab_orion_4node.json",
        "manifest_path": "/experiment/manifest.json",
        "manifest_sha256": "1" * 64,
        "commit": "a" * 40,
        "image_tag": image_tag,
        "image_id": image_id,
        "peer_premerge_mode": mode,
        "peer_premerge_shards_per_rpc": chunk,
        "containers_before": containers,
        "containers_after": json.loads(json.dumps(containers)),
        "controller_peer_id_before": 100,
        "controller_peer_id_after": 100,
        "cluster_before": cluster,
        "cluster_after": json.loads(json.dumps(cluster)),
        "collection_placement_before_sha256": "2" * 64,
        "collection_placement_after_sha256": "2" * 64,
    }


def write_sweep(tmp_path: Path, module) -> Path:
    root = tmp_path / "sweep"
    qps_means = {
        "enabled-all": 773.5,
        "enabled-16": 782.2,
        "enabled-8": 795.0,
        "enabled-4": 805.1,
        "enabled-2": 800.8,
        "enabled-1": 800.2,
        "disabled-all": 821.2,
    }
    results = result_rows(module)
    ids_sha, ids_scores_sha = module.canonical_result_hashes(results)
    for arm, mode, chunk in module.ARM_SETTINGS:
        arm_root = root / arm
        benchmark = arm_root / "benchmark"
        benchmark.mkdir(parents=True)
        qps_values = [qps_means[arm] - 1.0, qps_means[arm], qps_means[arm] + 1.0]
        p95_values = [250.0, 252.0, 254.0]
        p99_values = [252.0, 254.0, 256.0]
        rows = [
            {
                "run": repeat,
                "method": "orion",
                "api": "search",
                "query_count": module.BENCHMARK_QUERY_COUNT,
                "top_k": module.TOP_K,
                "batch_size": module.BATCH_SIZE,
                "recall_at_k": 0.955,
                "qps": qps,
                "latency_p95_ms": p95,
                "latency_p99_ms": p99,
            }
            for repeat, (qps, p95, p99) in enumerate(
                zip(qps_values, p95_values, p99_values, strict=True), start=1
            )
        ]
        with (benchmark / "stability_runs.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "method": module.EXPECTED_METHOD,
            "collection": "orion-r095",
            "stability_runs": module.STABILITY_REPEATS,
            "final_metrics": {
                "method": module.EXPECTED_METHOD,
                "api": module.EXPECTED_API,
                "query_count": module.BENCHMARK_QUERY_COUNT,
                "top_k": module.TOP_K,
                "batch_size": module.BATCH_SIZE,
                "stability_repeats": module.STABILITY_REPEATS,
                "recall_at_k": statistics.fmean([0.955, 0.955, 0.955]),
                "recall_stdev": statistics.stdev([0.955, 0.955, 0.955]),
                "qps": statistics.fmean(qps_values),
                "qps_stdev": statistics.stdev(qps_values),
                "latency_p95_ms": statistics.fmean(p95_values),
                "latency_p95_ms_stdev": statistics.stdev(p95_values),
                "latency_p99_ms": statistics.fmean(p99_values),
                "latency_p99_ms_stdev": statistics.stdev(p99_values),
            },
        }
        write_json(benchmark / "summary.json", summary)
        write_json(
            benchmark / "run_manifest.json",
            benchmark_manifest(module, mode, chunk, "orion-r095"),
        )

        telemetry_before, telemetry_after, telemetry_delta = telemetry_snapshots(
            module, mode, chunk
        )
        probe = {
            "schema_version": module.PROBE_SCHEMA_VERSION,
            "run_id": "run-v4",
            "collection": "orion-r095",
            "base_url": module.EXPECTED_CONTROLLER_HTTP_URL,
            "worker_urls": list(module.EXPECTED_WORKER_HTTP_URLS),
            "api": module.EXPECTED_API,
            "deployment": probe_deployment(module, mode, chunk),
            "dataset": fixed_dataset(module, probe=True),
            "vector_distance": "cosine",
            "vector_name": "",
            "query_dtype": "float32-le",
            "query_dimension": 200,
            "query_offset": 0,
            "query_count": module.PROBE_QUERY_COUNT,
            "warmup_query_count": module.PROBE_WARMUP_QUERY_COUNT,
            "warmup_query_sha256": "d" * 64,
            "top_k": module.TOP_K,
            "result_row_lengths": [module.TOP_K] * module.PROBE_QUERY_COUNT,
            "batch_size": module.BATCH_SIZE,
            "wall_s": 0.25,
            "qps": module.PROBE_QUERY_COUNT / 0.25,
            "query_sha256": "e" * 64,
            "request_contract": module.EXPECTED_PROBE_REQUEST_CONTRACT,
            "ids_sha256": ids_sha,
            "ids_scores_sha256": ids_scores_sha,
            "results": results,
            "telemetry_methods": list(module.TELEMETRY_METHODS),
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
            "telemetry_delta": telemetry_delta,
        }
        write_json(arm_root / "transport-probe.json", probe)
    return root


def mutate_json(path: Path, mutate):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    write_json(path, payload)


def test_analyze_valid_sweep_writes_proofs_and_selects_larger_top_two_chunk(
    tmp_path,
):
    module = load_module()
    root = write_sweep(tmp_path, module)
    output = tmp_path / "analysis"

    assert module.analyze(root, output) == output.resolve()
    assert sorted(path.name for path in output.iterdir()) == [
        "chunk_selection.json",
        "chunk_sweep_summary.csv",
        "transport_equivalence.json",
    ]
    selection = json.loads((output / "chunk_selection.json").read_text())
    assert selection["raw_best_arm"] == "enabled-4"
    assert selection["runner_up_arm"] == "enabled-2"
    assert selection["tie_applied"] is True
    assert selection["selected_arm"] == "enabled-4"
    assert selection["selected_chunk"] == "4"

    equivalence = json.loads((output / "transport_equivalence.json").read_text())
    assert equivalence["equivalent"] is True
    assert equivalence["strict_protocol"]["benchmark_warmup_query_count"] == 500
    assert equivalence["strict_protocol"]["probe_query_count"] == 200
    assert equivalence["strict_protocol"]["disabled_rpc_counts"] == [15, 16, 15]
    rpc = equivalence["rpc_relationship"]
    assert rpc["disabled"]["ordinary_total"] == 46
    assert [rpc["enabled"][arm]["actual_total"] for arm in module.ENABLED_ARMS] == [
        3,
        3,
        6,
        12,
        24,
        46,
    ]
    with (output / "chunk_sweep_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert [row["arm"] for row in rows] == [item[0] for item in module.ARM_SETTINGS]
    assert [row["arm"] for row in rows if row["selected"] == "True"] == [
        "enabled-4"
    ]
    assert {row["latency_p95_ms"] for row in rows} == {"252.0"}
    assert {row["latency_p99_ms"] for row in rows} == {"254.0"}


@pytest.mark.parametrize("field", ["query_sha256", "warmup_query_sha256"])
def test_analyze_rejects_query_or_warmup_mismatch(tmp_path, field):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/transport-probe.json"
    mutate_json(path, lambda payload: payload.__setitem__(field, "f" * 64))

    with pytest.raises(RuntimeError, match="query/warmup/results"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_exact_result_mismatch_even_with_valid_internal_hashes(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/transport-probe.json"

    def mutate(payload):
        payload["results"][0][0]["id"] = 999
        ids_sha, scores_sha = module.canonical_result_hashes(payload["results"])
        payload["ids_sha256"] = ids_sha
        payload["ids_scores_sha256"] = scores_sha

    mutate_json(path, mutate)
    with pytest.raises(RuntimeError, match="query/warmup/results"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_non_top_k_result_rows(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/transport-probe.json"

    def mutate(payload):
        payload["results"][0].pop()
        payload["result_row_lengths"][0] = 1
        ids_sha, scores_sha = module.canonical_result_hashes(payload["results"])
        payload["ids_sha256"] = ids_sha
        payload["ids_scores_sha256"] = scores_sha

    mutate_json(path, mutate)
    with pytest.raises(RuntimeError, match="top-k"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_exact_recall_mismatch(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    arm_root = root / "enabled-all"
    stability = arm_root / "benchmark/stability_runs.csv"
    with stability.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["recall_at_k"] = "0.954"
    with stability.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rewrite_summary_from_rows(arm_root)

    with pytest.raises(RuntimeError, match="exact recall"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_qps_cv_above_five_percent(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    arm_root = root / "enabled-all"
    stability = arm_root / "benchmark/stability_runs.csv"
    with stability.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row, qps in zip(rows, (80.0, 100.0, 120.0), strict=True):
        row["qps"] = str(qps)
    with stability.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rewrite_summary_from_rows(arm_root)

    with pytest.raises(RuntimeError, match="QPS CV"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_nonzero_grpc_status(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/transport-probe.json"

    def mutate(payload):
        worker = payload["worker_urls"][0]
        payload["telemetry_delta"][worker][module.COMPACT_BY_SHARD]["13"] = 1

    mutate_json(path, mutate)
    with pytest.raises(RuntimeError, match="non-zero gRPC status"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_compact_rpc_count_not_derived_from_disabled(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-8/transport-probe.json"

    def mutate(payload):
        worker = payload["worker_urls"][0]
        payload["telemetry_delta"][worker][module.COMPACT_BY_SHARD]["0"] += 1
        payload["telemetry_after"][worker][module.COMPACT_BY_SHARD]["0"] += 1

    mutate_json(path, mutate)
    with pytest.raises(RuntimeError, match="disabled-derived expectation"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_refuses_to_overwrite_output_directory(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    output = tmp_path / "analysis"
    output.mkdir()

    with pytest.raises(FileExistsError, match="overwrite"):
        module.analyze(root, output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warmup_query_count", 499),
        ("eval_query_count", 999),
        ("top_k", 9),
        ("batch_size", 100),
        ("stability_repeats", 2),
        ("api", "query"),
    ],
)
def test_analyze_rejects_nonfixed_benchmark_manifest_parameters(
    tmp_path, field, value
):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/benchmark/run_manifest.json"
    mutate_json(path, lambda payload: payload["parameters"].__setitem__(field, value))

    with pytest.raises(RuntimeError, match="benchmark parameter"):
        module.analyze(root, tmp_path / "analysis")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_count", 199),
        ("warmup_query_count", 499),
        ("top_k", 9),
        ("batch_size", 100),
        ("query_offset", 1),
    ],
)
def test_analyze_rejects_nonfixed_probe_parameters(tmp_path, field, value):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/transport-probe.json"
    mutate_json(path, lambda payload: payload.__setitem__(field, value))

    with pytest.raises(RuntimeError, match="probe|query_count|top_k"):
        module.analyze(root, tmp_path / "analysis")


@pytest.mark.parametrize(
    ("relative_path", "expected_schema"),
    [
        ("enabled-all/benchmark/run_manifest.json", 1),
        ("enabled-all/transport-probe.json", 2),
    ],
)
def test_analyze_rejects_wrong_manifest_schema_version(
    tmp_path, relative_path, expected_schema
):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / relative_path
    mutate_json(path, lambda payload: payload.__setitem__("schema_version", 999))

    with pytest.raises(RuntimeError, match=f"schema_version.*{expected_schema}"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_missing_collection_provenance(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/benchmark/run_manifest.json"

    def mutate(payload):
        del payload["collection_info"]["config"]["metadata"][
            "native_auto_shard_prepare"
        ]["provenance"]

    mutate_json(path, mutate)
    with pytest.raises(RuntimeError, match="collection provenance"):
        module.analyze(root, tmp_path / "analysis")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pending_operations", 99, "pending_operations"),
        ("peer_count", 1, "peer_count"),
        ("message_send_failures", {"peer": 1}, "message_send_failures"),
    ],
)
def test_analyze_rejects_unhealthy_probe_cluster_evidence(
    tmp_path, field, value, message
):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/transport-probe.json"
    mutate_json(
        path,
        lambda payload: payload["deployment"]["cluster_after"].__setitem__(
            field, value
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_probe_placement_change(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/transport-probe.json"
    mutate_json(
        path,
        lambda payload: payload["deployment"].__setitem__(
            "collection_placement_after_sha256", "3" * 64
        ),
    )

    with pytest.raises(RuntimeError, match="placement hash"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_replication_factor_other_than_one(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/benchmark/run_manifest.json"
    mutate_json(
        path,
        lambda payload: payload["collection_info"]["config"]["params"].__setitem__(
            "replication_factor", 2
        ),
    )

    with pytest.raises(RuntimeError, match="replication_factor"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_controller_lower_local_shard(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/benchmark/run_manifest.json"
    mutate_json(
        path,
        lambda payload: payload["collection_cluster"]["local_shards"].append(
            {"shard_id": 0, "state": "Active"}
        ),
    )

    with pytest.raises(RuntimeError, match="controller local shards"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_nonactive_remote_shard(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/benchmark/run_manifest.json"
    mutate_json(
        path,
        lambda payload: payload["collection_cluster"]["remote_shards"][0].__setitem__(
            "state", "Dead"
        ),
    )

    with pytest.raises(RuntimeError, match="remote shard 0 state"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_45_ordinary_shard_rpcs(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "disabled-all/transport-probe.json"

    def mutate(payload):
        worker = payload["worker_urls"][0]
        payload["telemetry_after"][worker][module.CORE_SEARCH_BATCH]["0"] -= 1
        payload["telemetry_delta"][worker][module.CORE_SEARCH_BATCH]["0"] -= 1

    mutate_json(path, mutate)
    with pytest.raises(RuntimeError, match="ordinary RPC counts"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_fewer_than_three_probe_workers(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/transport-probe.json"
    mutate_json(path, lambda payload: payload["worker_urls"].pop())

    with pytest.raises(RuntimeError, match="exactly 3 unique worker"):
        module.analyze(root, tmp_path / "analysis")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("collection", "other-collection", "collection mismatch"),
        ("api", "query", "probe API"),
    ],
)
def test_analyze_rejects_probe_collection_or_api_binding(
    tmp_path, field, value, message
):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/transport-probe.json"
    mutate_json(path, lambda payload: payload.__setitem__(field, value))

    with pytest.raises(RuntimeError, match=message):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_probe_dataset_binding(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/transport-probe.json"
    mutate_json(
        path, lambda payload: payload["dataset"].__setitem__("sha256", "0" * 64)
    )

    with pytest.raises(RuntimeError, match="probe dataset sha256"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_probe_commit_binding(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/transport-probe.json"
    mutate_json(
        path,
        lambda payload: payload["deployment"].__setitem__("commit", "c" * 40),
    )

    with pytest.raises(RuntimeError, match="commit mismatch"):
        module.analyze(root, tmp_path / "analysis")


@pytest.mark.parametrize("field", ["latency_p95_ms", "latency_p99_ms"])
def test_analyze_rejects_summary_latency_not_recomputed_from_runs(tmp_path, field):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/benchmark/summary.json"

    def mutate(payload):
        payload["final_metrics"][field] += 1.0

    mutate_json(path, mutate)
    with pytest.raises(RuntimeError, match=field):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_telemetry_delta_not_derived_from_snapshots(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-4/transport-probe.json"

    def mutate(payload):
        worker = payload["worker_urls"][0]
        payload["telemetry_after"][worker][module.COMPACT_BY_SHARD]["0"] += 1

    mutate_json(path, mutate)
    with pytest.raises(RuntimeError, match="telemetry_delta"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_benchmark_cpu_affinity_change(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/benchmark/run_manifest.json"
    mutate_json(path, lambda payload: payload["process_affinity"].pop())

    with pytest.raises(RuntimeError, match="process_affinity"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_client_side_routing_hint_contract(tmp_path):
    module = load_module()
    root = write_sweep(tmp_path, module)
    path = root / "enabled-all/benchmark/run_manifest.json"
    mutate_json(
        path,
        lambda payload: payload["request_contract"].__setitem__(
            "shard_selector", True
        ),
    )

    with pytest.raises(RuntimeError, match="request contract"):
        module.analyze(root, tmp_path / "analysis")


def choice_arms(module, qps_by_arm):
    arms = {}
    for arm, _mode, chunk in module.ARM_SETTINGS:
        qps = qps_by_arm.get(arm, 50.0)
        arms[arm] = {
            "arm": arm,
            "benchmark": {"qps_mean": qps, "qps_stdev": 0.0, "qps_cv": 0.0},
            "probe": {"chunk": chunk},
        }
    return arms


def test_choose_chunk_applies_tie_at_exactly_two_percent():
    module = load_module()
    selection = module.choose_chunk(
        choice_arms(module, {"enabled-1": 100.0, "enabled-16": 98.0})
    )

    assert selection["top_two_relative_gap"] == pytest.approx(0.02)
    assert selection["tie_applied"] is True
    assert selection["selected_arm"] == "enabled-16"


def test_choose_chunk_does_not_apply_tie_at_two_point_one_percent():
    module = load_module()
    selection = module.choose_chunk(
        choice_arms(module, {"enabled-1": 100.0, "enabled-all": 97.9})
    )

    assert selection["top_two_relative_gap"] == pytest.approx(0.021)
    assert selection["tie_applied"] is False
    assert selection["selected_arm"] == "enabled-1"


def test_choose_chunk_ignores_third_place_even_when_it_is_within_two_percent():
    module = load_module()
    selection = module.choose_chunk(
        choice_arms(
            module,
            {"enabled-1": 100.0, "enabled-2": 99.5, "enabled-all": 99.4},
        )
    )

    assert selection["tie_applied"] is True
    assert selection["runner_up_arm"] == "enabled-2"
    assert selection["ranked_enabled"][2]["arm"] == "enabled-all"
    assert selection["ranked_enabled"][2]["relative_gap_to_best"] < 0.02
    assert selection["selected_arm"] == "enabled-2"
