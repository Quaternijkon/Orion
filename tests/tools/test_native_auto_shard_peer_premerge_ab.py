from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/native_auto_shard_peer_premerge_ab.py"
    spec = importlib.util.spec_from_file_location(
        "native_auto_shard_peer_premerge_ab", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_chunk_test_helpers():
    path = REPO_ROOT / "tests/tools/test_native_auto_shard_chunk_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "native_auto_shard_chunk_sweep_test_helpers", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def set_manifest_artifact(manifest, target):
    artifact_sha = ("c" if target == "r090" else "f") * 64
    generation = 7 if target == "r090" else 8
    policy = manifest["collection_info"]["config"]["auto_shard_policy"]
    policy["artifact_sha256"] = artifact_sha
    policy["generation"] = generation
    manifest["live_policy"]["artifact_sha256"] = artifact_sha
    manifest["live_policy"]["generation"] = generation
    manifest["artifact"]["sha256"] = artifact_sha
    manifest["artifact"]["generation"] = generation


def write_benchmark(
    module,
    helpers,
    leaf_root: Path,
    *,
    mode: str,
    chunk_value: str,
    collection: str,
    target: str,
    qps_mean: float,
    recall: float,
):
    benchmark_root = leaf_root / "benchmark"
    benchmark_root.mkdir(parents=True)
    qps_values = [qps_mean - 1.0, qps_mean, qps_mean + 1.0]
    latency_base = 160.0 if target == "r090" else 250.0
    p95_values = [latency_base, latency_base + 2.0, latency_base + 4.0]
    p99_values = [latency_base + 2.0, latency_base + 4.0, latency_base + 6.0]
    rows = [
        {
            "run": repeat,
            "method": module.chunk.EXPECTED_METHOD,
            "api": module.chunk.EXPECTED_API,
            "query_count": module.FORMAL_BENCHMARK_QUERY_COUNT,
            "top_k": module.chunk.TOP_K,
            "batch_size": module.chunk.BATCH_SIZE,
            "recall_at_k": recall,
            "qps": qps,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
        }
        for repeat, (qps, p95, p99) in enumerate(
            zip(qps_values, p95_values, p99_values, strict=True), start=1
        )
    ]
    with (benchmark_root / "stability_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": module.chunk.EXPECTED_METHOD,
        "collection": collection,
        "stability_runs": module.chunk.STABILITY_REPEATS,
        "final_metrics": {
            "method": module.chunk.EXPECTED_METHOD,
            "api": module.chunk.EXPECTED_API,
            "query_count": module.FORMAL_BENCHMARK_QUERY_COUNT,
            "top_k": module.chunk.TOP_K,
            "batch_size": module.chunk.BATCH_SIZE,
            "stability_repeats": module.chunk.STABILITY_REPEATS,
            "recall_at_k": statistics.fmean([recall] * 3),
            "recall_stdev": statistics.stdev([recall] * 3),
            "qps": statistics.fmean(qps_values),
            "qps_stdev": statistics.stdev(qps_values),
            "latency_p95_ms": statistics.fmean(p95_values),
            "latency_p95_ms_stdev": statistics.stdev(p95_values),
            "latency_p99_ms": statistics.fmean(p99_values),
            "latency_p99_ms_stdev": statistics.stdev(p99_values),
        },
    }
    helpers.write_json(benchmark_root / "summary.json", summary)
    manifest = helpers.benchmark_manifest(
        module.chunk, mode, chunk_value, collection
    )
    set_manifest_artifact(manifest, target)
    helpers.write_json(benchmark_root / "run_manifest.json", manifest)


def target_results(module, helpers, target):
    results = helpers.result_rows(module.chunk)
    if target == "r095":
        for row in results:
            for point in row:
                point["id"] += 100_000
    return results


def write_probe(
    module,
    helpers,
    leaf_root: Path,
    *,
    arm_name: str,
    mode: str,
    chunk_value: str,
    collection: str,
    target: str,
):
    results = target_results(module, helpers, target)
    ids_sha, ids_scores_sha = module.chunk.canonical_result_hashes(results)
    before, after, delta = helpers.telemetry_snapshots(
        module.chunk, mode, chunk_value
    )
    deployment = helpers.probe_deployment(module.chunk, mode, chunk_value)
    manifest_digest_prefix = {
        "enabled-a1": "1",
        "disabled-b": "2",
        "enabled-a2": "3",
    }[arm_name]
    deployment["manifest_sha256"] = manifest_digest_prefix * 64
    probe = {
        "schema_version": module.chunk.PROBE_SCHEMA_VERSION,
        "run_id": "native-v4",
        "collection": collection,
        "base_url": module.chunk.EXPECTED_CONTROLLER_HTTP_URL,
        "worker_urls": list(module.chunk.EXPECTED_WORKER_HTTP_URLS),
        "api": module.chunk.EXPECTED_API,
        "deployment": deployment,
        "dataset": helpers.fixed_dataset(module.chunk, probe=True),
        "vector_distance": "cosine",
        "vector_name": "",
        "query_dtype": "float32-le",
        "query_dimension": 200,
        "query_offset": 0,
        "query_count": module.chunk.PROBE_QUERY_COUNT,
        "warmup_query_count": module.chunk.PROBE_WARMUP_QUERY_COUNT,
        "warmup_query_sha256": "d" * 64,
        "top_k": module.chunk.TOP_K,
        "result_row_lengths": [module.chunk.TOP_K]
        * module.chunk.PROBE_QUERY_COUNT,
        "batch_size": module.chunk.BATCH_SIZE,
        "wall_s": 0.25,
        "qps": module.chunk.PROBE_QUERY_COUNT / 0.25,
        "query_sha256": "e" * 64,
        "request_contract": module.chunk.EXPECTED_PROBE_REQUEST_CONTRACT,
        "ids_sha256": ids_sha,
        "ids_scores_sha256": ids_scores_sha,
        "results": results,
        "telemetry_methods": list(module.chunk.TELEMETRY_METHODS),
        "telemetry_before": before,
        "telemetry_after": after,
        "telemetry_delta": delta,
    }
    helpers.write_json(leaf_root / "transport-probe.json", probe)


def write_sandwich(tmp_path: Path, module, helpers) -> Path:
    root = tmp_path / "sandwich"
    qps = {
        "r090": {"enabled-a1": 1000.0, "disabled-b": 1020.0, "enabled-a2": 990.0},
        "r095": {"enabled-a1": 800.0, "disabled-b": 820.0, "enabled-a2": 792.0},
    }
    recall = {"r090": 0.903, "r095": 0.952}
    collections = {
        "r090": "dist_native-v4_orion_r090_u48",
        "r095": "dist_native-v4_orion_r095_u112",
    }
    previous_query_count = module.chunk.BENCHMARK_QUERY_COUNT
    module.chunk.BENCHMARK_QUERY_COUNT = module.FORMAL_BENCHMARK_QUERY_COUNT
    try:
        for arm_name, _arm_label, mode, chunk_value in module.ARM_SETTINGS:
            for target, target_directory in module.TARGET_SETTINGS:
                leaf_root = root / arm_name / target_directory
                write_benchmark(
                    module,
                    helpers,
                    leaf_root,
                    mode=mode,
                    chunk_value=chunk_value,
                    collection=collections[target],
                    target=target,
                    qps_mean=qps[target][arm_name],
                    recall=recall[target],
                )
                write_probe(
                    module,
                    helpers,
                    leaf_root,
                    arm_name=arm_name,
                    mode=mode,
                    chunk_value=chunk_value,
                    collection=collections[target],
                    target=target,
                )
    finally:
        module.chunk.BENCHMARK_QUERY_COUNT = previous_query_count
    return root


def mutate_json(path: Path, mutate):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_analyze_valid_sandwich_writes_four_bound_artifacts(tmp_path):
    module = load_module()
    helpers = load_chunk_test_helpers()
    root = write_sandwich(tmp_path, module, helpers)
    output = tmp_path / "analysis"

    assert module.analyze(root, output) == output.resolve()
    assert sorted(path.name for path in output.iterdir()) == sorted(
        module.OUTPUT_FILES
    )

    with (output / "ab_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["target"], row["arm"]) for row in rows] == [
        ("r090", "A1"),
        ("r090", "B"),
        ("r090", "A2"),
        ("r095", "A1"),
        ("r095", "B"),
        ("r095", "A2"),
    ]
    assert {row["benchmark_query_count"] for row in rows} == {"3000"}
    assert {row["legacy_rpc_count"] for row in rows} == {"0"}

    transport = json.loads((output / "transport_equivalence.json").read_text())
    assert transport["equivalent"] is True
    for target in ("r090", "r095"):
        proof = transport["targets"][target]
        assert proof["equivalent"] is True
        assert proof["rpc_contract"]["A1"]["compact_total"] == 12
        assert proof["rpc_contract"]["B"]["ordinary_total"] == 46
        assert proof["rpc_contract"]["A2"]["compact_total"] == 12
        assert all(
            arm["ids_scores_equal"] for arm in proof["arms"].values()
        )

    drift = json.loads((output / "drift_analysis.json").read_text())
    assert drift["all_targets_within_limit"] is True
    assert drift["targets"]["r090"]["enabled_qps_mean"] == 995.0
    assert drift["targets"]["r090"]["b_qps_delta_pct_vs_enabled_mean"] == pytest.approx(
        (1020.0 / 995.0 - 1.0) * 100.0
    )

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["protocol"]["benchmark_query_count"] == 3000
    assert manifest["protocol"]["enabled_compact_rpc_total"] == 12
    assert manifest["protocol"]["disabled_ordinary_rpc_total"] == 46
    assert manifest["binding"]["collections"] == {
        "r090": "dist_native-v4_orion_r090_u48",
        "r095": "dist_native-v4_orion_r095_u112",
    }


@pytest.mark.parametrize("field", ["query_sha256", "warmup_query_sha256"])
def test_analyze_rejects_query_or_warmup_mismatch(tmp_path, field):
    module = load_module()
    helpers = load_chunk_test_helpers()
    root = write_sandwich(tmp_path, module, helpers)
    path = root / "enabled-a2/orion-r090/transport-probe.json"
    mutate_json(path, lambda payload: payload.__setitem__(field, "f" * 64))

    with pytest.raises(RuntimeError, match="query/warmup"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_exact_ids_and_float32_scores_mismatch(tmp_path):
    module = load_module()
    helpers = load_chunk_test_helpers()
    root = write_sandwich(tmp_path, module, helpers)
    path = root / "enabled-a2/orion-r095/transport-probe.json"

    def mutate(payload):
        payload["results"][0][0]["score_f32_le_hex"] = "0000003f"
        ids_sha, scores_sha = module.chunk.canonical_result_hashes(
            payload["results"]
        )
        payload["ids_sha256"] = ids_sha
        payload["ids_scores_sha256"] = scores_sha

    mutate_json(path, mutate)
    with pytest.raises(RuntimeError, match="float32-score proof"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_enabled_rpc_count_other_than_twelve(tmp_path):
    module = load_module()
    helpers = load_chunk_test_helpers()
    root = write_sandwich(tmp_path, module, helpers)
    path = root / "enabled-a1/orion-r090/transport-probe.json"

    def mutate(payload):
        worker = payload["worker_urls"][0]
        method = module.chunk.COMPACT_BY_SHARD
        payload["telemetry_after"][worker][method]["0"] += 1
        payload["telemetry_delta"][worker][method]["0"] += 1

    mutate_json(path, mutate)
    with pytest.raises(RuntimeError, match="compact RPC counts"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_a1_a2_qps_drift_above_five_percent(tmp_path):
    module = load_module()
    helpers = load_chunk_test_helpers()
    root = write_sandwich(tmp_path, module, helpers)
    arm_root = root / "enabled-a2/orion-r090"
    stability_path = arm_root / "benchmark/stability_runs.csv"
    with stability_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row, qps in zip(rows, (899.0, 900.0, 901.0), strict=True):
        row["qps"] = str(qps)
    with stability_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    helpers.rewrite_summary_from_rows(arm_root)

    with pytest.raises(RuntimeError, match="A1/A2 QPS drift"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_manifest_sha_change_between_targets_in_one_arm(tmp_path):
    module = load_module()
    helpers = load_chunk_test_helpers()
    root = write_sandwich(tmp_path, module, helpers)
    path = root / "disabled-b/orion-r095/transport-probe.json"
    mutate_json(
        path,
        lambda payload: payload["deployment"].__setitem__(
            "manifest_sha256", "9" * 64
        ),
    )

    with pytest.raises(RuntimeError, match="manifest SHA differs between targets"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_rejects_nonformal_benchmark_query_count(tmp_path):
    module = load_module()
    helpers = load_chunk_test_helpers()
    root = write_sandwich(tmp_path, module, helpers)
    path = root / "enabled-a1/orion-r090/benchmark/run_manifest.json"
    mutate_json(
        path,
        lambda payload: payload["parameters"].__setitem__(
            "eval_query_count", 1000
        ),
    )

    with pytest.raises(RuntimeError, match="benchmark parameter eval_query_count"):
        module.analyze(root, tmp_path / "analysis")


def test_analyze_refuses_to_overwrite_output_directory(tmp_path):
    module = load_module()
    helpers = load_chunk_test_helpers()
    root = write_sandwich(tmp_path, module, helpers)
    output = tmp_path / "analysis"
    output.mkdir()

    with pytest.raises(FileExistsError, match="overwrite"):
        module.analyze(root, output)
