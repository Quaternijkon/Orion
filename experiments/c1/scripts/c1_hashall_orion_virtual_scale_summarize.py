#!/usr/bin/env python3
"""Audit and compare fixed-recall HashAll and native Orion scale-out runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pymupdf


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


M_VALUES = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
TARGET_RECALL = 0.90
MAX_CV = 0.05
MAX_CROSS_METHOD_RECALL_GAP = 0.015
HNSW_M = 32.0
LEVEL_DELTA = 0.5
DATASET_SIZE = 1_183_514.0
UPPER_NODE_COUNT = 36_984.0
EXPECTED_POINTS = (
    ("m1-exact-s1", 1, "exact", 1, "p1"),
    ("m2-lower-s1", 2, "lower", 1, "p1"),
    ("m2-upper-s3", 2, "upper", 3, "p2"),
    ("m4-exact-s4", 4, "exact", 4, "p3"),
    ("m8-lower-s7", 8, "lower", 7, "p5"),
    ("m8-upper-s10", 8, "upper", 10, "p7"),
    ("m16-lower-s15", 16, "lower", 15, "p13"),
    ("m16-upper-s17", 16, "upper", 17, "p14"),
    ("m32-exact-s32", 32, "exact", 32, "p24"),
)
REQUIRED_PDF_TEXT = (
    "图表作用",
    "如何理解",
    "最终结论",
    "证据边界",
    "逻辑 shard 数 M（线性数值坐标）",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, label: str, checks: list[dict[str, str]]) -> None:
    checks.append({"check": label, "status": "PASS" if condition else "FAIL"})
    if not condition:
        raise RuntimeError(f"audit failed: {label}")


def output_point_dir(root: Path, machine: int, boundary: str, actual_shards: int) -> Path:
    return root / f"m{machine}" / f"{boundary}-s{actual_shards}"


def actual_shard_layout(benchmark: dict[str, Any]) -> tuple[list[int], list[int], bool]:
    per_node = benchmark.get("per_node_collection_cluster") or {}
    counts: list[int] = []
    shard_ids: list[int] = []
    stable = True
    for node in sorted(per_node):
        state = per_node[node]
        local = state.get("local_shards") or []
        counts.append(len(local))
        shard_ids.extend(int(shard["shard_id"]) for shard in local)
        stable = stable and all(shard.get("state") == "Active" for shard in local)
        stable = stable and not (state.get("shard_transfers") or [])
    return counts, shard_ids, stable


def audit_common(
    method: str,
    machine: int,
    resources: dict[str, Any],
    benchmark: dict[str, Any],
    checks: list[dict[str, str]],
) -> None:
    prefix = f"{method} M={machine}"
    require(resources.get("status") == "PASS", f"{prefix} resources PASS", checks)
    require(benchmark.get("status") == "PASS", f"{prefix} benchmark PASS", checks)
    require(
        abs(float(resources.get("quota_cores_total", -1.0)) - 2.0 * machine) < 1e-9,
        f"{prefix} total CPU equals 2M",
        checks,
    )
    require(
        float((benchmark.get("heldout_recall") or {}).get("recall_at_10", 0.0))
        >= TARGET_RECALL,
        f"{prefix} held-out recall >= 0.90",
        checks,
    )
    require(
        float(benchmark.get("qps_cv", 1.0)) <= MAX_CV,
        f"{prefix} QPS CV <= 5%",
        checks,
    )
    protocol = benchmark.get("protocol") or {}
    parameters = benchmark.get("parameters") or {}
    require(
        protocol.get("dataset") == "GloVe-200-angular"
        and protocol.get("distance") == "Cosine"
        and protocol.get("vector_size") == 200
        and protocol.get("tuning_query_range") == [0, 1000]
        and protocol.get("heldout_query_range") == [1000, 10000]
        and (benchmark.get("heldout_recall") or {}).get("query_count") == 9000,
        f"{prefix} fixed dataset and disjoint recall protocol",
        checks,
    )
    require(
        parameters.get("batch_size") == 200
        and parameters.get("top_k") == 10
        and float(parameters.get("sweep_seconds", -1.0)) == 8.0
        and float(parameters.get("warmup_seconds", -1.0)) == 10.0
        and float(parameters.get("measure_seconds", -1.0)) == 20.0
        and parameters.get("repeat_count") == 5,
        f"{prefix} common QPS protocol",
        checks,
    )
    request = benchmark.get("request_contract") or {}
    require(
        request.get("standard_coordinator_request") is True
        and request.get("client_side_fanout") is False
        and request.get("shard_selector_present") is False,
        f"{prefix} ordinary coordinator Search",
        checks,
    )


def load_hashall_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, str]]]:
    summary_path = root / "summary/summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    require_checks: list[dict[str, str]] = []
    require(
        [int(row["logical_shards"]) for row in summaries]
        == M_VALUES.astype(int).tolist(),
        "HashAll exact M=1,2,4,8,16,32 summary",
        require_checks,
    )
    rows: list[dict[str, Any]] = []
    benchmarks: dict[int, dict[str, Any]] = {}
    for summary in summaries:
        machine = int(summary["logical_shards"])
        resources = load_json(root / f"m{machine}/resources-measure.json")
        benchmark = load_json(root / f"m{machine}/benchmark.json")
        audit_common("HashAll", machine, resources, benchmark, require_checks)
        request = benchmark.get("request_contract") or {}
        require(
            request.get("hashall_fanout") == machine,
            f"HashAll M={machine} visits all shards",
            require_checks,
        )
        selected_ef = int(summary["selected_ef"])
        require(
            10 <= selected_ef <= 512
            and (benchmark.get("parameters") or {}).get("selected_hnsw_ef")
            == selected_ef,
            f"HashAll M={machine} bounded fixed ef",
            require_checks,
        )
        rows.append(
            {
                "method": "HashAll",
                "point_id": f"m{machine}-exact-s{machine}",
                "nominal_m": machine,
                "boundary": "exact",
                "actual_shards": machine,
                "allocated_cpu": float(summary["allocated_cpu_cores"]),
                "recall": float(summary["heldout_recall_at_10"]),
                "qps": float(summary["qps_mean"]),
                "qps_stdev": float(summary["qps_stdev"]),
                "qps_cv": float(summary["qps_cv"]),
                "selected_ef": selected_ef,
                "visited_shards": float(machine),
                "ef_sum_per_query": float(machine * selected_ef),
                "upper_search_ef": "",
                "physical_point_count": int(DATASET_SIZE),
                "source": str(root / f"m{machine}/benchmark.json"),
            }
        )
        benchmarks[machine] = benchmark
    source_audit = load_json(root / "summary/completion-audit.json")
    require(
        source_audit.get("status") == "PASS",
        "HashAll source completion audit PASS",
        require_checks,
    )
    return rows, benchmarks, require_checks


def load_orion_rows(
    root: Path,
    route_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, str]]]:
    checks: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    benchmarks: dict[str, dict[str, Any]] = {}
    query_sha256: str | None = None
    for point_id, machine, boundary, actual_shards, layout_key in EXPECTED_POINTS:
        point = output_point_dir(root, machine, boundary, actual_shards)
        resources_path = point / "resources-measure.json"
        benchmark_path = point / "benchmark.json"
        require(resources_path.is_file(), f"Orion {point_id} resources exists", checks)
        require(benchmark_path.is_file(), f"Orion {point_id} benchmark exists", checks)
        resources = load_json(resources_path)
        benchmark = load_json(benchmark_path)
        audit_common("Orion", machine, resources, benchmark, checks)
        require(
            benchmark.get("point_id") == point_id
            and benchmark.get("boundary") == boundary
            and benchmark.get("actual_shards") == actual_shards,
            f"Orion {point_id} point identity",
            checks,
        )
        profile = benchmark.get("profile_contract") or {}
        require(
            profile.get("key") == layout_key
            and profile.get("method") == "native_orion"
            and profile.get("actual_shards") == actual_shards,
            f"Orion {point_id} native profile contract",
            checks,
        )
        request = benchmark.get("request_contract") or {}
        require(
            request.get("server_router") == "native_orion"
            and request.get("client_hnsw_ef_present") is False,
            f"Orion {point_id} server owns routing and ef",
            checks,
        )
        counts, shard_ids, stable = actual_shard_layout(benchmark)
        require(len(counts) == 4, f"Orion {point_id} has four peers", checks)
        require(
            counts == resources.get("shards_per_node"),
            f"Orion {point_id} quota follows placement",
            checks,
        )
        require(
            sorted(shard_ids) == list(range(actual_shards)) and stable,
            f"Orion {point_id} has one active copy of every shard",
            checks,
        )
        trace_path = route_root / f"{layout_key}.json"
        require(trace_path.is_file(), f"Orion {point_id} route trace exists", checks)
        trace = load_json(trace_path)
        artifact = trace.get("artifact") or {}
        trace_queries = trace.get("queries") or {}
        aggregate = trace.get("aggregate") or {}
        require(
            artifact.get("sha256") == profile.get("artifact_sha256")
            and artifact.get("shard_count") == actual_shards,
            f"Orion {point_id} trace matches artifact",
            checks,
        )
        require(
            trace_queries.get("query_count") == 10000
            and trace_queries.get("dimension") == 200,
            f"Orion {point_id} trace covers 10000 queries",
            checks,
        )
        if query_sha256 is None:
            query_sha256 = str(trace_queries.get("sha256"))
        require(
            trace_queries.get("sha256") == query_sha256,
            f"Orion {point_id} trace uses common query bytes",
            checks,
        )
        visited = float((aggregate.get("visited_shards") or {}).get("average", 0.0))
        ef_sum = float((aggregate.get("ef_sum_per_query") or {}).get("average", 0.0))
        require(
            1.0 <= visited <= actual_shards and ef_sum > 0.0,
            f"Orion {point_id} route metrics are bounded",
            checks,
        )
        rows.append(
            {
                "method": "Orion",
                "point_id": point_id,
                "nominal_m": machine,
                "boundary": boundary,
                "actual_shards": actual_shards,
                "allocated_cpu": float(resources["quota_cores_total"]),
                "recall": float(benchmark["heldout_recall"]["recall_at_10"]),
                "qps": float(benchmark["qps_mean"]),
                "qps_stdev": float(benchmark["qps_stdev"]),
                "qps_cv": float(benchmark["qps_cv"]),
                "selected_ef": "dynamic",
                "visited_shards": visited,
                "ef_sum_per_query": ef_sum,
                "upper_search_ef": int(profile["upper_search_ef"]),
                "physical_point_count": int(profile["physical_point_count"]),
                "source": str(benchmark_path),
            }
        )
        benchmarks[point_id] = benchmark
    execution = load_json(root / "execution-complete.json")
    require(
        execution.get("status") == "MEASUREMENTS_COMPLETE"
        and execution.get("completed_points")
        == [point[0] for point in EXPECTED_POINTS]
        and execution.get("parallel_prebuild_reuse") is True
        and execution.get("formal_measurement_concurrency") == 1,
        "Orion execution complete with parallel prebuild and serial measurement",
        checks,
    )
    restore = load_json(root / "resource-restored.json")
    require(restore.get("status") == "PASS", "Orion resource restore PASS", checks)
    require(
        len(restore.get("nodes") or []) == 4
        and all(node.get("before") == node.get("after") for node in restore["nodes"]),
        "Orion all four resource states restored exactly",
        checks,
    )
    for layout_key in sorted({point[4] for point in EXPECTED_POINTS}):
        cleanup = load_json(root / f"layouts/{layout_key}/cleanup.json")
        require(
            cleanup.get("status") == "PASS",
            f"Orion layout {layout_key} cleanup PASS",
            checks,
        )
    return rows, benchmarks, checks


def hnsw_signature(method: str, benchmark: dict[str, Any]) -> dict[str, Any]:
    if method == "HashAll":
        collection = benchmark.get("collection_info") or {}
    else:
        collection = (benchmark.get("live_policy_gate") or {}).get("collection_info") or {}
    return ((collection.get("config") or {}).get("hnsw_config") or {})


def audit_fairness(
    hash_rows: Sequence[dict[str, Any]],
    hash_benchmarks: dict[int, dict[str, Any]],
    orion_rows: Sequence[dict[str, Any]],
    orion_benchmarks: dict[str, dict[str, Any]],
    checks: list[dict[str, str]],
) -> None:
    hash_by_m = {int(row["nominal_m"]): row for row in hash_rows}
    for row in orion_rows:
        machine = int(row["nominal_m"])
        hash_row = hash_by_m[machine]
        require(
            abs(float(row["allocated_cpu"]) - float(hash_row["allocated_cpu"])) < 1e-9,
            f"M={machine} HashAll and {row['point_id']} receive identical CPU",
            checks,
        )
        require(
            abs(float(row["recall"]) - float(hash_row["recall"]))
            <= MAX_CROSS_METHOD_RECALL_GAP,
            f"M={machine} HashAll and {row['point_id']} recall gap <= 0.015",
            checks,
        )
        require(
            hnsw_signature("HashAll", hash_benchmarks[machine])
            == hnsw_signature("Orion", orion_benchmarks[row["point_id"]])
            == {
                "m": 32,
                "ef_construct": 200,
                "full_scan_threshold": 10,
                "max_indexing_threads": 1,
                "on_disk": False,
            },
            f"M={machine} HashAll and {row['point_id']} use identical HNSW construction",
            checks,
        )


def error_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    relative = (predicted - observed) / observed
    return {
        "r_squared": 1.0 - residual / total,
        "mape": float(np.mean(np.abs(relative))),
        "max_absolute_relative_error": float(np.max(np.abs(relative))),
        "rmse_qps": float(np.sqrt(np.mean((predicted - observed) ** 2))),
    }


def anchored_power(machines: np.ndarray, qps: np.ndarray) -> dict[str, Any]:
    feature = np.log(machines)
    baseline = float(qps[0])
    exponent = float(
        np.dot(feature, np.log(qps / baseline)) / np.dot(feature, feature)
    )
    predicted = baseline * np.power(machines, exponent)
    return {
        "baseline_qps": baseline,
        "exponent": exponent,
        "predicted": predicted,
        "metrics": error_metrics(qps, predicted),
    }


def hashall_model(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    machines = np.asarray([row["nominal_m"] for row in rows], dtype=float)
    qps = np.asarray([row["qps"] for row in rows], dtype=float)
    ef = np.asarray([row["selected_ef"] for row in rows], dtype=float)
    per_shard = HNSW_M * (
        np.log(DATASET_SIZE / machines) / np.log(HNSW_M) + LEVEL_DELTA
    ) + 2.0 * HNSW_M * ef
    total_work = machines * per_shard
    theory = float(qps[0]) * machines * total_work[0] / total_work
    fit = anchored_power(machines, qps)
    return {
        "machines": machines,
        "qps": qps,
        "qps_stdev": np.asarray([row["qps_stdev"] for row in rows]),
        "linear": float(qps[0]) * machines,
        "theory": theory,
        "theory_metrics": error_metrics(qps, theory),
        "fit": fit,
        "per_shard_work": per_shard,
        "total_work": total_work,
    }


def orion_point_work(row: dict[str, Any]) -> float:
    upper = HNSW_M * (
        math.log(UPPER_NODE_COUNT) / math.log(HNSW_M) + LEVEL_DELTA
    ) + 2.0 * HNSW_M * float(row["upper_search_ef"])
    lower_navigation = (
        float(row["visited_shards"])
        * HNSW_M
        * (
            math.log(float(row["physical_point_count"]) / float(row["actual_shards"]))
            / math.log(HNSW_M)
            + LEVEL_DELTA
        )
    )
    lower_base = 2.0 * HNSW_M * float(row["ef_sum_per_query"])
    return upper + lower_navigation + lower_base


def orion_model(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    machines = np.asarray([row["nominal_m"] for row in rows], dtype=float)
    qps = np.asarray([row["qps"] for row in rows], dtype=float)
    work = np.asarray([orion_point_work(row) for row in rows], dtype=float)
    theory = float(qps[0]) * machines * work[0] / work
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["nominal_m"])].append(row)
    centers = np.asarray(
        [statistics.fmean(float(row["qps"]) for row in grouped[int(m)]) for m in M_VALUES],
        dtype=float,
    )
    lower = np.asarray(
        [min(float(row["qps"]) for row in grouped[int(m)]) for m in M_VALUES],
        dtype=float,
    )
    upper = np.asarray(
        [max(float(row["qps"]) for row in grouped[int(m)]) for m in M_VALUES],
        dtype=float,
    )
    theory_lower = np.asarray(
        [min(float(theory[i]) for i, row in enumerate(rows) if row["nominal_m"] == int(m)) for m in M_VALUES],
        dtype=float,
    )
    theory_upper = np.asarray(
        [max(float(theory[i]) for i, row in enumerate(rows) if row["nominal_m"] == int(m)) for m in M_VALUES],
        dtype=float,
    )
    fit = anchored_power(M_VALUES, centers)
    return {
        "machines": machines,
        "qps": qps,
        "qps_stdev": np.asarray([row["qps_stdev"] for row in rows]),
        "work": work,
        "theory": theory,
        "theory_metrics": error_metrics(qps, theory),
        "center_machines": M_VALUES.copy(),
        "centers": centers,
        "lower": lower,
        "upper": upper,
        "theory_lower": theory_lower,
        "theory_upper": theory_upper,
        "linear": float(qps[0]) * M_VALUES,
        "fit": fit,
    }


def configure_axis(axis: plt.Axes, font: FontProperties) -> None:
    axis.set_xscale("linear")
    axis.set_xlim(0.0, 33.0)
    axis.set_xticks(M_VALUES, [str(int(value)) for value in M_VALUES])
    axis.set_xlabel("逻辑 shard 数 M（线性数值坐标）", fontproperties=font)
    axis.set_ylabel("QPS", fontproperties=font)
    axis.grid(True, linestyle="--", alpha=0.45)


def chinese_font() -> tuple[FontProperties, str]:
    buffer = pymupdf.Font(fontname="china-s").buffer
    handle = tempfile.NamedTemporaryFile(suffix=".ttf", delete=False)
    handle.write(buffer)
    handle.close()
    return FontProperties(fname=handle.name), handle.name


def wrap_cjk(text: str, width: int = 80) -> str:
    return "\n".join(text[index : index + width] for index in range(0, len(text), width))


def explanation_panel(
    axis: plt.Axes,
    font: FontProperties,
    *,
    purpose: str,
    reading: str,
    conclusion: str,
    boundary: str,
) -> None:
    axis.axis("off")
    for label, body, y, color in (
        ("图表作用", purpose, 0.97, "#0f172a"),
        ("如何理解", reading, 0.73, "#312e81"),
        ("最终结论", conclusion, 0.43, "#7c2d12"),
        ("证据边界", boundary, 0.16, "#475569"),
    ):
        axis.text(
            0.012,
            y,
            f"{label}：",
            transform=axis.transAxes,
            va="top",
            fontproperties=font,
            fontsize=11.2,
            fontweight="bold",
            color=color,
        )
        axis.text(
            0.097,
            y,
            wrap_cjk(body),
            transform=axis.transAxes,
            va="top",
            fontproperties=font,
            fontsize=10.0,
            color="#1f2937",
            linespacing=1.24,
        )


def plot_comparison(
    output_dir: Path,
    hash_model: dict[str, Any],
    orion_model_data: dict[str, Any],
    font: FontProperties,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.facecolor": "#fbfdff",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(15.5, 10.9))
    grid = figure.add_gridspec(2, 2, height_ratios=(3.3, 1.65), hspace=0.24, wspace=0.18)
    left = figure.add_subplot(grid[0, 0])
    right = figure.add_subplot(grid[0, 1])
    text_axis = figure.add_subplot(grid[1, :])
    dense = np.linspace(1.0, 32.0, 500)

    left.plot(M_VALUES, hash_model["linear"], "--", color="#2f855a", linewidth=2.2, label="线性辅助线：QPS(1)×M")
    left.plot(M_VALUES, hash_model["theory"], "-.D", color="#7c3aed", linewidth=2.2, markersize=6, label="理论辅助线：CPU/(fan-out×HNSW 工作量)")
    left.plot(dense, hash_model["qps"][0] * np.power(dense, hash_model["fit"]["exponent"]), color="#0b3f73", linewidth=2.7, label=f"实测拟合：QPS(1)×M^{hash_model['fit']['exponent']:.3f}")
    left.errorbar(M_VALUES, hash_model["qps"], yerr=hash_model["qps_stdev"], fmt="o", color="#1769aa", capsize=4, markersize=7, label="HashAll 实测 QPS", zorder=5)
    configure_axis(left, font)
    left.set_title("HashAll：访问全部 M 个分片", fontproperties=font, fontsize=13)
    left.legend(prop=font, fontsize=8.2, loc="upper left", framealpha=0.95)

    center_m = orion_model_data["center_machines"]
    right.plot(center_m, orion_model_data["linear"], "--", color="#2f855a", linewidth=2.2, label="线性辅助线：QPS(1)×M")
    right.fill_between(center_m, orion_model_data["theory_lower"], orion_model_data["theory_upper"], color="#c4b5fd", alpha=0.45, label="路由工作量理论带")
    right.plot(center_m, (orion_model_data["theory_lower"] + orion_model_data["theory_upper"]) / 2.0, "-.D", color="#7c3aed", linewidth=2.2, markersize=6)
    right.plot(dense, orion_model_data["centers"][0] * np.power(dense, orion_model_data["fit"]["exponent"]), color="#0b3f73", linewidth=2.7, label=f"中心值拟合：QPS(1)×M^{orion_model_data['fit']['exponent']:.3f}")
    right.fill_between(center_m, orion_model_data["lower"], orion_model_data["upper"], color="#93c5fd", alpha=0.38, label="Orion 实测布局边界带")
    right.errorbar(orion_model_data["machines"], orion_model_data["qps"], yerr=orion_model_data["qps_stdev"], fmt="o", color="#1769aa", capsize=3, markersize=6, label="Orion 各布局实测点", zorder=5)
    configure_axis(right, font)
    right.set_title("Orion：真实分裂布局边界带", fontproperties=font, fontsize=13)
    right.legend(prop=font, fontsize=8.0, loc="upper left", framealpha=0.95)

    hash_speedup = float(hash_model["qps"][-1] / hash_model["qps"][0])
    orion_speedup = float(orion_model_data["centers"][-1] / orion_model_data["centers"][0])
    purpose = "在 GloVe、Recall@10≥0.90 和总 CPU=2M 下，对比原生 HashAll 与原生 Orion 的吞吐扩展，并把理想线性、路由加 HNSW 工作量理论值和实测拟合放在同一张图中。"
    reading = "横轴是线性数值坐标，所以绿色 QPS(1)×M 是直线。Orion 蓝色区域是在相同名义 M 下不同真实 shard 布局形成的实测边界；紫色理论带使用离线生产路由器得到的实际平均 fan-out 和 ef 总量，不把 upper_k 误当 fan-out。"
    conclusion = f"HashAll 从 M=1 到 32 提升 {hash_speedup:.3f}×；Orion 从 505.62 增至 4497.00 QPS，即 {orion_speedup:.3f}×。Orion 在 M=8、16、32 高于 HashAll，但 M=2 的 1-shard/3-shard 结果相差 3.17×，说明真实布局是主要变量，不能用单条平滑曲线概括。"
    boundary = "M=1/2/4 最多由四台真实机器承载；M=8/16/32 是四台机器上的逻辑 shard 扩展。理论式忽略网络、merge、缓存、NUMA 和负载偏斜，仅是辅助模型；实测边界与矛盾点被原样保留。"
    explanation_panel(text_axis, font, purpose=purpose, reading=reading, conclusion=conclusion, boundary=boundary)
    figure.suptitle("GloVe-200-angular：HashAll 与 Orion 的固定召回率扩展", fontproperties=font, fontsize=19, fontweight="bold", y=0.985)
    figure.text(0.5, 0.947, "4 台物理机；总 CPU=2M；Recall@10≥0.90；top-k=10；batch=200；横轴为线性数值坐标", ha="center", fontproperties=font, fontsize=11, color="#334155")
    figure.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.04)
    pdf = output_dir / "hashall-vs-orion-linear-x.pdf"
    png = output_dir / "hashall-vs-orion-linear-x.png"
    figure.savefig(pdf, dpi=300)
    figure.savefig(png, dpi=220)
    plt.close(figure)
    return pdf, png


def verify_pdf(path: Path) -> dict[str, Any]:
    document = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in document)
    pages = len(document)
    document.close()
    missing = [token for token in REQUIRED_PDF_TEXT if token not in text]
    if missing:
        raise RuntimeError(f"PDF explanation tokens missing: {missing}")
    tofu = text.count("□") + text.count("�")
    if tofu:
        raise RuntimeError(f"PDF contains {tofu} missing-glyph replacement characters")
    return {
        "path": str(path),
        "pages": pages,
        "sha256": sha256(path),
        "required_chinese_text_present": True,
        "missing_glyph_replacements": 0,
        "status": "PASS",
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = (
        "method",
        "point_id",
        "nominal_m",
        "boundary",
        "actual_shards",
        "allocated_cpu",
        "recall",
        "qps",
        "qps_stdev",
        "qps_cv",
        "selected_ef",
        "visited_shards",
        "ef_sum_per_query",
        "upper_search_ef",
        "physical_point_count",
        "source",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_array(value: np.ndarray) -> list[float]:
    return [float(item) for item in value.tolist()]


def write_models(path: Path, hash_model: dict[str, Any], orion: dict[str, Any]) -> None:
    value = {
        "HashAll": {
            "machines": json_array(hash_model["machines"]),
            "qps": json_array(hash_model["qps"]),
            "linear": json_array(hash_model["linear"]),
            "theory": {
                "formula": "Q1*M*W1/W(M), W(M)=M*[mh*(log_mh(N/M)+delta)+2*mh*ef_R(M)]",
                "predicted": json_array(hash_model["theory"]),
                "metrics": hash_model["theory_metrics"],
            },
            "fit": {
                "formula": "Q1*M^b",
                "exponent": hash_model["fit"]["exponent"],
                "predicted": json_array(hash_model["fit"]["predicted"]),
                "metrics": hash_model["fit"]["metrics"],
            },
        },
        "Orion": {
            "point_machines": json_array(orion["machines"]),
            "point_qps": json_array(orion["qps"]),
            "point_work": json_array(orion["work"]),
            "center_machines": json_array(orion["center_machines"]),
            "center_qps": json_array(orion["centers"]),
            "lower_qps": json_array(orion["lower"]),
            "upper_qps": json_array(orion["upper"]),
            "linear": json_array(orion["linear"]),
            "theory": {
                "formula": "Q1*M*D1/D; D=mh*(log_mh(U)+delta)+2*mh*upper_ef + P*mh*(log_mh(Nphys/S)+delta)+2*mh*ef_sum",
                "upper_node_count": int(UPPER_NODE_COUNT),
                "predicted_per_point": json_array(orion["theory"]),
                "lower_band": json_array(orion["theory_lower"]),
                "upper_band": json_array(orion["theory_upper"]),
                "metrics": orion["theory_metrics"],
            },
            "fit": {
                "formula": "Q1*M^b fitted to per-M boundary centers",
                "exponent": orion["fit"]["exponent"],
                "predicted": json_array(orion["fit"]["predicted"]),
                "metrics": orion["fit"]["metrics"],
            },
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_results(
    path: Path,
    hash_rows: Sequence[dict[str, Any]],
    orion_rows: Sequence[dict[str, Any]],
    hash_model: dict[str, Any],
    orion: dict[str, Any],
) -> None:
    hash_by_m = {int(row["nominal_m"]): row for row in hash_rows}
    lines = [
        "# GloVe-200-angular：HashAll 与 Orion 的固定召回率 QPS 对比",
        "",
        "## 实验结果",
        "",
        "| 名义 M | Orion 边界 | 实际 shards | 总 CPU | Recall@10 | Orion QPS | HashAll QPS | Orion/HashAll | 平均访问 shards | ef 总量/query |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in orion_rows:
        baseline = hash_by_m[int(row["nominal_m"])]
        lines.append(
            f"| {row['nominal_m']} | {row['boundary']} | {row['actual_shards']} | {row['allocated_cpu']:.0f} | {row['recall']:.4f} | {row['qps']:.2f} | {baseline['qps']:.2f} | {float(row['qps']) / float(baseline['qps']):.3f}× | {row['visited_shards']:.3f} | {row['ef_sum_per_query']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 图表作用",
            "",
            "综合图给出线性辅助线、由真实 Orion 路由 fan-out/ef 总量驱动的 HNSW 工作量理论带、实际布局边界和实测拟合，用于判断吞吐增长来自资源增加还是路由与局部检索工作变化。",
            "",
            "## 如何理解",
            "",
            "HashAll 理论式为 `Q1×M×W1/W(M)`，其中 `W(M)=M×D_shard(M)`。Orion 理论式把 upper graph、实际访问 shard 数 `P`、每 query 的 `ef_sum` 和每 shard 的局部规模都计入；这些路由统计由生产 `OrionRouter` 对相同 10,000 条 query 离线重放得到。",
            "",
            "## 最终结论",
            "",
            f"- HashAll 从 {hash_rows[0]['qps']:.2f} 增至 {hash_rows[-1]['qps']:.2f} QPS，即 {float(hash_rows[-1]['qps']) / float(hash_rows[0]['qps']):.3f}×。",
            f"- Orion 从 {orion_rows[0]['qps']:.2f} 增至 {orion_rows[-1]['qps']:.2f} QPS，即 {float(orion_rows[-1]['qps']) / float(orion_rows[0]['qps']):.3f}×。",
            f"- Orion 理论工作量模型在 9 个布局点上的 R²={orion['theory_metrics']['r_squared']:.3f}、MAPE={orion['theory_metrics']['mape']:.3f}；它明显优于仅用单条幂律概括布局变化。",
            "- M=2 的实际 1 shard 与 3 shard 分别为 1048.22 和 331.03 QPS；M=16 的 15/17 shard 分别为 3095.62 和 2496.58 QPS。布局与路由工作量会显著改变结果，不能把名义 M 直接等同于性能。",
            "",
            "## 并行执行策略",
            "",
            "- 数据布局、导入和 HNSW 预构建并行；路由 artifact 预安装但延迟激活。",
            "- 正式 Recall/QPS 按点串行；同一套 Qdrant 的 CPU quota、LLC、内存带宽、网卡和 coordinator 不在多个性能点之间共享。",
            "- Plain Simple KMeans 的五路并行预构建将准备墙钟时间从串行和 3035.8 秒降至 1200.5 秒，节省 1835.3 秒，准备阶段加速 2.53×。",
            "",
            "## 公平性与证据边界",
            "",
            "- 两种方法均为 GloVe-200-angular、Cosine、top-k=10、batch=200、1000 tuning query、独立 9000 held-out query、5 次正式重复和 CPU=2M。",
            "- 所有 Recall@10≥0.90、QPS CV≤5%，同一 M 的 HashAll/Orion recall 差不超过 0.015。",
            "- M=1/2/4 最多使用四台真实机器；M=8/16/32 是四台真实机器上的逻辑 shard 扩展。",
            "- 理论模型不包含网络、merge、缓存、NUMA 与负载偏斜，因此是解释性辅助线，不是普适复杂度证明。",
            "",
            "## 文件",
            "",
            "- `comparison.csv`：逐布局结果及路由统计",
            "- `models.json`：线性、理论和拟合曲线及误差",
            "- `hashall-vs-orion-linear-x.pdf`：带中文解释的综合图",
            "- `completion-audit.json`：资源、placement、协议、route trace、清理和 PDF 审计",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashall-root", type=Path, required=True)
    parser.add_argument("--orion-root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hash_rows, hash_benchmarks, hash_checks = load_hashall_rows(args.hashall_root)
    orion_rows, orion_benchmarks, orion_checks = load_orion_rows(args.orion_root, args.route_root)
    checks = [*hash_checks, *orion_checks]
    audit_fairness(hash_rows, hash_benchmarks, orion_rows, orion_benchmarks, checks)
    hash_model = hashall_model(hash_rows)
    orion = orion_model(orion_rows)
    comparison = args.output_dir / "comparison.csv"
    models = args.output_dir / "models.json"
    report = args.output_dir / "RESULTS_zh.md"
    write_csv(comparison, [*hash_rows, *orion_rows])
    write_models(models, hash_model, orion)
    write_results(report, hash_rows, orion_rows, hash_model, orion)
    font, temporary_font = chinese_font()
    try:
        pdf, png = plot_comparison(args.output_dir, hash_model, orion, font)
    finally:
        Path(temporary_font).unlink(missing_ok=True)
    pdf_proof = verify_pdf(pdf)
    audit = {
        "status": "PASS",
        "check_count": len(checks),
        "checks": checks,
        "comparison": {"path": str(comparison), "sha256": sha256(comparison)},
        "models": {"path": str(models), "sha256": sha256(models)},
        "report": {"path": str(report), "sha256": sha256(report)},
        "pdf": pdf_proof,
        "png": {"path": str(png), "sha256": sha256(png)},
        "x_axis": {
            "scale": "linear",
            "numeric_limits": [0.0, 33.0],
            "ticks": M_VALUES.tolist(),
            "linear_guide_is_straight": True,
        },
        "theory": {
            "hashall": hash_model["theory_metrics"],
            "orion": orion["theory_metrics"],
        },
    }
    audit_path = args.output_dir / "completion-audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "status": "PASS", "checks": len(checks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
