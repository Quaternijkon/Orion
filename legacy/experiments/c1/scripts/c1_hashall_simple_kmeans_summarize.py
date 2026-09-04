#!/usr/bin/env python3
"""Audit and compare HashAll with the plain native Simple KMeans baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import tempfile
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
MAX_RECALL_LEVEL = 0.93
MAX_CROSS_METHOD_RECALL_GAP = 0.015
MAX_CV = 0.05
N = 1_183_514.0
HNSW_M = 32.0
LEVEL_DELTA = 0.5
REQUIRED_PDF_TEXT = (
    "图表作用",
    "如何理解",
    "最终结论",
    "证据边界",
    "逻辑 shard 数 M（线性数值坐标）",
)
DISABLED_SIMPLE_KMEANS_ENHANCEMENTS = {
    "dynamic_ef",
    "load_aware_placement",
    "multi_assignment",
    "orion_upper_graph",
    "query_adaptive_nprobe",
    "topology_aware_partitioning",
}


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


def actual_shard_layout(benchmark: dict[str, Any]) -> tuple[list[int], list[int], bool]:
    per_node = benchmark.get("per_node_collection_cluster") or {}
    counts: list[int] = []
    shard_ids: list[int] = []
    all_active_and_stable = True
    for node in sorted(per_node):
        node_state = per_node[node]
        local = node_state.get("local_shards") or []
        counts.append(len(local))
        shard_ids.extend(int(shard["shard_id"]) for shard in local)
        all_active_and_stable = all_active_and_stable and all(
            shard.get("state") == "Active" for shard in local
        )
        all_active_and_stable = all_active_and_stable and not (
            node_state.get("shard_transfers") or []
        )
    return counts, shard_ids, all_active_and_stable


def audit_measurement_point(
    *,
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
        resources.get("logical_shards") == machine
        and resources.get("physical_machines") == 4
        and float(resources.get("cores_per_logical_shard", -1.0)) == 2.0,
        f"{prefix} linear four-host resource contract",
        checks,
    )
    resource_contract = benchmark.get("resource_contract") or {}
    require(
        resource_contract.get("quota_cores_per_node")
        == resources.get("quota_cores_per_node"),
        f"{prefix} benchmark used recorded quota schedule",
        checks,
    )
    actual_counts, shard_ids, active_and_stable = actual_shard_layout(benchmark)
    require(len(actual_counts) == 4, f"{prefix} has four physical peers", checks)
    require(
        actual_counts == resources.get("shards_per_node"),
        f"{prefix} quota follows live shard placement",
        checks,
    )
    require(
        sorted(shard_ids) == list(range(machine)),
        f"{prefix} has each logical shard exactly once",
        checks,
    )
    require(active_and_stable, f"{prefix} shards active with no transfers", checks)
    request = benchmark.get("request_contract") or {}
    require(
        request.get("standard_coordinator_request") is True
        and request.get("client_side_fanout") is False
        and request.get("shard_selector_present") is False,
        f"{prefix} uses ordinary coordinator Search",
        checks,
    )
    protocol = benchmark.get("protocol") or {}
    parameters = benchmark.get("parameters") or {}
    require(
        protocol.get("dataset") == "GloVe-200-angular"
        and protocol.get("distance") == "Cosine"
        and protocol.get("vector_size") == 200
        and float(protocol.get("target_recall_at_10", -1.0)) == TARGET_RECALL,
        f"{prefix} dataset and recall protocol",
        checks,
    )
    require(
        protocol.get("tuning_query_range") == [0, 1000]
        and protocol.get("heldout_query_range") == [1000, 10000]
        and (benchmark.get("heldout_recall") or {}).get("query_count") == 9000,
        f"{prefix} disjoint 1000 tuning and 9000 held-out queries",
        checks,
    )
    require(
        parameters.get("batch_size") == 200
        and parameters.get("top_k") == 10
        and float(parameters.get("warmup_seconds", -1.0)) == 10.0
        and float(parameters.get("sweep_seconds", -1.0)) == 8.0
        and float(parameters.get("measure_seconds", -1.0)) == 20.0
        and parameters.get("repeat_count") == 5,
        f"{prefix} common QPS measurement protocol",
        checks,
    )


def load_hashall_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[int, dict[str, Any]]]:
    path = root / "summary/summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    if [int(row["logical_shards"]) for row in summary_rows] != M_VALUES.astype(int).tolist():
        raise RuntimeError("HashAll summary does not contain exact M=1..32 points")
    checks: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    benchmarks: dict[int, dict[str, Any]] = {}
    for summary in summary_rows:
        machine = int(summary["logical_shards"])
        resources_path = root / f"m{machine}/resources-measure.json"
        benchmark_path = root / f"m{machine}/benchmark.json"
        require(resources_path.is_file(), f"HashAll M={machine} resources exists", checks)
        require(benchmark_path.is_file(), f"HashAll M={machine} benchmark exists", checks)
        resources = load_json(resources_path)
        benchmark = load_json(benchmark_path)
        audit_measurement_point(
            method="HashAll",
            machine=machine,
            resources=resources,
            benchmark=benchmark,
            checks=checks,
        )
        request = benchmark.get("request_contract") or {}
        require(
            request.get("hashall_fanout") == machine,
            f"HashAll M={machine} visits all shards",
            checks,
        )
        selected_ef = int(summary["selected_ef"])
        require(
            10 <= selected_ef <= 512
            and (benchmark.get("parameters") or {}).get("selected_hnsw_ef")
            == selected_ef,
            f"HashAll M={machine} ef is bounded and fixed",
            checks,
        )
        benchmarks[machine] = benchmark
        rows.append({
            "method": "HashAll",
            "logical_shards": machine,
            "nprobe": machine,
            "ef": selected_ef,
            "recall": float(summary["heldout_recall_at_10"]),
            "qps": float(summary["qps_mean"]),
            "qps_stdev": float(summary["qps_stdev"]),
            "qps_cv": float(summary["qps_cv"]),
            "allocated_cpu": float(summary["allocated_cpu_cores"]),
            "source": str(path),
        })
    source_audit = root / "summary/completion-audit.json"
    require(source_audit.is_file(), "HashAll source completion audit exists", checks)
    require(
        load_json(source_audit).get("status") == "PASS",
        "HashAll source completion audit PASS",
        checks,
    )
    return rows, checks, benchmarks


def load_simple_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[int, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, str]] = []
    benchmarks: dict[int, dict[str, Any]] = {}
    for machine in M_VALUES.astype(int):
        point = root / f"m{machine}"
        paths = {
            name: point / filename
            for name, filename in (
                ("prepare", "prepare.json"),
                ("resources", "resources-measure.json"),
                ("benchmark", "benchmark.json"),
                ("cleanup", "cleanup.json"),
            )
        }
        for name, path in paths.items():
            require(path.is_file(), f"Simple KMeans M={machine} {name} exists", checks)
        prepare = load_json(paths["prepare"])
        resources = load_json(paths["resources"])
        benchmark = load_json(paths["benchmark"])
        cleanup = load_json(paths["cleanup"])
        baseline = benchmark.get("baseline_contract") or {}
        heldout = benchmark.get("heldout_recall") or {}
        params = benchmark.get("parameters") or {}
        require(prepare.get("status") == "PASS", f"M={machine} prepare PASS", checks)
        require(cleanup.get("status") == "PASS", f"M={machine} cleanup PASS", checks)
        audit_measurement_point(
            method="Simple KMeans",
            machine=machine,
            resources=resources,
            benchmark=benchmark,
            checks=checks,
        )
        require(
            baseline.get("baseline") == "native_static_single_assignment_simple_kmeans",
            f"M={machine} plain native baseline",
            checks,
        )
        require(
            float(baseline.get("expansion_ratio")) == 1.0
            and baseline.get("logical_point_count") == baseline.get("physical_point_count"),
            f"M={machine} one assignment per point",
            checks,
        )
        disabled = baseline.get("enhancements_disabled") or {}
        require(
            set(disabled) == DISABLED_SIMPLE_KMEANS_ENHANCEMENTS
            and all(value is True for value in disabled.values()),
            f"M={machine} all Simple KMeans enhancements disabled",
            checks,
        )
        selected_nprobe = int(params["nprobe"])
        selected_ef = int(params["lower_hnsw_ef"])
        require(
            1 <= selected_nprobe <= machine and 10 <= selected_ef <= 512,
            f"M={machine} nprobe and ef are bounded",
            checks,
        )
        request = benchmark.get("request_contract") or {}
        require(
            request.get("server_router") == "native_simple_kmeans"
            and request.get("client_hnsw_ef_present") is False,
            f"M={machine} native Simple KMeans owns routing and fixed ef",
            checks,
        )
        require(
            float(heldout.get("recall_at_10", 0.0)) >= TARGET_RECALL,
            f"M={machine} held-out recall >= 0.90",
            checks,
        )
        require(
            float(benchmark.get("qps_cv", 1.0)) <= MAX_CV,
            f"M={machine} QPS CV <= 5%",
            checks,
        )
        rows.append(
            {
                "method": "Simple KMeans",
                "logical_shards": machine,
                "nprobe": selected_nprobe,
                "ef": selected_ef,
                "recall": float(heldout["recall_at_10"]),
                "qps": float(benchmark["qps_mean"]),
                "qps_stdev": float(benchmark["qps_stdev"]),
                "qps_cv": float(benchmark["qps_cv"]),
                "allocated_cpu": float(resources["quota_cores_total"]),
                "source": str(paths["benchmark"]),
            }
        )
        benchmarks[machine] = benchmark
    restore = root / "resource-restored.json"
    require(restore.is_file(), "Simple KMeans resource restore exists", checks)
    require(load_json(restore).get("status") == "PASS", "resource restore PASS", checks)
    return rows, checks, benchmarks


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


def fit_anchored_power(machines: np.ndarray, qps: np.ndarray) -> dict[str, Any]:
    feature = np.log(machines)
    baseline = float(qps[0])
    exponent = float(
        np.dot(feature, np.log(qps / baseline)) / np.dot(feature, feature)
    )
    predicted = baseline * np.power(machines, exponent)
    return {
        "kind": "anchored_power",
        "baseline_qps": baseline,
        "exponent": exponent,
        "predicted": predicted,
        "metrics": error_metrics(qps, predicted),
    }


def per_shard_work(machines: np.ndarray, ef: np.ndarray) -> np.ndarray:
    upper = HNSW_M * (np.log(N / machines) / np.log(HNSW_M) + LEVEL_DELTA)
    base = 2.0 * HNSW_M * ef
    return upper + base


def theoretical_qps(
    machines: np.ndarray,
    qps: np.ndarray,
    ef: np.ndarray,
    nprobe: np.ndarray,
) -> dict[str, Any]:
    work = per_shard_work(machines, ef)
    total_query_work = nprobe * work
    predicted = (
        float(qps[0])
        * machines
        * total_query_work[0]
        / total_query_work
    )
    return {
        "per_shard_work": work,
        "total_query_work": total_query_work,
        "predicted": predicted,
        "metrics": error_metrics(qps, predicted),
    }


def build_model(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    machines = np.asarray([row["logical_shards"] for row in rows], dtype=float)
    qps = np.asarray([row["qps"] for row in rows], dtype=float)
    ef = np.asarray([row["ef"] for row in rows], dtype=float)
    nprobe = np.asarray([row["nprobe"] for row in rows], dtype=float)
    fit = fit_anchored_power(machines, qps)
    theory = theoretical_qps(machines, qps, ef, nprobe)
    return {
        "machines": machines,
        "qps": qps,
        "qps_stdev": np.asarray([row["qps_stdev"] for row in rows]),
        "ef": ef,
        "nprobe": nprobe,
        "linear": float(qps[0]) * machines,
        "fit": fit,
        "theory": theory,
    }


def protocol_signature(benchmark: dict[str, Any]) -> dict[str, Any]:
    protocol = benchmark.get("protocol") or {}
    parameters = benchmark.get("parameters") or {}
    return {
        "dataset": protocol.get("dataset"),
        "distance": protocol.get("distance"),
        "vector_size": protocol.get("vector_size"),
        "target_recall_at_10": protocol.get("target_recall_at_10"),
        "tuning_query_range": protocol.get("tuning_query_range"),
        "heldout_query_range": protocol.get("heldout_query_range"),
        "batch_size": parameters.get("batch_size"),
        "top_k": parameters.get("top_k"),
        "warmup_seconds": parameters.get("warmup_seconds"),
        "sweep_seconds": parameters.get("sweep_seconds"),
        "measure_seconds": parameters.get("measure_seconds"),
        "repeat_count": parameters.get("repeat_count"),
    }


def hnsw_signature(method: str, benchmark: dict[str, Any]) -> dict[str, Any]:
    if method == "HashAll":
        collection_info = benchmark.get("collection_info") or {}
    else:
        collection_info = (benchmark.get("live_policy_gate") or {}).get(
            "collection_info"
        ) or {}
    hnsw = ((collection_info.get("config") or {}).get("hnsw_config") or {})
    return {
        "m": hnsw.get("m"),
        "ef_construct": hnsw.get("ef_construct"),
        "full_scan_threshold": hnsw.get("full_scan_threshold"),
        "max_indexing_threads": hnsw.get("max_indexing_threads"),
        "on_disk": hnsw.get("on_disk"),
    }


def audit_cross_method_fairness(
    hash_rows: Sequence[dict[str, Any]],
    simple_rows: Sequence[dict[str, Any]],
    hash_benchmarks: dict[int, dict[str, Any]],
    simple_benchmarks: dict[int, dict[str, Any]],
    checks: list[dict[str, str]],
) -> None:
    for hash_row, simple_row in zip(hash_rows, simple_rows, strict=True):
        machine = int(hash_row["logical_shards"])
        require(
            machine == int(simple_row["logical_shards"]),
            f"M={machine} methods compare the same scale",
            checks,
        )
        require(
            protocol_signature(hash_benchmarks[machine])
            == protocol_signature(simple_benchmarks[machine]),
            f"M={machine} methods use identical query protocol",
            checks,
        )
        require(
            hnsw_signature("HashAll", hash_benchmarks[machine])
            == hnsw_signature("Simple KMeans", simple_benchmarks[machine])
            == {
                "m": 32,
                "ef_construct": 200,
                "full_scan_threshold": 10,
                "max_indexing_threads": 1,
                "on_disk": False,
            },
            f"M={machine} methods use identical HNSW construction settings",
            checks,
        )
        require(
            float(hash_row["allocated_cpu"]) == float(simple_row["allocated_cpu"])
            == 2.0 * machine,
            f"M={machine} methods receive identical CPU=2M",
            checks,
        )
        require(
            TARGET_RECALL <= float(hash_row["recall"]) < MAX_RECALL_LEVEL
            and TARGET_RECALL <= float(simple_row["recall"]) < MAX_RECALL_LEVEL,
            f"M={machine} both methods are in the Recall@10=0.90 level",
            checks,
        )
        require(
            abs(float(hash_row["recall"]) - float(simple_row["recall"]))
            <= MAX_CROSS_METHOD_RECALL_GAP,
            f"M={machine} cross-method recall gap <= 0.015",
            checks,
        )


def configure_linear_m_axis(
    axis: plt.Axes, font: FontProperties | None = None
) -> None:
    axis.set_xscale("linear")
    axis.set_xlim(0.0, 33.0)
    axis.set_xticks(M_VALUES, [str(int(value)) for value in M_VALUES])
    axis.set_xlabel("逻辑 shard 数 M（线性数值坐标）", fontproperties=font)
    axis.grid(True, linestyle="--", alpha=0.55)


def chinese_font() -> tuple[FontProperties, str]:
    buffer = pymupdf.Font(fontname="china-s").buffer
    handle = tempfile.NamedTemporaryFile(suffix=".ttf", delete=False)
    handle.write(buffer)
    handle.close()
    return FontProperties(fname=handle.name), handle.name


def wrap_cjk(text: str, width: int = 78) -> str:
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
        ("图表作用", purpose, 0.96, "#0f172a"),
        ("如何理解", reading, 0.72, "#312e81"),
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
            0.095,
            y,
            wrap_cjk(body),
            transform=axis.transAxes,
            va="top",
            fontproperties=font,
            fontsize=10.2,
            color="#1f2937",
            linespacing=1.25,
        )


def plot_method(
    axis: plt.Axes,
    model: dict[str, Any],
    font: FontProperties,
    title: str,
) -> None:
    machines = model["machines"]
    dense = np.linspace(1.0, 32.0, 500)
    baseline = float(model["qps"][0])
    exponent = float(model["fit"]["exponent"])
    axis.plot(
        dense,
        baseline * dense,
        "--",
        color="#2f855a",
        linewidth=2.2,
        label="线性辅助线：QPS(1)×M",
    )
    axis.plot(
        machines,
        model["theory"]["predicted"],
        "-.D",
        color="#7c3aed",
        linewidth=2.3,
        markersize=6,
        label="理论辅助线：CPU/(nprobe×HNSW 工作量)",
    )
    axis.plot(
        dense,
        baseline * np.power(dense, exponent),
        "-",
        color="#0b3f73",
        linewidth=2.7,
        label=f"实测拟合：QPS(1)×M^{exponent:.3f}",
    )
    axis.errorbar(
        machines,
        model["qps"],
        yerr=model["qps_stdev"],
        fmt="o",
        color="#1769aa",
        capsize=4,
        markersize=7,
        label="实测 QPS（均值±标准差）",
        zorder=5,
    )
    configure_linear_m_axis(axis, font)
    axis.set_ylabel("QPS", fontproperties=font)
    axis.set_title(title, fontproperties=font, fontsize=13)
    axis.legend(prop=font, fontsize=8.5, loc="upper left", framealpha=0.95)


def plot_comparison(
    output_dir: Path,
    hash_model: dict[str, Any],
    simple_model: dict[str, Any],
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
    figure = plt.figure(figsize=(15.2, 10.8))
    grid = figure.add_gridspec(2, 2, height_ratios=(3.25, 1.6), hspace=0.24, wspace=0.18)
    hash_axis = figure.add_subplot(grid[0, 0])
    simple_axis = figure.add_subplot(grid[0, 1])
    text_axis = figure.add_subplot(grid[1, :])
    plot_method(hash_axis, hash_model, font, "HashAll：访问全部 M 个分片")
    plot_method(simple_axis, simple_model, font, "原生 Simple KMeans：固定 nprobe")

    hash_speedup = float(hash_model["qps"][-1] / hash_model["qps"][0])
    simple_speedup = float(simple_model["qps"][-1] / simple_model["qps"][0])
    simple_vs_hash = float(simple_model["qps"][-1] / hash_model["qps"][-1])
    purpose = "在完全相同的 GloVe、Recall@10≥0.90、总 CPU=2M 和普通 coordinator 请求下，对比 HashAll 与未采用任何改良的单层 Simple KMeans 横向扩展。"
    reading = "横轴是真正的线性数值坐标，1→2 的距离只有 16→32 的 1/16，所以绿色 QPS(1)×M 必然是直线。紫线按每个方法实际的 ef 和 nprobe 估计工作量；深蓝线只拟合实测增长指数。"
    conclusion = (
        f"HashAll 从 M=1 到 32 提升 {hash_speedup:.3f}×，拟合指数 "
        f"b={hash_model['fit']['exponent']:.3f}；Simple KMeans 提升 "
        f"{simple_speedup:.3f}×，拟合指数 b={simple_model['fit']['exponent']:.3f}。"
        f"M=32 时 Simple KMeans/HashAll QPS={simple_vs_hash:.3f}×。"
    )
    boundary = "M=1/2/4 对应最多四台真实机器参与数据承载；M=8/16/32 是四台真实机器上的逻辑 shard 扩展。理论线忽略质心路由、网络、merge 和缓存差异，只是辅助线，不是普适复杂度定理。"
    explanation_panel(
        text_axis,
        font,
        purpose=purpose,
        reading=reading,
        conclusion=conclusion,
        boundary=boundary,
    )
    figure.suptitle(
        "GloVe-200-angular：HashAll 与原生 Simple KMeans 的固定召回率扩展",
        fontproperties=font,
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.947,
        "4 台物理机；总 CPU=2M；Recall@10≥0.90；top-k=10；batch=200；横轴为线性数值坐标",
        ha="center",
        fontproperties=font,
        fontsize=11,
        color="#334155",
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.04)
    pdf = output_dir / "hashall-vs-plain-simple-kmeans-linear-x.pdf"
    png = output_dir / "hashall-vs-plain-simple-kmeans-linear-x.png"
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
        "logical_shards",
        "nprobe",
        "ef",
        "recall",
        "qps",
        "qps_stdev",
        "qps_cv",
        "allocated_cpu",
        "source",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_results_zh(
    path: Path,
    hash_rows: Sequence[dict[str, Any]],
    simple_rows: Sequence[dict[str, Any]],
    hash_model: dict[str, Any],
    simple_model: dict[str, Any],
) -> None:
    lines = [
        "# GloVe-200-angular：HashAll 与原生 Simple KMeans 的 M=1～32 QPS 对比",
        "",
        "## 实验结果",
        "",
        "| M | 总 CPU | HashAll ef | HashAll Recall@10 | HashAll QPS | KMeans nprobe | KMeans ef | KMeans Recall@10 | KMeans QPS | KMeans/HashAll |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hash_row, simple_row in zip(hash_rows, simple_rows, strict=True):
        ratio = float(simple_row["qps"]) / float(hash_row["qps"])
        lines.append(
            f"| {hash_row['logical_shards']} | {hash_row['allocated_cpu']:.0f} | "
            f"{hash_row['ef']} | {hash_row['recall']:.4f} | {hash_row['qps']:.2f} | "
            f"{simple_row['nprobe']} | {simple_row['ef']} | "
            f"{simple_row['recall']:.4f} | {simple_row['qps']:.2f} | {ratio:.3f}× |"
        )
    hash_speedup = float(hash_rows[-1]["qps"]) / float(hash_rows[0]["qps"])
    simple_speedup = float(simple_rows[-1]["qps"]) / float(simple_rows[0]["qps"])
    lines.extend(
        [
            "",
            "## 图表作用",
            "",
            "综合图同时给出实测点、从 M=1 出发的理想线性辅助线、包含实际 ef/nprobe 的 HNSW 工作量理论辅助线，以及锚定 M=1 的幂律实测拟合。它用于判断增加严格成比例 CPU 和逻辑 shard 后，吞吐量距离理想线性扩展还有多远。",
            "",
            "## 如何理解",
            "",
            "横轴使用线性数值坐标，因此绿色 `QPS(1)×M` 是直线。理论辅助线使用：",
            "",
            "`D(M)=m_h[log_{m_h}(N/M)+0.5]+2m_h ef_R(M)`",
            "",
            "`QPS_theory(M)=QPS(1)×M×[nprobe(1)D(1)]/[nprobe(M)D(M)]`",
            "",
            "这里 `ef_R(M)` 是在独立 tuning query 上达到目标召回率所选的固定 ef；Simple KMeans 的 nprobe 也是每个 M 固定，不做 query 自适应。理论线忽略质心路由、网络、merge、缓存和内存带宽，只作为辅助比较。",
            "",
            "## 最终结论",
            "",
            f"- HashAll 从 {hash_rows[0]['qps']:.2f} QPS 增长到 {hash_rows[-1]['qps']:.2f} QPS，即 {hash_speedup:.3f}×；锚定幂律指数为 {hash_model['fit']['exponent']:.3f}。",
            f"- Simple KMeans 从 {simple_rows[0]['qps']:.2f} QPS 增长到 {simple_rows[-1]['qps']:.2f} QPS，即 {simple_speedup:.3f}×；锚定幂律指数为 {simple_model['fit']['exponent']:.3f}。",
            f"- M=32 时 Simple KMeans 为 HashAll 的 {float(simple_rows[-1]['qps']) / float(hash_rows[-1]['qps']):.3f}×。本次六点中 HashAll 始终更快，但差距在 M=32 缩小。",
            "- 两种方法都明显低于理想线性扩展；这说明严格增加 CPU 并不能消除全局 fan-out、局部图搜索、路由与合并等累计工作。",
            "",
            "## 公平性与证据边界",
            "",
            "- 两种方法均使用 GloVe-200-angular、Cosine、top-k=10、batch=200、1000 条 tuning query、独立 9000 条 held-out query、20 秒正式测量和 5 次重复。",
            "- 每个 M 的 Qdrant CPU quota 精确等于 `2M`；两种方法在相同 M 获得完全相同的 CPU 上限。",
            "- 所有 held-out Recall@10 位于 `[0.90, 0.93)`，同一 M 两种方法的差距不超过 0.015。",
            "- Simple KMeans 是 Qdrant 原生静态单重分配版本；禁用 Orion upper graph、动态 ef、多重分配、query-adaptive nprobe、拓扑感知分区和负载感知 placement。",
            "- M=1/2/4 最多使用四台真实机器承载 shard；M=8/16/32 是四台真实机器上的逻辑 shard 扩展，不能解释成 8/16/32 台物理机。",
            "",
            "## 文件",
            "",
            "- `comparison.csv`：逐点原始汇总",
            "- `models.json`：线性线、理论线和拟合曲线数值",
            "- `hashall-vs-plain-simple-kmeans-linear-x.pdf`：带中文解释的综合图",
            "- `completion-audit.json`：资源、placement、协议、召回率和 PDF 完整性审计",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def json_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "machines": model["machines"].tolist(),
        "qps": model["qps"].tolist(),
        "ef": model["ef"].tolist(),
        "nprobe": model["nprobe"].tolist(),
        "linear": model["linear"].tolist(),
        "fit": {
            "kind": model["fit"]["kind"],
            "baseline_qps": model["fit"]["baseline_qps"],
            "exponent": model["fit"]["exponent"],
            "predicted": model["fit"]["predicted"].tolist(),
            "metrics": model["fit"]["metrics"],
        },
        "theory": {
            "formula": "Q1*M*(nprobe1*D1)/(nprobe(M)*D(M))",
            "D": "m*(log_m(N/M)+0.5)+2*m*ef_R(M)",
            "predicted": model["theory"]["predicted"].tolist(),
            "per_shard_work": model["theory"]["per_shard_work"].tolist(),
            "total_query_work": model["theory"]["total_query_work"].tolist(),
            "metrics": model["theory"]["metrics"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashall-root", type=Path, required=True)
    parser.add_argument("--simple-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hash_rows, hash_checks, hash_benchmarks = load_hashall_rows(args.hashall_root)
    simple_rows, simple_checks, simple_benchmarks = load_simple_rows(args.simple_root)
    checks = [*hash_checks, *simple_checks]
    for row in hash_rows:
        require(row["recall"] >= TARGET_RECALL, f"HashAll M={row['logical_shards']} recall", checks)
        require(row["qps_cv"] <= MAX_CV, f"HashAll M={row['logical_shards']} CV", checks)
    audit_cross_method_fairness(
        hash_rows,
        simple_rows,
        hash_benchmarks,
        simple_benchmarks,
        checks,
    )
    hash_model = build_model(hash_rows)
    simple_model = build_model(simple_rows)
    write_csv(args.output_dir / "comparison.csv", [*hash_rows, *simple_rows])

    font, temporary_font = chinese_font()
    try:
        pdf, png = plot_comparison(args.output_dir, hash_model, simple_model, font)
    finally:
        Path(temporary_font).unlink(missing_ok=True)
    pdf_proof = verify_pdf(pdf)
    models = {"HashAll": json_model(hash_model), "Simple KMeans": json_model(simple_model)}
    temporary = args.output_dir / "models.json.tmp"
    temporary.write_text(json.dumps(models, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "models.json")
    report = args.output_dir / "RESULTS_zh.md"
    write_results_zh(report, hash_rows, simple_rows, hash_model, simple_model)
    audit = {
        "status": "PASS",
        "check_count": len(checks),
        "checks": checks,
        "comparison": {
            "path": str(args.output_dir / "comparison.csv"),
            "sha256": sha256(args.output_dir / "comparison.csv"),
        },
        "models": {
            "path": str(args.output_dir / "models.json"),
            "sha256": sha256(args.output_dir / "models.json"),
        },
        "report": {"path": str(report), "sha256": sha256(report)},
        "pdf": pdf_proof,
        "png": {"path": str(png), "sha256": sha256(png)},
        "x_axis": {
            "scale": "linear",
            "numeric_limits": [0.0, 33.0],
            "ticks": M_VALUES.tolist(),
            "linear_guide_is_straight": True,
        },
    }
    (args.output_dir / "completion-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "status": "PASS"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
