#!/usr/bin/env python3
"""Audit, summarize, fit, and plot the HashAll virtual scale experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
import pymupdf


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


LOGICAL_SHARDS = (1, 2, 4, 8, 16, 32)
PHYSICAL_MACHINES = 4
FULL_CLUSTER_CORES = 64.0
CORES_PER_LOGICAL_SHARD = 2.0
TARGET_RECALL = 0.90
MAX_QPS_CV = 0.05
DEFAULT_DATASET_SIZE = 1_000_000.0
DEFAULT_HNSW_M = 32.0
QDRANT_LEVEL_DELTA = 0.5
REQUIRED_PDF_TEXT = ("图表作用", "如何理解", "最终结论", "证据边界")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_metadata(root: Path) -> dict[str, Any]:
    """Load dataset and search settings from the measured artifacts."""
    preflight = load_json(root / "preflight.json")
    benchmark = load_json(root / "m1" / "benchmark.json")
    prepare = load_json(root / "m1" / "prepare.json")
    dataset = preflight.get("dataset") or {}
    protocol = benchmark.get("protocol") or {}
    parameters = benchmark.get("parameters") or {}
    collection_body = prepare.get("collection_body") or {}
    vectors = collection_body.get("vectors") or {}
    hnsw = collection_body.get("hnsw_config") or {}
    shapes = dataset.get("shapes") or {}
    train_shape = shapes.get("train") or []
    dataset_name = str(
        dataset.get("name") or protocol.get("dataset") or "unknown-dataset"
    )
    dataset_key = str(
        dataset.get("key")
        or protocol.get("dataset_key")
        or dataset_name.lower().replace(" ", "-")
    )
    return {
        "dataset_name": dataset_name,
        "dataset_key": dataset_key,
        "dataset_path": str(dataset.get("path") or ""),
        "train_count": int(train_shape[0]),
        "vector_size": int(
            dataset.get("vector_size")
            or protocol.get("vector_size")
            or vectors.get("size")
        ),
        "qdrant_distance": str(
            dataset.get("qdrant_distance")
            or protocol.get("distance")
            or vectors.get("distance")
        ),
        "hdf5_distance": dataset.get("hdf5_distance"),
        "target_recall": float(
            protocol.get("target_recall_at_10", TARGET_RECALL)
        ),
        "top_k": int(parameters.get("top_k", 10)),
        "hnsw_m": float(hnsw.get("m", DEFAULT_HNSW_M)),
        "ef_construct": int(hnsw.get("ef_construct", 200)),
        "ef_lower_bound": int(protocol.get("ef_lower_bound", parameters.get("top_k", 10))),
        "ef_upper_bound": int(protocol.get("ef_upper_bound", max(parameters.get("selected_hnsw_ef", 10), 10))),
    }


def expected_shards_per_node(logical_shards: int) -> list[int]:
    return [
        logical_shards // PHYSICAL_MACHINES
        + (1 if index < logical_shards % PHYSICAL_MACHINES else 0)
        for index in range(PHYSICAL_MACHINES)
    ]


def coefficient_of_determination(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if total == 0.0:
        return 1.0 if residual == 0.0 else float("-inf")
    return 1.0 - residual / total


def error_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    relative = (predicted - observed) / observed
    return {
        "r_squared": coefficient_of_determination(observed, predicted),
        "mape": float(np.mean(np.abs(relative))),
        "max_absolute_relative_error": float(np.max(np.abs(relative))),
        "rmse_qps": float(np.sqrt(np.mean((predicted - observed) ** 2))),
    }


def fit_anchored_log2(
    machines: np.ndarray, qps: np.ndarray
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Fit Q(M)=Q(1)+a*log2(M), keeping the measured M=1 point fixed."""
    feature = np.log2(machines)
    baseline = float(qps[0])
    slope = float(np.dot(feature, qps - baseline) / np.dot(feature, feature))
    predicted = baseline + slope * feature
    return slope, predicted, error_metrics(qps, predicted)


def hnsw_distance_proxy(
    machines: np.ndarray,
    ef: np.ndarray,
    *,
    dataset_size: float = DEFAULT_DATASET_SIZE,
    hnsw_m: float = DEFAULT_HNSW_M,
) -> np.ndarray:
    upper = hnsw_m * (
        np.log(dataset_size / machines) / np.log(hnsw_m) + QDRANT_LEVEL_DELTA
    )
    base = 2.0 * hnsw_m * ef
    return upper + base


def theoretical_qps(
    machines: np.ndarray,
    ef: np.ndarray,
    baseline_qps: float,
    *,
    dataset_size: float = DEFAULT_DATASET_SIZE,
    hnsw_m: float = DEFAULT_HNSW_M,
) -> tuple[np.ndarray, np.ndarray]:
    distance_proxy = hnsw_distance_proxy(
        machines, ef, dataset_size=dataset_size, hnsw_m=hnsw_m
    )
    qps = baseline_qps * distance_proxy[0] / distance_proxy
    return distance_proxy, qps


def mean_node_cpu(repeats: Sequence[dict[str, Any]]) -> dict[str, float]:
    hosts = sorted(repeats[0]["cpu_average_cores"])
    return {
        host: statistics.fmean(
            float(repeat["cpu_average_cores"][host]) for repeat in repeats
        )
        for host in hosts
    }


def selected_tuning_recall(benchmark: dict[str, Any], selected_ef: int) -> float:
    matches = [
        float(row["recall_at_10"])
        for row in benchmark["tuning_sweep"]
        if int(row["hnsw_ef"]) == selected_ef
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one tuning row for ef={selected_ef}, got {matches}")
    return matches[0]


def require(condition: bool, label: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": label, "status": "PASS" if condition else "FAIL"})
    if not condition:
        raise ValueError(f"audit check failed: {label}")


def audit_and_collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = experiment_metadata(root)
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for logical_shards in LOGICAL_SHARDS:
        point_root = root / f"m{logical_shards}"
        paths = {
            name: point_root / filename
            for name, filename in (
                ("prepare", "prepare.json"),
                ("resources", "resources-measure.json"),
                ("benchmark", "benchmark.json"),
                ("cleanup", "cleanup.json"),
            )
        }
        for name, path in paths.items():
            require(path.is_file(), f"M={logical_shards} {name} exists", checks)
        prepare = load_json(paths["prepare"])
        resources = load_json(paths["resources"])
        benchmark = load_json(paths["benchmark"])
        cleanup = load_json(paths["cleanup"])

        prefix = f"M={logical_shards}"
        require(prepare.get("status") == "PASS", f"{prefix} prepare PASS", checks)
        require(
            prepare.get("single_hnsw_segment_gate", {}).get("status") == "PASS",
            f"{prefix} one nonempty HNSW graph per shard",
            checks,
        )
        require(
            prepare.get("shard_layout_gate", {}).get("status") == "PASS",
            f"{prefix} shard layout PASS",
            checks,
        )
        expected_shards = expected_shards_per_node(logical_shards)
        require(
            resources.get("shards_per_node") == expected_shards,
            f"{prefix} shards per node exact",
            checks,
        )
        expected_quota = CORES_PER_LOGICAL_SHARD * logical_shards
        require(
            math.isclose(
                float(resources.get("quota_cores_total", -1.0)),
                expected_quota,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            f"{prefix} total CPU quota equals 2M",
            checks,
        )
        require(
            resources.get("status") == "PASS",
            f"{prefix} resource contract PASS",
            checks,
        )
        require(
            int(resources.get("physical_machines", 0)) == PHYSICAL_MACHINES,
            f"{prefix} physical host count is four",
            checks,
        )
        require(
            benchmark.get("status") == "PASS",
            f"{prefix} benchmark PASS",
            checks,
        )
        heldout_recall = float(benchmark["heldout_recall"]["recall_at_10"])
        require(
            heldout_recall >= metadata["target_recall"],
            f"{prefix} held-out Recall@10 >= {metadata['target_recall']:.2f}",
            checks,
        )
        qps_cv = float(benchmark["qps_cv"])
        require(qps_cv <= MAX_QPS_CV, f"{prefix} QPS CV <= 5%", checks)
        require(
            bool(benchmark["saturation_selection"]["knee_observed"]),
            f"{prefix} concurrency knee observed",
            checks,
        )
        repeat_count = len(benchmark["repeats"])
        require(
            5 <= repeat_count <= 7,
            f"{prefix} formal repeat count is 5..7",
            checks,
        )
        require(cleanup.get("status") == "PASS", f"{prefix} cleanup PASS", checks)
        require(
            cleanup.get("remaining_collections") == [],
            f"{prefix} cleanup left no collection",
            checks,
        )

        selected_ef = int(benchmark["parameters"]["selected_hnsw_ef"])
        tuning_recall = selected_tuning_recall(benchmark, selected_ef)
        repeats = benchmark["repeats"]
        qps_mean = float(benchmark["qps_mean"])
        cpu_used = statistics.fmean(
            float(repeat["cpu_average_cores_total"]) for repeat in repeats
        )
        node_cpu = mean_node_cpu(repeats)
        allocated_cpu = float(resources["quota_cores_total"])
        cpu_us_per_query = cpu_used / qps_mean * 1e6
        rows.append(
            {
                "logical_shards": logical_shards,
                "physical_machines": PHYSICAL_MACHINES,
                "shards_per_node": expected_shards,
                "allocated_cpu_cores": allocated_cpu,
                "quota_cores_per_node": resources["quota_cores_per_node"],
                "selected_ef": selected_ef,
                "tuning_recall_at_10": tuning_recall,
                "heldout_recall_at_10": heldout_recall,
                "selected_concurrency": int(
                    benchmark["parameters"]["selected_concurrency"]
                ),
                "qps_mean": qps_mean,
                "qps_stdev": float(benchmark["qps_stdev"]),
                "qps_cv": qps_cv,
                "repeat_count": repeat_count,
                "cpu_used_cores_mean": cpu_used,
                "cpu_utilization_of_quota": cpu_used / allocated_cpu,
                "cpu_us_per_query": cpu_us_per_query,
                "cpu_us_per_query_per_shard": cpu_us_per_query / logical_shards,
                "qps_per_allocated_core": qps_mean / allocated_cpu,
                "qps_per_used_core": qps_mean / cpu_used,
                "node_cpu_used_cores_mean": node_cpu,
                "source_prepare": str(paths["prepare"].resolve()),
                "source_resources": str(paths["resources"].resolve()),
                "source_benchmark": str(paths["benchmark"].resolve()),
                "source_cleanup": str(paths["cleanup"].resolve()),
            }
        )

    restore_path = root / "resource-restored.json"
    require(restore_path.is_file(), "resource restoration record exists", checks)
    restore = load_json(restore_path)
    require(restore.get("status") == "PASS", "resource restoration PASS", checks)
    require(
        all(node.get("valid") for node in restore.get("nodes", []))
        and len(restore.get("nodes", [])) == PHYSICAL_MACHINES,
        "all four resource restorations valid",
        checks,
    )
    return rows, checks


def add_derived_columns(
    rows: list[dict[str, Any]],
    *,
    dataset_size: float = DEFAULT_DATASET_SIZE,
    hnsw_m: float = DEFAULT_HNSW_M,
) -> dict[str, Any]:
    machines = np.asarray([row["logical_shards"] for row in rows], dtype=float)
    qps = np.asarray([row["qps_mean"] for row in rows], dtype=float)
    ef = np.asarray([row["selected_ef"] for row in rows], dtype=float)
    baseline = float(qps[0])
    slope, fitted_qps, fit_metrics = fit_anchored_log2(machines, qps)
    distance_proxy, theory_qps = theoretical_qps(
        machines,
        ef,
        baseline,
        dataset_size=dataset_size,
        hnsw_m=hnsw_m,
    )
    theory_metrics = error_metrics(qps, theory_qps)
    linear_qps = baseline * machines
    old_inverse_log_qps = baseline * math.log(dataset_size) / np.log(
        dataset_size / machines
    )
    old_metrics = error_metrics(qps, old_inverse_log_qps)

    previous_qps: float | None = None
    for index, row in enumerate(rows):
        row.update(
            {
                "qps_relative_to_m1": float(qps[index] / baseline),
                "qps_relative_to_previous": None
                if previous_qps is None
                else float(qps[index] / previous_qps),
                "linear_qps": float(linear_qps[index]),
                "linear_scaling_efficiency": float(qps[index] / linear_qps[index]),
                "hnsw_distance_proxy": float(distance_proxy[index]),
                "theory_qps": float(theory_qps[index]),
                "empirical_log_fit_qps": float(fitted_qps[index]),
                "old_inverse_log_qps": float(old_inverse_log_qps[index]),
            }
        )
        previous_qps = float(qps[index])

    return {
        "machines": machines,
        "qps": qps,
        "ef": ef,
        "dataset_size": dataset_size,
        "hnsw_m": hnsw_m,
        "baseline_qps": baseline,
        "linear_qps": linear_qps,
        "distance_proxy": distance_proxy,
        "theory_qps": theory_qps,
        "theory_metrics": theory_metrics,
        "fit_slope_qps_per_doubling": slope,
        "fitted_qps": fitted_qps,
        "fit_metrics": fit_metrics,
        "old_inverse_log_qps": old_inverse_log_qps,
        "old_metrics": old_metrics,
    }


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    fields = [
        "logical_shards",
        "physical_machines",
        "shards_per_node",
        "allocated_cpu_cores",
        "quota_cores_per_node",
        "selected_ef",
        "tuning_recall_at_10",
        "heldout_recall_at_10",
        "selected_concurrency",
        "qps_mean",
        "qps_stdev",
        "qps_cv",
        "qps_relative_to_m1",
        "qps_relative_to_previous",
        "linear_qps",
        "linear_scaling_efficiency",
        "cpu_used_cores_mean",
        "cpu_utilization_of_quota",
        "cpu_us_per_query",
        "cpu_us_per_query_per_shard",
        "qps_per_allocated_core",
        "qps_per_used_core",
        "hnsw_distance_proxy",
        "theory_qps",
        "empirical_log_fit_qps",
        "old_inverse_log_qps",
        "repeat_count",
        "source_benchmark",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field) for field in fields}
            output["shards_per_node"] = "/".join(map(str, row["shards_per_node"]))
            output["quota_cores_per_node"] = "/".join(
                f"{value:g}" for value in row["quota_cores_per_node"]
            )
            writer.writerow(output)
    return path


def chinese_font() -> tuple[FontProperties, str]:
    font_buffer = pymupdf.Font(fontname="china-s").buffer
    handle = tempfile.NamedTemporaryFile(suffix=".ttf", delete=False)
    handle.write(font_buffer)
    handle.close()
    return FontProperties(fname=handle.name), handle.name


def wrap_cjk(text: str, width: int = 78) -> str:
    return "\n".join(text[index : index + width] for index in range(0, len(text), width))


def add_explanation_panel(
    axis: plt.Axes,
    font: FontProperties,
    *,
    purpose: str,
    reading: str,
    conclusion: str,
    boundary: str,
) -> None:
    axis.axis("off")
    labels = ("图表作用", "如何理解", "最终结论", "证据边界")
    bodies = (purpose, reading, conclusion, boundary)
    colors = ("#0f172a", "#312e81", "#7c2d12", "#475569")
    y_positions = (0.96, 0.72, 0.43, 0.16)
    for label, body, color, y in zip(labels, bodies, colors, y_positions, strict=True):
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


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.9,
            "axes.facecolor": "#fbfdff",
            "figure.facecolor": "white",
            "grid.color": "#d7dee8",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    figure.savefig(pdf, dpi=300)
    figure.savefig(png, dpi=220)
    plt.close(figure)
    return pdf, png


def plot_qps_scaling(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    model: dict[str, Any],
    font: FontProperties,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    machines = model["machines"]
    qps = model["qps"]
    qps_std = np.asarray([row["qps_stdev"] for row in rows], dtype=float)
    baseline = model["baseline_qps"]
    dense = np.linspace(1.0, 32.0, 400)
    fit_dense = baseline + model["fit_slope_qps_per_doubling"] * np.log2(dense)
    linear_dense = baseline * dense

    figure = plt.figure(figsize=(15.2, 10.8))
    grid = figure.add_gridspec(2, 2, height_ratios=(3.25, 1.6), hspace=0.24, wspace=0.18)
    full = figure.add_subplot(grid[0, 0])
    zoom = figure.add_subplot(grid[0, 1])
    text_axis = figure.add_subplot(grid[1, :])
    colors = {
        "observed": "#1769aa",
        "linear": "#2f855a",
        "theory": "#7c3aed",
        "fit": "#0b3f73",
    }

    full.plot(
        dense,
        linear_dense,
        color=colors["linear"],
        linestyle="--",
        linewidth=2.4,
        label="理想线性：QPS(1)×M",
    )
    for axis in (full, zoom):
        axis.set_xscale("linear")
        axis.plot(
            machines,
            model["theory_qps"],
            color=colors["theory"],
            linestyle="-.",
            marker="D",
            markersize=6.5,
            linewidth=2.4,
            label="固定召回率 HNSW 理论辅助线",
        )
        axis.plot(
            dense,
            fit_dense,
            color=colors["fit"],
            linewidth=2.8,
            label="实测对数拟合：QPS(1)+a·log₂M",
        )
        axis.errorbar(
            machines,
            qps,
            yerr=qps_std,
            fmt="o",
            color=colors["observed"],
            ecolor=colors["observed"],
            capsize=4,
            markersize=7,
            label="实测 QPS（均值±标准差）",
            zorder=5,
        )
        axis.set_xticks(machines, [str(int(value)) for value in machines])
        axis.set_xlim(0.0, 33.0)
        axis.set_xlabel("逻辑 shard 数 M（4 台真实机器）", fontproperties=font, fontsize=11)
        axis.set_ylabel("QPS", fontproperties=font, fontsize=11)
        axis.grid(True)

    full.set_ylim(0, float(model["linear_qps"][-1]) * 1.06)
    full.set_title("全尺度：与理想线性扩展比较", fontproperties=font, fontsize=13)
    full.legend(prop=font, loc="upper left", fontsize=9, framealpha=0.96)
    zoom_values = np.concatenate((qps, model["theory_qps"], fit_dense))
    zoom_min = max(0.0, float(np.min(zoom_values)) * 0.82)
    zoom_max = float(np.max(zoom_values)) * 1.18
    zoom.set_ylim(zoom_min, zoom_max)
    zoom.set_title("非线性区域放大", fontproperties=font, fontsize=13)
    zoom.legend(prop=font, loc="upper left", fontsize=9, framealpha=0.96)
    for machine, value in zip(machines, qps, strict=True):
        zoom.annotate(
            f"{value / 1000:.1f}k",
            (machine, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontproperties=font,
            fontsize=8.5,
            color=colors["observed"],
        )

    fit_metrics = model["fit_metrics"]
    theory_metrics = model["theory_metrics"]
    purpose = "比较严格线性 CPU 预算下的实测吞吐、理想线性扩展、含 ef 的 HNSW 局部复杂度理论辅助线，以及实际结果的简约对数拟合。"
    reading = "横轴使用等比例线性坐标，点的水平距离与 M 的数值差成比例，因此绿色 QPS(1)×M 辅助线是直线。紫色线用 D=m[log_m(N/M)+0.5]+2m·ef_R(M) 计算局部工作量；深蓝线拟合每次翻倍带来的 QPS 增量。"
    speedup = float(qps[-1] / qps[0])
    conclusion = (
        f"QPS 从 {qps[0]:,.0f} 增至 {qps[-1]:,.0f}，M=32 为 {speedup:.3f}×。"
        f"实测拟合为 QPS={baseline:.1f}+{model['fit_slope_qps_per_doubling']:.1f}·log₂M，"
        f"R²={fit_metrics['r_squared']:.3f}；理论辅助线 R²={theory_metrics['r_squared']:.3f}，"
        "用来判断实测更接近哪类增长趋势；是否可称为近似对数增长必须结合拟合误差，而不能预设。"
    )
    boundary = "所有点都运行在 4 台真实机器上；M=8/16/32 是每台承载多个逻辑 shard 的虚拟扩展，不等价于 8/16/32 台真实机器。理论线采用单位系数，只是辅助解释，不是普适定律。"
    add_explanation_panel(
        text_axis,
        font,
        purpose=purpose,
        reading=reading,
        conclusion=conclusion,
        boundary=boundary,
    )
    figure.suptitle(
        f"HashAll 虚拟扩展：QPS 随逻辑 shard 数变化（{metadata['dataset_name']}）",
        fontproperties=font,
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.947,
        "4 台物理机；总 CPU=2M；M=32 使用全部 64 CPU；"
        f"Recall@10≥{metadata['target_recall']:.2f}；top-k={metadata['top_k']}",
        ha="center",
        fontproperties=font,
        fontsize=11,
        color="#334155",
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.04)
    return save_figure(figure, output_dir, "hashall-virtual-linear-qps-scaling")


def plot_resource_recall(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    font: FontProperties,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    machines = np.asarray([row["logical_shards"] for row in rows], dtype=float)
    allocated = np.asarray([row["allocated_cpu_cores"] for row in rows])
    used = np.asarray([row["cpu_used_cores_mean"] for row in rows])
    recall = np.asarray([row["heldout_recall_at_10"] for row in rows])
    ef = np.asarray([row["selected_ef"] for row in rows])

    figure = plt.figure(figsize=(15.2, 10.8))
    grid = figure.add_gridspec(2, 2, height_ratios=(3.1, 1.65), hspace=0.24, wspace=0.20)
    cpu_axis = figure.add_subplot(grid[0, 0])
    recall_axis = figure.add_subplot(grid[0, 1])
    text_axis = figure.add_subplot(grid[1, :])

    cpu_axis.plot(machines, allocated, "--o", color="#2f855a", linewidth=2.4, label="分配 CPU=2M")
    cpu_axis.plot(machines, used, "-s", color="#1769aa", linewidth=2.4, label="实测平均使用 CPU")
    cpu_axis.set_yscale("log", base=2)
    cpu_axis.set_xticks(machines, [str(int(value)) for value in machines])
    cpu_axis.set_xlim(0.0, 33.0)
    cpu_axis.set_yticks([2, 4, 8, 16, 32, 64], ["2", "4", "8", "16", "32", "64"])
    cpu_axis.set_xlabel("逻辑 shard 数 M", fontproperties=font, fontsize=11)
    cpu_axis.set_ylabel("CPU 核数", fontproperties=font, fontsize=11)
    cpu_axis.set_title("资源合同与实际使用", fontproperties=font, fontsize=13)
    cpu_axis.legend(prop=font, fontsize=9, framealpha=0.96)
    cpu_axis.grid(True)

    recall_axis.plot(machines, recall, "-o", color="#1769aa", linewidth=2.4, label="Held-out Recall@10")
    recall_axis.axhline(
        metadata["target_recall"],
        color="#dc2626",
        linestyle="--",
        linewidth=1.8,
        label=f"目标 Recall={metadata['target_recall']:.2f}",
    )
    recall_axis.set_xticks(machines, [str(int(value)) for value in machines])
    recall_axis.set_xlim(0.0, 33.0)
    recall_padding = max(0.005, 0.12 * float(np.ptp(recall)))
    recall_axis.set_ylim(
        min(metadata["target_recall"], float(np.min(recall))) - recall_padding,
        min(1.0, float(np.max(recall)) + recall_padding),
    )
    recall_axis.set_xlabel("逻辑 shard 数 M", fontproperties=font, fontsize=11)
    recall_axis.set_ylabel("Recall@10", fontproperties=font, fontsize=11)
    ef_axis = recall_axis.twinx()
    ef_axis.step(machines, ef, where="mid", color="#d97706", linewidth=2.2, label="选定 ef")
    ef_axis.scatter(machines, ef, color="#d97706", marker="D", s=40)
    ef_axis.set_ylabel("efSearch", fontproperties=font, fontsize=11, color="#b45309")
    ef_min = float(np.min(ef))
    ef_max = float(np.max(ef))
    ef_padding = max(2.0, 0.08 * (ef_max - ef_min or ef_max))
    ef_axis.set_ylim(max(0.0, ef_min - ef_padding), ef_max + ef_padding)
    handles1, labels1 = recall_axis.get_legend_handles_labels()
    handles2, labels2 = ef_axis.get_legend_handles_labels()
    recall_axis.legend(handles1 + handles2, labels1 + labels2, prop=font, fontsize=9, loc="upper left", framealpha=0.96)
    recall_axis.set_title("固定召回率调参与 ef 下限", fontproperties=font, fontsize=13)
    recall_axis.grid(True)

    purpose = (
        "证明各 M 点的计算资源确实按 2M 严格增长，并展示为保持 "
        f"Recall@10≥{metadata['target_recall']:.2f} 所选择的 ef 及实际 held-out 召回率。"
    )
    reading = "左图绿色线是配额合同，蓝线是正式重复中的平均 CPU 使用量；右图蓝线是独立 9000 个 query 的召回率，橙线是最小满足 tuning 门槛的 ef。"
    floor_indices = np.flatnonzero(ef <= metadata["ef_lower_bound"])
    if floor_indices.size:
        first_floor_m = int(machines[int(floor_indices[0])])
        ef_interpretation = (
            f"M={first_floor_m} 起触及实验允许的 ef 下限 "
            f"{metadata['ef_lower_bound']}，后续若 Recall 高于目标便无法再通过降低 ef 对齐。"
        )
    else:
        ef_interpretation = (
            f"所有点的 ef 都高于实验下限 {metadata['ef_lower_bound']}，说明该数据集在 "
            f"Recall={metadata['target_recall']:.2f} 附近仍需要较宽的候选搜索。"
        )
    conclusion = (
        f"资源从 2 CPU 线性增至 64 CPU，M=32 使用全部资源。ef 依次为 {'/'.join(str(int(value)) for value in ef)}；"
        + ef_interpretation
    )
    boundary = (
        "0.05 CPU 的无 shard 节点控制面保底已经从有 shard 节点预算中扣除，所以总配额仍精确等于 2M。"
        f"ef 下限为 {metadata['ef_lower_bound']}；本实验 top-k={metadata['top_k']}，"
        "下限用于保证搜索候选宽度不小于返回结果数。"
    )
    add_explanation_panel(
        text_axis,
        font,
        purpose=purpose,
        reading=reading,
        conclusion=conclusion,
        boundary=boundary,
    )
    figure.suptitle(
        "HashAll 虚拟扩展：线性资源与固定召回率审计",
        fontproperties=font,
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )
    figure.subplots_adjust(left=0.07, right=0.94, top=0.91, bottom=0.04)
    return save_figure(figure, output_dir, "hashall-virtual-linear-resource-recall")


def plot_efficiency(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    font: FontProperties,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    machines = np.asarray([row["logical_shards"] for row in rows], dtype=float)
    efficiency = 100.0 * np.asarray([row["linear_scaling_efficiency"] for row in rows])
    qps_per_core = np.asarray([row["qps_per_allocated_core"] for row in rows])
    cpu_per_query = np.asarray([row["cpu_us_per_query"] for row in rows])
    per_shard_cpu = np.asarray([row["cpu_us_per_query_per_shard"] for row in rows])

    figure = plt.figure(figsize=(15.2, 10.8))
    grid = figure.add_gridspec(2, 2, height_ratios=(3.1, 1.65), hspace=0.24, wspace=0.20)
    efficiency_axis = figure.add_subplot(grid[0, 0])
    work_axis = figure.add_subplot(grid[0, 1])
    text_axis = figure.add_subplot(grid[1, :])

    efficiency_axis.plot(machines, efficiency, "-o", color="#7c3aed", linewidth=2.5, label="线性扩展效率")
    efficiency_axis.set_xticks(machines, [str(int(value)) for value in machines])
    efficiency_axis.set_xlim(0.0, 33.0)
    efficiency_axis.set_xlabel("逻辑 shard 数 M", fontproperties=font, fontsize=11)
    efficiency_axis.set_ylabel("相对理想线性的效率（%）", fontproperties=font, fontsize=11)
    efficiency_axis.set_ylim(0, 105)
    efficiency_axis.grid(True)
    core_axis = efficiency_axis.twinx()
    core_axis.plot(machines, qps_per_core, "--s", color="#d97706", linewidth=2.2, label="QPS/分配 CPU")
    core_axis.set_ylabel("QPS / 分配 CPU 核", fontproperties=font, fontsize=11, color="#b45309")
    handles1, labels1 = efficiency_axis.get_legend_handles_labels()
    handles2, labels2 = core_axis.get_legend_handles_labels()
    efficiency_axis.legend(handles1 + handles2, labels1 + labels2, prop=font, fontsize=9, framealpha=0.96)
    efficiency_axis.set_title("吞吐扩展效率", fontproperties=font, fontsize=13)

    work_axis.plot(machines, cpu_per_query, "-o", color="#dc2626", linewidth=2.5, label="集群 CPU μs/query")
    work_axis.plot(machines, per_shard_cpu, "--s", color="#1769aa", linewidth=2.3, label="平均每 shard CPU μs/query")
    work_axis.set_yscale("log")
    work_axis.set_xticks(machines, [str(int(value)) for value in machines])
    work_axis.set_xlim(0.0, 33.0)
    work_axis.set_xlabel("逻辑 shard 数 M", fontproperties=font, fontsize=11)
    work_axis.set_ylabel("CPU 时间（μs/query）", fontproperties=font, fontsize=11)
    work_axis.set_title("局部工作下降与总工作放大", fontproperties=font, fontsize=13)
    work_axis.legend(prop=font, fontsize=9, framealpha=0.96)
    work_axis.grid(True)

    purpose = "解释为什么 QPS 能随 M 增长但不能接近线性：把局部 shard 搜索变快与一个全局 query 需要访问全部 shard 的累计工作分开观察。"
    reading = "左图显示相对 QPS(1)×M 的效率和单位分配 CPU 的吞吐；右图红线是一次全局 query 消耗的集群累计 CPU，蓝线将其除以 M，近似表示单 shard 工作。"
    speedup = float(rows[-1]["qps_relative_to_m1"])
    per_shard_ratio = float(per_shard_cpu[-1] / per_shard_cpu[0])
    conclusion = (
        f"单 shard CPU 工作从 {per_shard_cpu[0]:.1f} 降至 {per_shard_cpu[-1]:.1f} μs/query（{per_shard_ratio:.3f}×），"
        f"但全局 CPU 工作从 {cpu_per_query[0]:.1f} 增至 {cpu_per_query[-1]:.1f} μs/query（{cpu_per_query[-1]/cpu_per_query[0]:.2f}×）。"
        f"因此 M=32 达到 {speedup:.3f}× QPS，线性效率为 {efficiency[-1]:.2f}%。"
    )
    boundary = "CPU 时间来自四个 Qdrant 容器 cgroup 的平均使用量除以 QPS，包含服务端图搜索、调度和协调开销，但不包含客户端 CPU；它是系统工作量指标，不等同于纯 distance-computation 计数。"
    add_explanation_panel(
        text_axis,
        font,
        purpose=purpose,
        reading=reading,
        conclusion=conclusion,
        boundary=boundary,
    )
    figure.suptitle(
        f"HashAll 虚拟扩展：资源效率与每 query 工作量（{metadata['dataset_name']}）",
        fontproperties=font,
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )
    figure.subplots_adjust(left=0.07, right=0.94, top=0.91, bottom=0.04)
    return save_figure(figure, output_dir, "hashall-virtual-linear-efficiency")


def verify_pdf(path: Path) -> dict[str, Any]:
    document = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in document)
    pages = len(document)
    document.close()
    missing = [token for token in REQUIRED_PDF_TEXT if token not in text]
    if pages != 1 or missing:
        raise ValueError(f"PDF verification failed for {path}: pages={pages}, missing={missing}")
    return {"path": str(path.resolve()), "pages": pages, "required_text": "PASS"}


def render_report(
    rows: Sequence[dict[str, Any]],
    model: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    lines = [
        "# HashAll 四机虚拟 M=1～32 严格线性资源扩展实验",
        "",
        "## 实验结果",
        "",
        "| 逻辑 M | 物理机 | shard/节点 | CPU 总配额 | ef | Held-out Recall@10 | 饱和并发 | QPS | 相对 M=1 | 线性效率 | QPS CV |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {logical_shards} | 4 | {shards} | {allocated_cpu_cores:.0f} | {selected_ef} | "
            "{heldout_recall_at_10:.4f} | {selected_concurrency} | {qps_mean:,.0f} ± {qps_stdev:,.0f} | "
            "{qps_relative_to_m1:.3f}× | {efficiency:.2f}% | {cv:.3f}% |".format(
                **row,
                shards="/".join(map(str, row["shards_per_node"])),
                efficiency=100.0 * row["linear_scaling_efficiency"],
                cv=100.0 * row["qps_cv"],
            )
        )
    slope = model["fit_slope_qps_per_doubling"]
    fit_metrics = model["fit_metrics"]
    theory_metrics = model["theory_metrics"]
    speedup = float(rows[-1]["qps_relative_to_m1"])
    efficiency = 100.0 * float(rows[-1]["linear_scaling_efficiency"])
    ef_values = [int(row["selected_ef"]) for row in rows]
    floor_points = [
        int(row["logical_shards"])
        for row in rows
        if int(row["selected_ef"]) <= metadata["ef_lower_bound"]
    ]
    if floor_points:
        ef_statement = (
            f"M≥{floor_points[0]} 时 ef 触及实验下限 {metadata['ef_lower_bound']}；"
            "之后若 Recall 高于目标，无法再靠降低 ef 精确对齐，因此高 M 的 QPS 比较偏保守。"
        )
    else:
        ef_statement = (
            f"六个点的 ef 均高于实验下限 {metadata['ef_lower_bound']}，"
            f"实际序列为 {'/'.join(map(str, ef_values))}；这说明 {metadata['dataset_name']} "
            "在目标召回率附近仍需要较宽的搜索候选集。"
        )
    lines.extend(
        [
            "",
            "## 拟合与结论",
            "",
            f"实测对数拟合：`QPS(M) = {model['baseline_qps']:.1f} + {slope:.1f} × log2(M)`，"
            f"`R²={fit_metrics['r_squared']:.3f}`，MAPE={100*fit_metrics['mape']:.2f}%。",
            "",
            "理论辅助线采用：",
            "",
            "`D(M)=m_h[log_{m_h}(N/M)+0.5]+2m_h ef_R(M)`",
            "",
            "`QPS_theory(M)=QPS(1)D(1)/D(M)`",
            "",
            f"该理论线在六点上的 `R²={theory_metrics['r_squared']:.3f}`，MAPE={100*theory_metrics['mape']:.2f}%。",
            "",
            f"结论：QPS 从 {rows[0]['qps_mean']:,.1f} 增至 {rows[-1]['qps_mean']:,.1f}，"
            f"即 {speedup:.3f}×；M=32 的线性效率为 {efficiency:.2f}%。"
            f"对数拟合 R²={fit_metrics['r_squared']:.3f}，理论辅助线 R²={theory_metrics['r_squared']:.3f}；"
            "这两个数用于描述本次六点曲线，不能预先当作普适复杂度结论。",
            "",
            ef_statement,
            "",
            "## 证据边界",
            "",
            "- 全部实验只使用 4 台真实机器；M=8/16/32 是一台机器承载多个逻辑 shard，不是对应数量的独立机器。",
            "- CPU quota 严格为 2M，M=32 才开放完整 64 CPU。构建阶段使用全资源，只有调参与正式测量阶段实施线性 quota。",
            f"- 数据集为 {metadata['dataset_name']}、{metadata['qdrant_distance']}、"
            f"{metadata['train_count']:,} 个 {metadata['vector_size']} 维向量、top-{metadata['top_k']}、"
            f"HNSW m={metadata['hnsw_m']:g}、efConstruct={metadata['ef_construct']}、原生 Qdrant HashAll。",
            "- 理论辅助线采用单位系数，不能外推为其他数据集、更多真实网络节点或通用 HNSW 定律。",
            "",
            "## 图表",
            "",
            "- `figures/hashall-virtual-linear-qps-scaling.pdf`",
            "- `figures/hashall-virtual-linear-resource-recall.pdf`",
            "- `figures/hashall-virtual-linear-efficiency.pdf`",
            "",
        ]
    )
    return "\n".join(lines)


def create_manifest(
    root: Path,
    output_dir: Path,
    generated: Iterable[Path],
    pdf_audits: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    preflight = root / "preflight.json"
    sources = [{"path": str(preflight.resolve()), "sha256": sha256(preflight)}]
    for logical_shards in LOGICAL_SHARDS:
        for filename in ("prepare.json", "resources-measure.json", "benchmark.json", "cleanup.json"):
            path = root / f"m{logical_shards}" / filename
            sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
    restore = root / "resource-restored.json"
    sources.append({"path": str(restore.resolve()), "sha256": sha256(restore)})
    artifacts = [
        {"path": str(path.resolve()), "sha256": sha256(path)} for path in generated
    ]
    return {
        "status": "PASS",
        "record_type": "hashall_virtual_linear_cpu_manifest",
        "physical_machines": PHYSICAL_MACHINES,
        "logical_shards": list(LOGICAL_SHARDS),
        "dataset": metadata,
        "source_files": sources,
        "generated_artifacts": artifacts,
        "pdf_audits": list(pdf_audits),
        "output_directory": str(output_dir.resolve()),
    }


def verify_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "PASS":
        raise ValueError("manifest is not PASS")
    for group in ("source_files", "generated_artifacts"):
        for entry in manifest[group]:
            path = Path(entry["path"])
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError(f"manifest verification failed: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/proj/intelisys-PG0/exp/orion-distributed/"
            "hashall-virtual-linear-cpu-20260824"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "summary"
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata = experiment_metadata(root)
    rows, checks = audit_and_collect(root)
    model = add_derived_columns(
        rows,
        dataset_size=metadata["train_count"],
        hnsw_m=metadata["hnsw_m"],
    )
    summary_csv = write_summary_csv(output_dir / "summary.csv", rows)
    model_json = write_json(
        output_dir / "models.json",
        {
            "status": "PASS",
            "dataset": metadata,
            "theory": {
                "distance_proxy": "m_h * (log_m_h(N/M) + 0.5) + 2*m_h*ef_R(M)",
                "qps": "QPS(1) * D(1) / D(M)",
                "unit_coefficients": True,
                "metrics": model["theory_metrics"],
            },
            "empirical_fit": {
                "formula": "QPS(1) + a*log2(M)",
                "qps_1": model["baseline_qps"],
                "a_qps_per_doubling": model["fit_slope_qps_per_doubling"],
                "metrics": model["fit_metrics"],
            },
            "old_no_ef_reference": {
                "formula": "QPS(1)*ln(N)/ln(N/M)",
                "metrics": model["old_metrics"],
            },
            "evidence_boundary": (
                f"six {metadata['dataset_name']} points on four physical hosts; "
                "M>4 is logical-shard virtualization"
            ),
        },
    )

    configure_plot_style()
    font, temporary_font = chinese_font()
    try:
        qps_pdf, qps_png = plot_qps_scaling(
            figures_dir, rows, model, font, metadata
        )
        resource_pdf, resource_png = plot_resource_recall(
            figures_dir, rows, font, metadata
        )
        efficiency_pdf, efficiency_png = plot_efficiency(
            figures_dir, rows, font, metadata
        )
    finally:
        Path(temporary_font).unlink(missing_ok=True)

    report_path = output_dir / "RESULTS_zh.md"
    report_path.write_text(render_report(rows, model, metadata), encoding="utf-8")
    audit_path = write_json(
        output_dir / "completion-audit.json",
        {
            "status": "PASS",
            "record_type": "hashall_virtual_linear_cpu_completion_audit",
            "physical_machines": PHYSICAL_MACHINES,
            "logical_shards": list(LOGICAL_SHARDS),
            "dataset": metadata,
            "all_measurements_present": True,
            "check_count": len(checks),
            "checks": checks,
            "headline": {
                "qps_m1": rows[0]["qps_mean"],
                "qps_m32": rows[-1]["qps_mean"],
                "speedup_m32": rows[-1]["qps_relative_to_m1"],
                "linear_efficiency_m32": rows[-1]["linear_scaling_efficiency"],
                "empirical_log_fit_r_squared": model["fit_metrics"]["r_squared"],
                "theory_r_squared": model["theory_metrics"]["r_squared"],
            },
        },
    )
    completion_path = write_json(
        output_dir / "execution-complete-all.json",
        {
            "status": "PASS",
            "record_type": "hashall_virtual_linear_cpu_complete_all",
            "physical_machines": PHYSICAL_MACHINES,
            "logical_shards": list(LOGICAL_SHARDS),
            "dataset": metadata,
            "full_cluster_cores": FULL_CLUSTER_CORES,
            "cores_per_logical_shard": CORES_PER_LOGICAL_SHARD,
            "summary_csv": str(summary_csv.resolve()),
            "report": str(report_path.resolve()),
            "completion_audit": str(audit_path.resolve()),
        },
    )

    pdf_audits = [verify_pdf(path) for path in (qps_pdf, resource_pdf, efficiency_pdf)]
    generated = [
        summary_csv,
        model_json,
        qps_pdf,
        qps_png,
        resource_pdf,
        resource_png,
        efficiency_pdf,
        efficiency_png,
        report_path,
        audit_path,
        completion_path,
    ]
    manifest = create_manifest(root, output_dir, generated, pdf_audits, metadata)
    manifest_path = write_json(output_dir / "manifest.json", manifest)
    verify_manifest(load_json(manifest_path))
    print(
        json.dumps(
            {
                "status": "PASS",
                "summary_csv": str(summary_csv.resolve()),
                "report": str(report_path.resolve()),
                "manifest": str(manifest_path.resolve()),
                "qps_m1": rows[0]["qps_mean"],
                "qps_m32": rows[-1]["qps_mean"],
                "speedup_m32": rows[-1]["qps_relative_to_m1"],
                "log_fit_r_squared": model["fit_metrics"]["r_squared"],
                "theory_r_squared": model["theory_metrics"]["r_squared"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
