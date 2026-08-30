#!/usr/bin/env python3
"""Plot HashAll, plain Simple KMeans, and the updated Orion evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
MAX_RECALL = 0.93
MAX_CV = 0.05
EXPECTED_ORION = (
    ("m1-exact-s1", 1, "exact", 1),
    ("m2-exact-s2", 2, "exact", 2),
    ("m4-exact-s4", 4, "exact", 4),
    ("m8-exact-s8", 8, "exact", 8),
    ("m16-exact-s16", 16, "exact", 16),
    ("m32-exact-s32", 32, "exact", 32),
)
REQUIRED_PDF_TEXT = (
    "图表作用",
    "如何理解",
    "最终结论",
    "证据边界",
    "线性数值坐标",
)
SIMPLE_DISABLED = {
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


def protocol_signature(benchmark: dict[str, Any]) -> tuple[Any, ...]:
    protocol = benchmark.get("protocol") or {}
    parameters = benchmark.get("parameters") or {}
    return (
        protocol.get("dataset"),
        protocol.get("distance"),
        protocol.get("vector_size"),
        protocol.get("tuning_query_range"),
        protocol.get("heldout_query_range"),
        parameters.get("top_k"),
        parameters.get("batch_size"),
        parameters.get("sweep_seconds"),
        parameters.get("warmup_seconds"),
        parameters.get("measure_seconds"),
    )


EXPECTED_SIGNATURE = (
    "GloVe-200-angular",
    "Cosine",
    200,
    [0, 1000],
    [1000, 10000],
    10,
    200,
    8.0,
    10.0,
    20.0,
)


def audit_point(
    method: str,
    machine: int,
    benchmark: dict[str, Any],
    resources: dict[str, Any],
    checks: list[dict[str, str]],
) -> None:
    prefix = f"{method} M={machine}"
    require(benchmark.get("status") == "PASS", f"{prefix} benchmark PASS", checks)
    require(resources.get("status") == "PASS", f"{prefix} resources PASS", checks)
    require(
        abs(float(resources.get("quota_cores_total", -1.0)) - 2.0 * machine) < 1e-9,
        f"{prefix} CPU equals 2M",
        checks,
    )
    recall = float((benchmark.get("heldout_recall") or {}).get("recall_at_10", 0.0))
    require(
        TARGET_RECALL <= recall < MAX_RECALL,
        f"{prefix} recall is in [0.90, 0.93)",
        checks,
    )
    require(
        float(benchmark.get("qps_cv", 1.0)) <= MAX_CV,
        f"{prefix} QPS CV <= 5%",
        checks,
    )
    require(
        protocol_signature(benchmark) == EXPECTED_SIGNATURE,
        f"{prefix} common query protocol",
        checks,
    )
    require(
        int((benchmark.get("parameters") or {}).get("repeat_count", 0)) >= 5,
        f"{prefix} has at least five QPS repeats",
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


def load_hashall(root: Path, checks: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for machine in M_VALUES.astype(int):
        benchmark_path = root / f"m{machine}/benchmark.json"
        benchmark = load_json(benchmark_path)
        resources = load_json(root / f"m{machine}/resources-measure.json")
        preparation = load_json(root / f"m{machine}/prepare.json")
        audit_point("HashAll", machine, benchmark, resources, checks)
        body = preparation.get("collection_body") or {}
        hnsw = body.get("hnsw_config") or {}
        optimizer = body.get("optimizers_config") or {}
        require(
            hnsw.get("m") == 32
            and hnsw.get("ef_construct") == 200
            and optimizer.get("max_segment_size_kb") == 2_000_000,
            f"HashAll M={machine} common lower-index contract",
            checks,
        )
        require(
            (benchmark.get("request_contract") or {}).get("hashall_fanout") == machine,
            f"HashAll M={machine} visits every shard",
            checks,
        )
        rows.append(
            {
                "method": "HashAll",
                "nominal_m": machine,
                "actual_shards": machine,
                "recall": float(benchmark["heldout_recall"]["recall_at_10"]),
                "qps": float(benchmark["qps_mean"]),
                "qps_stdev": float(benchmark["qps_stdev"]),
                "qps_cv": float(benchmark["qps_cv"]),
                "expansion_ratio": 1.0,
                "allocated_cpu": float(resources["quota_cores_total"]),
                "source": str(benchmark_path),
            }
        )
    audit = load_json(root / "summary/completion-audit.json")
    require(audit.get("status") == "PASS", "HashAll source audit PASS", checks)
    return rows


def load_simple(root: Path, checks: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for machine in M_VALUES.astype(int):
        benchmark_path = root / f"m{machine}/benchmark.json"
        benchmark = load_json(benchmark_path)
        resources = load_json(root / f"m{machine}/resources-measure.json")
        preparation = load_json(root / f"m{machine}/prepare.json")
        audit_point("Simple KMeans", machine, benchmark, resources, checks)
        hnsw = (preparation.get("preparation_manifest") or {}).get("hnsw") or {}
        require(
            hnsw.get("m") == 32
            and hnsw.get("ef_construct") == 200
            and hnsw.get("max_segment_size_kb") == 2_000_000,
            f"Simple KMeans M={machine} common lower-index contract",
            checks,
        )
        baseline = benchmark.get("baseline_contract") or {}
        disabled = baseline.get("enhancements_disabled") or {}
        require(
            baseline.get("baseline") == "native_static_single_assignment_simple_kmeans"
            and float(baseline.get("expansion_ratio", -1.0)) == 1.0
            and baseline.get("logical_point_count") == baseline.get("physical_point_count"),
            f"Simple KMeans M={machine} is plain single-assignment baseline",
            checks,
        )
        require(
            set(disabled) == SIMPLE_DISABLED and all(value is True for value in disabled.values()),
            f"Simple KMeans M={machine} enhancements disabled",
            checks,
        )
        require(
            (benchmark.get("request_contract") or {}).get("server_router")
            == "native_simple_kmeans",
            f"Simple KMeans M={machine} native router",
            checks,
        )
        rows.append(
            {
                "method": "Simple KMeans",
                "nominal_m": machine,
                "actual_shards": machine,
                "recall": float(benchmark["heldout_recall"]["recall_at_10"]),
                "qps": float(benchmark["qps_mean"]),
                "qps_stdev": float(benchmark["qps_stdev"]),
                "qps_cv": float(benchmark["qps_cv"]),
                "expansion_ratio": float(baseline["expansion_ratio"]),
                "allocated_cpu": float(resources["quota_cores_total"]),
                "source": str(benchmark_path),
            }
        )
    restore = load_json(root / "resource-restored.json")
    require(restore.get("status") == "PASS", "Simple KMeans resource restore PASS", checks)
    return rows


def load_orion(root: Path, checks: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point_id, machine, boundary, actual_shards in EXPECTED_ORION:
        point = root / f"m{machine}/{boundary}-s{actual_shards}"
        benchmark_path = point / "benchmark.json"
        benchmark = load_json(benchmark_path)
        resources = load_json(point / "resources-measure.json")
        audit_point(f"Orion {point_id}", machine, benchmark, resources, checks)
        require(
            benchmark.get("point_id") == point_id
            and benchmark.get("actual_shards") == actual_shards
            and (benchmark.get("request_contract") or {}).get("server_router") == "native_orion",
            f"Orion {point_id} identity and native router",
            checks,
        )
        profile = benchmark.get("profile_contract") or {}
        assignment = (
            (profile.get("l1_partition_layout_proof") or {}).get("assignment") or {}
        )
        protocol = benchmark.get("protocol") or {}
        live_info = (benchmark.get("live_policy_gate") or {}).get(
            "collection_info"
        ) or {}
        live_config = live_info.get("config") or {}
        live_hnsw = live_config.get("hnsw_config") or {}
        live_optimizer = live_config.get("optimizer_config") or {}
        require(
            profile.get("method") == "native_orion_c_cnbr_bmr10"
            and profile.get("fixed_p_exact") is True
            and profile.get("actual_shards") == machine,
            f"Orion {point_id} is exact fixed-P C_CNBR+BMR_10",
            checks,
        )
        require(
            profile.get("logical_point_count") == 1_183_514
            and 1.0 <= float(profile.get("expansion_ratio", 0.0)) <= 1.10,
            f"Orion {point_id} copy budget is at most 10 percent",
            checks,
        )
        require(
            protocol.get("fission_enabled") is False
            and protocol.get("logical_shard_rule") == "exact fixed P = nominal M"
            and protocol.get("load_balancing_policy") == "C_CNBR"
            and protocol.get("multi_assignment_policy") == "BMR_10",
            f"Orion {point_id} routing policy contract",
            checks,
        )
        require(
            live_hnsw.get("m") == 32
            and live_hnsw.get("ef_construct") == 200
            and live_optimizer.get("max_segment_size") == 2_000_000,
            f"Orion {point_id} common lower-index contract",
            checks,
        )
        rows.append(
            {
                "method": "Orion",
                "point_id": point_id,
                "nominal_m": machine,
                "boundary": boundary,
                "actual_shards": actual_shards,
                "recall": float(benchmark["heldout_recall"]["recall_at_10"]),
                "qps": float(benchmark["qps_mean"]),
                "qps_stdev": float(benchmark["qps_stdev"]),
                "qps_cv": float(benchmark["qps_cv"]),
                "expansion_ratio": float(profile["expansion_ratio"]),
                "upper_k": int(profile["upper_k"]),
                "dynamic_ef_base": int(profile["dynamic_ef_base"]),
                "dynamic_ef_factor": int(profile["dynamic_ef_factor"]),
                "max_shard_physical_points": int(
                    assignment["physical_copy_load_max"]
                ),
                "shard_load_max_over_mean": float(
                    assignment["physical_copy_load_max_over_mean"]
                ),
                "shard_load_cv": float(assignment["physical_copy_load_cv"]),
                "allocated_cpu": float(resources["quota_cores_total"]),
                "source": str(benchmark_path),
            }
        )
    execution = load_json(root / "execution-complete.json")
    require(
        execution.get("status") == "MEASUREMENTS_COMPLETE"
        and execution.get("completed_points") == [row[0] for row in EXPECTED_ORION],
        "Orion all six exact fixed-P layouts complete",
        checks,
    )
    restore = load_json(root / "resource-restored.json")
    restore_nodes = restore.get("nodes") or []
    require(
        restore.get("status") == "PASS"
        and len(restore_nodes) == 4
        and all(
            node.get("valid") is True
            and node.get("exact_docker_metadata_restored") is True
            for node in restore_nodes
        ),
        "Orion four-node resource restore PASS",
        checks,
    )
    return rows


def load_p32_research_winner(
    summary_path: Path, checks: list[dict[str, str]]
) -> dict[str, Any]:
    summary_path = summary_path.expanduser().resolve()
    summary = load_json(summary_path)
    completion = load_json(summary_path.with_name("completion-audit.json"))
    gate = summary.get("owner_policy_finalist_online_qps_gate") or {}
    require(summary.get("status") == "PASS", "P=32 owner A/B summary PASS", checks)
    require(completion.get("status") == "PASS", "P=32 owner A/B audit PASS", checks)
    require(
        gate.get("passed") is True
        and gate.get("decision") == "ARM_B_OWNER_POLICY_QPS_GATE_PASS",
        "P=32 owner-policy finalist online QPS gate PASS",
        checks,
    )
    require(
        summary.get("pair_count") == 5
        and summary.get("extended_to_seven") is False,
        "P=32 owner A/B has five stable pairs",
        checks,
    )
    recall = float(summary["heldout_recall_at_10"]["B"])
    qps = summary["qps"]["B"]
    ratio = summary["paired_ratio_arm_b_over_arm_a"][
        "confidence_interval"
    ]
    require(
        TARGET_RECALL <= recall < MAX_RECALL,
        "P=32 historical-primary+single_rank recall is in [0.90, 0.93)",
        checks,
    )
    require(
        float(qps["cv"]) <= MAX_CV,
        "P=32 historical-primary+single_rank QPS CV <= 5%",
        checks,
    )
    require(
        float(ratio["lower"]) > 1.0,
        "P=32 historical-primary paired CI lower bound > 1",
        checks,
    )
    return {
        "method": "Orion P=32 owner-policy winner",
        "point_id": "p32-historical-primary-single-rank",
        "nominal_m": 32,
        "boundary": "P32-only research candidate",
        "actual_shards": 32,
        "allocated_cpu": 64.0,
        "recall": recall,
        "qps": float(qps["mean"]),
        "qps_stdev": float(qps["stdev"]),
        "qps_cv": float(qps["cv"]),
        "expansion_ratio": 1.0,
        "source": str(summary_path),
        "paired_ratio_geomean": float(ratio["geometric_mean"]),
        "paired_ratio_ci_lower": float(ratio["lower"]),
        "paired_ratio_ci_upper": float(ratio["upper"]),
        "full_curve_claim": False,
    }


def arrays(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([row["nominal_m"] for row in rows], dtype=float),
        np.asarray([row["qps"] for row in rows], dtype=float),
        np.asarray([row["qps_stdev"] for row in rows], dtype=float),
    )


def chinese_font() -> tuple[FontProperties, str]:
    buffer = pymupdf.Font(fontname="china-s").buffer
    handle = tempfile.NamedTemporaryFile(suffix=".ttf", delete=False)
    handle.write(buffer)
    handle.close()
    return FontProperties(fname=handle.name), handle.name


def wrap_cjk(text: str, width: int = 82) -> str:
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
            fontsize=11.3,
            fontweight="bold",
            color=color,
        )
        axis.text(
            0.105,
            y,
            wrap_cjk(body),
            transform=axis.transAxes,
            va="top",
            fontproperties=font,
            fontsize=10.2,
            color="#1f2937",
            linespacing=1.23,
        )


def configure_axis(axis: plt.Axes, font: FontProperties) -> None:
    axis.set_xscale("linear")
    axis.set_xlim(0.0, 33.0)
    axis.set_xticks(M_VALUES, [str(int(value)) for value in M_VALUES])
    axis.set_xlabel("名义机器数 M / 资源扩展倍数（线性数值坐标）", fontproperties=font)
    axis.set_ylabel("QPS", fontproperties=font)
    axis.grid(True, linestyle="--", alpha=0.30)


def plot_observed(
    axis: plt.Axes,
    hash_rows: Sequence[dict[str, Any]],
    simple_rows: Sequence[dict[str, Any]],
    orion_rows: Sequence[dict[str, Any]],
    p32_winner: dict[str, Any],
    font: FontProperties,
    *,
    annotate_shards: bool,
    value_scale: float = 1.0,
) -> None:
    colors = {"HashAll": "#1769aa", "Simple KMeans": "#d97706", "Orion": "#7c3aed"}
    hash_m, hash_q, hash_std = arrays(hash_rows)
    simple_m, simple_q, simple_std = arrays(simple_rows)
    orion_m, orion_q, orion_std = arrays(orion_rows)
    hash_q, hash_std = hash_q / value_scale, hash_std / value_scale
    simple_q, simple_std = simple_q / value_scale, simple_std / value_scale
    orion_q, orion_std = orion_q / value_scale, orion_std / value_scale
    axis.errorbar(
        hash_m,
        hash_q,
        yerr=hash_std,
        color=colors["HashAll"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        capsize=4,
        label="HashAll 实测",
        zorder=5,
    )
    axis.errorbar(
        simple_m,
        simple_q,
        yerr=simple_std,
        color=colors["Simple KMeans"],
        marker="s",
        linewidth=2.5,
        markersize=7,
        capsize=4,
        label="原生 Simple KMeans 实测",
        zorder=5,
    )
    axis.errorbar(
        orion_m,
        orion_q,
        yerr=orion_std,
        fmt="D-",
        color=colors["Orion"],
        linewidth=2.6,
        markersize=6,
        capsize=4,
        label="Orion（C_CNBR+BMR_10）实测",
        zorder=6,
    )
    axis.errorbar(
        [float(p32_winner["nominal_m"])],
        [float(p32_winner["qps"]) / value_scale],
        yerr=[float(p32_winner["qps_stdev"]) / value_scale],
        fmt="*",
        color="#dc2626",
        markeredgecolor="#7f1d1d",
        markeredgewidth=0.8,
        markersize=15,
        capsize=4,
        label="P=32 最佳组合：历史 primary+single_rank",
        zorder=8,
    )
    configure_axis(axis, font)


def save_chart(
    output_dir: Path,
    filename: str,
    hash_rows: Sequence[dict[str, Any]],
    simple_rows: Sequence[dict[str, Any]],
    orion_rows: Sequence[dict[str, Any]],
    p32_winner: dict[str, Any],
    font: FontProperties,
    *,
    include_linear: bool,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.facecolor": "#fbfdff",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(12.4, 12.0) if include_linear else (14.8, 10.3))
    grid = figure.add_gridspec(2, 1, height_ratios=(3.25, 1.65), hspace=0.22)
    axis = figure.add_subplot(grid[0, 0])
    text_axis = figure.add_subplot(grid[1, 0])
    hashall_m1_qps = float(hash_rows[0]["qps"])
    m32 = {
        "HashAll": float(hash_rows[-1]["qps"]),
        "Simple KMeans": float(simple_rows[-1]["qps"]),
        "Orion": float(orion_rows[-1]["qps"]),
    }
    research_qps = float(p32_winner["qps"])
    winner = max(m32, key=m32.get)
    plot_observed(
        axis,
        hash_rows,
        simple_rows,
        orion_rows,
        p32_winner,
        font,
        annotate_shards=not include_linear,
        value_scale=hashall_m1_qps if include_linear else 1.0,
    )
    if include_linear:
        axis.plot(
            M_VALUES,
            M_VALUES,
            linestyle="--",
            linewidth=2.2,
            color="#111827",
            label="HashAll 线性辅助线：y=M（斜率=1）",
            zorder=1,
        )
        axis.set_xlim(0.0, 33.0)
        axis.set_ylim(0.0, 33.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_ylabel(
            f"归一化 QPS（QPS ÷ HashAll M=1 的 {hashall_m1_qps:.2f}）",
            fontproperties=font,
        )
        purpose = "在统一归一化坐标中比较三种方法的吞吐扩展，并只保留 HashAll 的理想线性辅助线。归一化后该辅助线严格为 y=M，数学斜率等于 1。"
        reading = f"所有实测 QPS 都除以 HashAll 在 M=1 的 {hashall_m1_qps:.2f} QPS，因此 HashAll 的理想线性值依次为 1、2、4、8、16、32。黑色虚线是图中唯一辅助线；其他三条曲线均来自实测。"
        conclusion = (
            f"M=32 时 {winner} 的实测 QPS 最高。归一化值分别为："
            f"HashAll {m32['HashAll'] / hashall_m1_qps:.2f}、Simple KMeans "
            f"{m32['Simple KMeans'] / hashall_m1_qps:.2f}、Orion "
            f"{m32['Orion'] / hashall_m1_qps:.2f}；理想线性值为 32。"
            f"红色星号是仅在 P=32 完成在线验证的最佳 owner-policy 组合，归一化值为 "
            f"{research_qps / hashall_m1_qps:.2f}，不连成扩展曲线。"
        )
        title = "三种分片方法的 QPS 扩展：唯一线性辅助线斜率为 1"
    else:
        observed_max = max(
            max(float(row["qps"]) for row in hash_rows),
            max(float(row["qps"]) for row in simple_rows),
            max(float(row["qps"]) for row in orion_rows),
            research_qps,
        )
        axis.set_ylim(0.0, observed_max * 1.13)
        purpose = "放大三种方法的实际 QPS 区域，去掉所有辅助模型与线性参考，使方法间的交叉、差距和 Orion 布局波动可以直接读取。"
        reading = "蓝线是 HashAll，橙线是无改良的原生静态单重分配 Simple KMeans，紫线是逐点完成拓展实验的固定-P C_CNBR+BMR_10 Orion。红色星号是九组合筛选后由 held-out 配对确认的历史 primary+single_rank P=32 winner；它没有 P=1..16 的同族实测点，因此不连线。"
        conclusion = (
            f"M=32 时 QPS 为 HashAll {m32['HashAll']:.2f}、Simple KMeans "
            f"{m32['Simple KMeans']:.2f}、Orion {m32['Orion']:.2f}；"
            f"完整扩展曲线由 {winner} 领先；另有 P=32 owner-policy winner "
            f"{research_qps:.2f} QPS。各 M 的完整差异应直接按曲线和数据表读取。"
        )
        title = "三种分片方法的 QPS 扩展：实测区域局部放大"
    boundary = "数据为 GloVe-200-angular、Recall@10∈[0.90,0.93)、总 CPU=2M、逻辑 shard 数 P=M。系统只有四台物理主机；M=8/16/32 是四台物理机上的逻辑 shard 与算力扩展，不代表对应数量的真实机器。"
    explanation_panel(
        text_axis,
        font,
        purpose=purpose,
        reading=reading,
        conclusion=conclusion,
        boundary=boundary,
    )
    figure.suptitle(title, fontproperties=font, fontsize=19, fontweight="bold", y=0.984)
    figure.text(
        0.5,
        0.947,
        "GloVe-200-angular；Recall@10≈0.90；总 Qdrant CPU=2M；横轴为线性数值坐标",
        ha="center",
        fontproperties=font,
        fontsize=11,
        color="#334155",
    )
    axis.legend(prop=font, fontsize=8.6, loc="upper left", framealpha=0.95, ncol=2)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.035)
    pdf = output_dir / f"{filename}.pdf"
    png = output_dir / f"{filename}.png"
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
        raise RuntimeError(f"PDF explanation tokens missing in {path}: {missing}")
    tofu = text.count("□") + text.count("�")
    if tofu:
        raise RuntimeError(f"PDF contains {tofu} missing-glyph characters: {path}")
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
        "expansion_ratio",
        "upper_k",
        "dynamic_ef_base",
        "dynamic_ef_factor",
        "max_shard_physical_points",
        "shard_load_max_over_mean",
        "shard_load_cv",
        "source",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_results_markdown(
    path: Path,
    hash_rows: Sequence[dict[str, Any]],
    simple_rows: Sequence[dict[str, Any]],
    orion_rows: Sequence[dict[str, Any]],
    p32_winner: dict[str, Any],
) -> None:
    lines = [
        "# 更新后的 Orion 横向扩展实验结果",
        "",
        "实验条件：GloVe-200-angular、Cosine、Top-10、Held-out Recall@10∈[0.90,0.93)、逻辑 shard 数 P=M、总 Qdrant CPU=2M。",
        "",
        "| M | 总 CPU | HashAll Recall | HashAll QPS | Simple KMeans Recall | Simple KMeans QPS | Orion Recall | Orion QPS | Orion / HashAll | Orion 索引膨胀 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hash_row, simple_row, orion_row in zip(
        hash_rows, simple_rows, orion_rows, strict=True
    ):
        lines.append(
            "| {m} | {cpu:.0f} | {hr:.5f} | {hq:.2f} | {sr:.5f} | {sq:.2f} | "
            "{orr:.5f} | {oq:.2f} | {ratio:.3f}x | {expansion:.6f}x |".format(
                m=int(hash_row["nominal_m"]),
                cpu=float(orion_row["allocated_cpu"]),
                hr=float(hash_row["recall"]),
                hq=float(hash_row["qps"]),
                sr=float(simple_row["recall"]),
                sq=float(simple_row["qps"]),
                orr=float(orion_row["recall"]),
                oq=float(orion_row["qps"]),
                ratio=float(orion_row["qps"]) / float(hash_row["qps"]),
                expansion=float(orion_row["expansion_ratio"]),
            )
        )
    orion_speedup = float(orion_rows[-1]["qps"]) / float(orion_rows[0]["qps"])
    hash_speedup = float(hash_rows[-1]["qps"]) / float(hash_rows[0]["qps"])
    simple_speedup = float(simple_rows[-1]["qps"]) / float(simple_rows[0]["qps"])
    m32_values = {
        "HashAll": float(hash_rows[-1]["qps"]),
        "Simple KMeans": float(simple_rows[-1]["qps"]),
        "Orion": float(orion_rows[-1]["qps"]),
    }
    winner = max(m32_values, key=m32_values.get)
    lines.extend(
        [
            "",
            "## Orion 路由与负载证据",
            "",
            "| M | upper_k | dynamic ef (base/factor) | 最大 shard 物理点数 | shard max/mean | shard load CV |",
            "|---:|---:|---:|---:|---:|---:|",
            *[
                "| {m} | {upper_k} | {base}/{factor} | {max_points} | {max_mean:.3f} | {cv:.3f} |".format(
                    m=int(row["nominal_m"]),
                    upper_k=int(row["upper_k"]),
                    base=int(row["dynamic_ef_base"]),
                    factor=int(row["dynamic_ef_factor"]),
                    max_points=int(row["max_shard_physical_points"]),
                    max_mean=float(row["shard_load_max_over_mean"]),
                    cv=float(row["shard_load_cv"]),
                )
                for row in orion_rows
            ],
            "",
            "## 结论",
            "",
            f"- 从 M=1 到 M=32，HashAll、Simple KMeans、Orion 的实测加速比分别为 `{hash_speedup:.3f}x`、`{simple_speedup:.3f}x`、`{orion_speedup:.3f}x`。",
            f"- M=32 时吞吐最高的是 `{winner}`；三者 QPS 分别为 HashAll `{m32_values['HashAll']:.2f}`、Simple KMeans `{m32_values['Simple KMeans']:.2f}`、Orion `{m32_values['Orion']:.2f}`。",
            f"- Orion 曲线不是单调线性扩展：M=1 到 M=2 从 `{float(orion_rows[0]['qps']):.2f}` 降到 `{float(orion_rows[1]['qps']):.2f}`，M=4 后才逐步回升。固定 Recall 窗口并不保证不同 M 的检索预算完全相同，因此应结合实际 Recall、路由参数和 shard 负载解释。",
            f"- M=2 的最大 shard 仍有 `{int(orion_rows[1]['max_shard_physical_points']):,}` 个物理点，是平均值的 `{float(orion_rows[1]['shard_load_max_over_mean']):.3f}x`；同时 upper_k 从 M=1 的 `{int(orion_rows[0]['upper_k'])}` 增至 `{int(orion_rows[1]['upper_k'])}`。最大 shard 仅小幅缩小，而路由/检索预算和多 shard 协调增加，可以解释 M=2 没有获得吞吐扩展。该实验没有单独计时路由、RPC 和合并，因此这仍是与证据一致的解释，而不是已隔离的因果分解。",
            f"- 到 M=16/32，最大 shard 已降至 `{int(orion_rows[4]['max_shard_physical_points']):,}` / `{int(orion_rows[5]['max_shard_physical_points']):,}` 个物理点，绝对检索规模下降与更多 CPU 的收益开始超过协调和副本开销。相对负载不均衡并未消失，说明不能用单一 balance 指标替代在线 QPS。",
            "- Orion 在 M=2/4/8 低于 HashAll，在 M=1/16/32 高于 HashAll；本轮结果支持大规模逻辑扩展点的收益，但不支持全区间单调或线性扩展。",
            "- 上述比较只使用落入同一 Recall 窗口的点；没有用更高召回率的 Orion 点替代正式结果。",
            "",
            "## P=32 owner × 多分配策略在线 tournament",
            "",
            "| 策略 | Held-out Recall@10 | QPS | 相对 historical primary+BMR_10 | 95% 配对 CI |",
            "|---|---:|---:|---:|---:|",
            (
                f"| 历史 primary+single_rank | {float(p32_winner['recall']):.5f} | "
                f"{float(p32_winner['qps']):.2f} | "
                f"{float(p32_winner['paired_ratio_geomean']):.4f}x | "
                f"[{float(p32_winner['paired_ratio_ci_lower']):.4f}, "
                f"{float(p32_winner['paired_ratio_ci_upper']):.4f}] |"
            ),
            "",
            "- 该候选在 P=32 的五轮配对中 5/5 胜出，但 historical primary 是 checksum-bound 的 P=32 owner，尚未定义并实测同族 P=1..16 owner，因此图中以红色星号表示，不连接为 M=1..32 的扩展曲线。",
            "- 完整紫色扩展曲线仍是逐点实测的 C_CNBR+BMR_10；不能用单个 P=32 winner 替换其它 M 的数据。",
            "",
            "## 证据边界",
            "",
            "- 物理部署始终只有四台主机。M=1/2/4 可对应不超过四台物理主机的资源点；M=8/16/32 是四台物理主机上的逻辑 shard 与 CPU 配额扩展，不能称为 8/16/32 台真实机器。",
            "- 结论限定于本次 GloVe-200-angular、Cosine、Qdrant 版本、C_CNBR+BMR_10 bundle、批大小 200 和总 CPU=2M 的合同，不直接外推为所有数据集或所有分布式 HNSW 的通用规律。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashall-root", type=Path, required=True)
    parser.add_argument("--simple-root", type=Path, required=True)
    parser.add_argument("--orion-root", type=Path, required=True)
    parser.add_argument("--p32-ab-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, str]] = []
    hash_rows = load_hashall(args.hashall_root, checks)
    simple_rows = load_simple(args.simple_root, checks)
    orion_rows = load_orion(args.orion_root, checks)
    p32_winner = load_p32_research_winner(args.p32_ab_summary, checks)
    require(
        [row["nominal_m"] for row in hash_rows]
        == [row["nominal_m"] for row in simple_rows]
        == M_VALUES.astype(int).tolist(),
        "HashAll and Simple KMeans cover exact M=1..32",
        checks,
    )
    output_csv = args.output_dir / "three-method-qps.csv"
    write_csv(output_csv, [*hash_rows, *simple_rows, *orion_rows, p32_winner])
    results_md = args.output_dir / "RESULTS_zh.md"
    write_results_markdown(
        results_md, hash_rows, simple_rows, orion_rows, p32_winner
    )
    font, temporary_font = chinese_font()
    try:
        full_pdf, full_png = save_chart(
            args.output_dir,
            "three-method-qps-with-linear-guides",
            hash_rows,
            simple_rows,
            orion_rows,
            p32_winner,
            font,
            include_linear=True,
        )
        zoom_pdf, zoom_png = save_chart(
            args.output_dir,
            "three-method-qps-observed-zoom",
            hash_rows,
            simple_rows,
            orion_rows,
            p32_winner,
            font,
            include_linear=False,
        )
    finally:
        Path(temporary_font).unlink(missing_ok=True)
    pdfs = [verify_pdf(full_pdf), verify_pdf(zoom_pdf)]
    audit = {
        "status": "PASS",
        "check_count": len(checks),
        "checks": checks,
        "dataset": "GloVe-200-angular",
        "target_recall_level": "[0.90, 0.93)",
        "resource_rule": "total Qdrant CPU = 2M",
        "methods": [
            "HashAll",
            "plain native Simple KMeans",
            "native Orion C_CNBR+BMR_10 fixed-P",
            "P=32 owner-policy winner historical primary+single_rank",
        ],
        "full_scale_chart": {
            "linear_guides": "HashAll only: y=M after dividing every QPS by HashAll M=1 QPS",
            "hashall_m1_qps_normalizer": float(hash_rows[0]["qps"]),
            "linear_guide_data_slope": 1.0,
            "other_model_or_fit_guides": False,
            "pdf": pdfs[0],
            "png": {"path": str(full_png), "sha256": sha256(full_png)},
        },
        "observed_zoom_chart": {
            "linear_guides": False,
            "other_model_or_fit_guides": False,
            "pdf": pdfs[1],
            "png": {"path": str(zoom_png), "sha256": sha256(zoom_png)},
        },
        "data": {"path": str(output_csv), "sha256": sha256(output_csv)},
        "p32_research_winner": {
            "full_curve_claim": False,
            "summary": str(args.p32_ab_summary.expanduser().resolve()),
            "recall": float(p32_winner["recall"]),
            "qps": float(p32_winner["qps"]),
            "paired_ratio_geomean": float(p32_winner["paired_ratio_geomean"]),
            "paired_ratio_ci": [
                float(p32_winner["paired_ratio_ci_lower"]),
                float(p32_winner["paired_ratio_ci_upper"]),
            ],
        },
        "results_markdown": {
            "path": str(results_md),
            "sha256": sha256(results_md),
        },
        "x_axis": {
            "scale": "linear",
            "ticks": M_VALUES.tolist(),
            "physical_machine_boundary": "M>4 is logical shard/resource scaling on four physical hosts",
        },
    }
    audit_path = args.output_dir / "completion-audit.json"
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, audit_path)
    print(json.dumps({"status": "PASS", "output_dir": str(args.output_dir), "checks": len(checks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
