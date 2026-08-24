#!/usr/bin/env python3
"""Fit and plot a fixed-recall HNSW plus HashAll resource model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pymupdf


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


HOSTS = ["10.10.1.1", "10.10.1.2", "10.10.1.3", "10.10.1.4"]
FOOTER_HEIGHT = 244.0


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def chinese_font() -> tuple[FontProperties, str]:
    font_buffer = pymupdf.Font(fontname="china-s").buffer
    handle = tempfile.NamedTemporaryFile(suffix=".ttf", delete=False)
    handle.write(font_buffer)
    handle.close()
    return FontProperties(fname=handle.name), handle.name


def annotate_pdf(
    source_pdf: Path,
    destination: Path,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    shard_r2: float,
    max_error_pct: float,
) -> None:
    source = pymupdf.open(source_pdf)
    output = pymupdf.open()
    font_buffer = pymupdf.Font(fontname="china-s").buffer
    purpose = (
        "在 Recall@10≥0.90 且 ef 随 M 校准的条件下，用逐节点 CPU 时间建立 HNSW/HashAll"
        "工作量模型，并将模型 QPS 与实测值及理想线性扩展比较。"
    )
    reading = (
        "橙色点线只包含单 shard 模型 T_shard=α+β·ef(M)·ln(N/M)；紫色虚线进一步加入"
        "coordinator/fan-out 项 γ(M-1)。模型参数来自每台 Qdrant 容器的 CPU 时间，"
        "最终 QPS 只用 M=1 起点归一化。右图放大非线性区域。"
    )
    conclusion = (
        f"拟合得到 α={alpha:.1f} μs、β={beta:.3f} μs、γ={gamma:.1f} μs，"
        f"单 shard 工作量拟合 R²={shard_r2:.3f}。仅考虑局部 HNSW 会预测 M=4 约 97.3k QPS；"
        f"加入 HashAll fan-out 后预测约 82.3k，与实测 82.3k 接近，四点最大相对误差约"
        f" {max_error_pct:.1f}%。"
    )
    boundary = (
        "模型只有 M=1～4 四个规模点，ef 与 M 同时变化，属于当前 SIFT1M/当前硬件上的"
        "半经验辅助模型，不是 HNSW 的普适复杂度定理。若要外推，应补做每个 M 内的 ef 扫描、"
        "距离计算数和 graph_nodes_visited 采样。"
    )
    body = (
        f"用途：{purpose}\n\n"
        f"如何理解：{reading}\n\n"
        f"最终结论：{conclusion}\n\n"
        f"证据边界：{boundary}"
    )
    for page_index, source_page in enumerate(source):
        rectangle = source_page.rect
        page = output.new_page(
            width=rectangle.width, height=rectangle.height + FOOTER_HEIGHT
        )
        page.show_pdf_page(rectangle, source, page_index, keep_proportion=False)
        page.draw_line(
            pymupdf.Point(28, rectangle.height + 14),
            pymupdf.Point(rectangle.width - 28, rectangle.height + 14),
            color=(0.68, 0.73, 0.80),
            width=0.8,
        )
        page.insert_font(fontname="hashall-model-cjk", fontbuffer=font_buffer)
        page.insert_textbox(
            pymupdf.Rect(
                32, rectangle.height + 23, rectangle.width - 32, rectangle.height + 49
            ),
            "中文图解",
            fontname="hashall-model-cjk",
            fontsize=13,
            color=(0.05, 0.22, 0.48),
        )
        body_rectangle = pymupdf.Rect(
            32,
            rectangle.height + 53,
            rectangle.width - 32,
            rectangle.height + FOOTER_HEIGHT - 12,
        )
        for fontsize in (10.2, 9.8, 9.4, 9.0):
            result = page.insert_textbox(
                body_rectangle,
                body,
                fontname="hashall-model-cjk",
                fontsize=fontsize,
                lineheight=1.20,
                color=(0.07, 0.07, 0.08),
            )
            if result >= 0:
                break
        else:
            raise RuntimeError("Chinese model explanation does not fit")
    temporary = destination.with_name(destination.stem + ".tmp.pdf")
    output.save(temporary, garbage=4, deflate=True)
    output.close()
    source.close()
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    bench_files = {
        1: args.root / "bench" / "m1-clean.json",
        2: args.root / "bench" / "m2-clean.json",
        3: args.root / "bench" / "m3-clean.json",
        4: args.root / "bench" / "m4-clean-confirm.json",
    }
    machines = np.asarray([1.0, 2.0, 3.0, 4.0])
    dataset_size = 1_000_000.0
    records = [load(bench_files[int(machine)]) for machine in machines]
    qps = np.asarray([record["qps_mean"] for record in records], dtype=float)
    qps_std = np.asarray([record["qps_stdev"] for record in records], dtype=float)
    ef = np.asarray(
        [record["parameters"]["selected_hnsw_ef"] for record in records],
        dtype=float,
    )

    shard_cpu_us = []
    coordinator_cpu_us = []
    for machine, record in zip(machines.astype(int), records, strict=True):
        mean_cores = {
            host: statistics.fmean(
                repeat["cpu_average_cores"][host] for repeat in record["repeats"]
            )
            for host in HOSTS
        }
        coordinator_cpu_us.append(mean_cores[HOSTS[0]] / record["qps_mean"] * 1e6)
        if machine == 1:
            shard_cpu_us.append(coordinator_cpu_us[-1])
        else:
            shard_cpu_us.append(
                statistics.fmean(
                    mean_cores[HOSTS[index]] / record["qps_mean"] * 1e6
                    for index in range(1, machine)
                )
            )
    shard_cpu_us = np.asarray(shard_cpu_us)
    coordinator_cpu_us = np.asarray(coordinator_cpu_us)

    hnsw_feature = ef * np.log(dataset_size / machines)
    design = np.column_stack([np.ones(len(machines)), hnsw_feature])
    alpha, beta = np.linalg.lstsq(design, shard_cpu_us, rcond=None)[0]
    predicted_shard_cpu_us = design @ np.asarray([alpha, beta])
    shard_r2 = 1.0 - np.sum((shard_cpu_us - predicted_shard_cpu_us) ** 2) / np.sum(
        (shard_cpu_us - np.mean(shard_cpu_us)) ** 2
    )

    fanout_feature = machines - 1.0
    coordinator_residual = coordinator_cpu_us - predicted_shard_cpu_us
    gamma = float(
        np.dot(fanout_feature[1:], coordinator_residual[1:])
        / np.dot(fanout_feature[1:], fanout_feature[1:])
    )
    predicted_hashall_cpu_us = predicted_shard_cpu_us + gamma * fanout_feature

    baseline_qps = qps[0]
    baseline_shard_time = predicted_shard_cpu_us[0]
    local_hnsw_qps = baseline_qps * baseline_shard_time / predicted_shard_cpu_us
    hashall_model_qps = baseline_qps * baseline_shard_time / predicted_hashall_cpu_us
    linear_qps = baseline_qps * machines
    relative_error = (hashall_model_qps - qps) / qps
    max_error_pct = float(np.max(np.abs(relative_error)) * 100.0)

    font, temporary_font = chinese_font()
    try:
        plt.rcParams.update(
            {
                "axes.unicode_minus": False,
                "axes.grid": True,
                "grid.alpha": 0.22,
                "grid.linestyle": "--",
                "figure.facecolor": "white",
                "axes.facecolor": "#fbfdff",
            }
        )
        figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.7), constrained_layout=True)
        figure.suptitle(
            "固定召回率下的 HNSW + HashAll 复杂度模型（SIFT1M）",
            fontproperties=font,
            fontsize=16,
            fontweight="bold",
        )
        colors = {
            "observed": "#1769aa",
            "linear": "#2e8b57",
            "local": "#d97706",
            "hashall": "#7c3aed",
        }
        for axis in axes:
            axis.errorbar(
                machines,
                qps,
                yerr=qps_std,
                color=colors["observed"],
                marker="o",
                markersize=7,
                linewidth=2.5,
                capsize=4,
                label="实测饱和 QPS",
                zorder=4,
            )
            axis.plot(
                machines,
                local_hnsw_qps,
                color=colors["local"],
                marker="s",
                linestyle=":",
                linewidth=2.3,
                label="仅局部 HNSW 模型",
            )
            axis.plot(
                machines,
                hashall_model_qps,
                color=colors["hashall"],
                marker="D",
                linestyle="-.",
                linewidth=2.3,
                label="HNSW + HashAll fan-out 模型",
            )
            axis.set_xlabel("真实机器数 / shard 数 M", fontproperties=font, fontsize=11)
            axis.set_ylabel("QPS", fontproperties=font, fontsize=11)
            axis.set_xticks(machines)
        axes[0].plot(
            machines,
            linear_qps,
            color=colors["linear"],
            linestyle="--",
            linewidth=2.0,
            label="理想线性扩展",
        )
        axes[0].set_ylim(50_000, 290_000)
        axes[0].set_title("全尺度比较", fontproperties=font, fontsize=12)
        axes[0].legend(prop=font, loc="upper left", framealpha=0.93)
        axes[1].set_ylim(64_000, 102_000)
        axes[1].set_title("非线性区域放大", fontproperties=font, fontsize=12)
        axes[1].legend(prop=font, loc="upper left", framealpha=0.93)
        for machine, observed, modeled in zip(
            machines, qps, hashall_model_qps, strict=True
        ):
            separated = modeled - observed > 500.0
            axes[1].annotate(
                f"{observed / 1000:.1f}k",
                (machine, observed),
                textcoords="offset points",
                xytext=(0, -18 if separated else 9),
                ha="center",
                fontsize=8.8,
                color=colors["observed"],
                fontproperties=font,
            )
            axes[1].annotate(
                f"{modeled / 1000:.1f}k",
                (machine, modeled),
                textcoords="offset points",
                xytext=(0, 9 if separated else -18),
                ha="center",
                fontsize=8.4,
                color=colors["hashall"],
                fontproperties=font,
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        png_path = args.output_dir / "hashall-fixed-recall-complexity-model.png"
        base_pdf = args.output_dir / "hashall-fixed-recall-complexity-model.base.pdf"
        pdf_path = args.output_dir / "hashall-fixed-recall-complexity-model.pdf"
        figure.savefig(png_path, dpi=200, bbox_inches="tight")
        figure.savefig(base_pdf, bbox_inches="tight")
        plt.close(figure)
        annotate_pdf(
            base_pdf,
            pdf_path,
            alpha=float(alpha),
            beta=float(beta),
            gamma=gamma,
            shard_r2=float(shard_r2),
            max_error_pct=max_error_pct,
        )
        base_pdf.unlink()
    finally:
        Path(temporary_font).unlink(missing_ok=True)

    output_rows = []
    for index, machine in enumerate(machines.astype(int)):
        output_rows.append(
            {
                "machine_count": machine,
                "ef": int(ef[index]),
                "hnsw_feature_ef_log_n_over_m": hnsw_feature[index],
                "measured_shard_cpu_us_per_query": shard_cpu_us[index],
                "modeled_shard_cpu_us_per_query": predicted_shard_cpu_us[index],
                "measured_coordinator_cpu_us_per_query": coordinator_cpu_us[index],
                "modeled_hashall_cpu_us_per_query": predicted_hashall_cpu_us[index],
                "observed_qps": qps[index],
                "local_hnsw_model_qps": local_hnsw_qps[index],
                "hashall_model_qps": hashall_model_qps[index],
                "hashall_model_relative_error": relative_error[index],
                "linear_qps": linear_qps[index],
            }
        )
    csv_path = args.output_dir / "hashall-fixed-recall-complexity-model.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    metadata = {
        "model": {
            "shard_time": "alpha + beta * ef(M) * ln(N/M)",
            "hashall_time": "shard_time + gamma * (M-1)",
            "qps": "QPS(1) * hashall_time(1) / hashall_time(M)",
            "alpha_cpu_us": float(alpha),
            "beta_cpu_us": float(beta),
            "gamma_cpu_us_per_extra_shard": gamma,
            "shard_fit_r_squared": float(shard_r2),
            "max_absolute_qps_relative_error": max_error_pct / 100.0,
        },
        "fit_source": "per-node cgroup CPU time; final QPS is used only for M=1 normalization",
        "evidence_boundary": "four SIFT1M M=1..4 points; semi-empirical, not a universal HNSW law",
        "outputs": {
            "pdf": str(pdf_path),
            "png": str(png_path),
            "csv": str(csv_path),
        },
    }
    json_path = args.output_dir / "hashall-fixed-recall-complexity-model.json"
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
