#!/usr/bin/env python3
"""Plot native HashAll QPS against linear and inverse-log reference curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pymupdf


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


FOOTER_HEIGHT = 230.0


def chinese_font() -> tuple[FontProperties, str]:
    font_buffer = pymupdf.Font(fontname="china-s").buffer
    handle = tempfile.NamedTemporaryFile(suffix=".ttf", delete=False)
    handle.write(font_buffer)
    handle.close()
    return FontProperties(fname=handle.name), handle.name


def annotate_pdf(source_pdf: Path, destination: Path) -> None:
    source = pymupdf.open(source_pdf)
    output = pymupdf.open()
    font_buffer = pymupdf.Font(fontname="china-s").buffer
    purpose = (
        "呈现原生 HashAll 在 1～4 台真实机器上的饱和 QPS，并与从 M=1 同一点出发的"
        "理想线性扩展和由单机 HNSW 复杂度 ln(N/M) 推导出的 k/ln(N/M) QPS 辅助线比较。"
    )
    reading = (
        "蓝色实线和误差棒是实测值；绿色虚线是 QPS(1)×M；橙色点线是"
        "k/ln(N/M)，其中使用自然对数且 k=QPS(1)·ln(N)，所以三条线在 M=1 重合。"
        "右图把各曲线除以 M=1 的起点，以便直接读取扩展倍数。"
    )
    conclusion = (
        "实测 QPS 为 68.6k→77.7k→78.9k→82.3k，M=4 仅为起点的 1.199×，"
        "远低于线性辅助线的 4×，但高于固定其他因素时逆对数复杂度辅助线的约 1.112×。"
        "额外提升主要来自实测 ef 随 M 从 24 降到 14，而辅助线只表达局部索引规模变化。"
    )
    boundary = (
        "数据仅对应 SIFT1M、Recall@10≥0.90、HNSW m=32、efConstruct=200、"
        "Qdrant 原生服务端 HashAll 和本次硬件配置；辅助线用于比较，不构成复杂度证明。"
    )
    body = (
        f"用途：{purpose}\n\n"
        f"如何理解：{reading}\n\n"
        f"最终结论：{conclusion}\n\n"
        f"证据边界：{boundary}"
    )
    for index, source_page in enumerate(source):
        rectangle = source_page.rect
        page = output.new_page(
            width=rectangle.width, height=rectangle.height + FOOTER_HEIGHT
        )
        page.show_pdf_page(rectangle, source, index, keep_proportion=False)
        page.draw_line(
            pymupdf.Point(28, rectangle.height + 14),
            pymupdf.Point(rectangle.width - 28, rectangle.height + 14),
            color=(0.68, 0.73, 0.80),
            width=0.8,
        )
        page.insert_font(fontname="hashall-cjk", fontbuffer=font_buffer)
        page.insert_textbox(
            pymupdf.Rect(
                32, rectangle.height + 23, rectangle.width - 32, rectangle.height + 49
            ),
            "中文图解",
            fontname="hashall-cjk",
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
                fontname="hashall-cjk",
                fontsize=fontsize,
                lineheight=1.20,
                color=(0.07, 0.07, 0.08),
            )
            if result >= 0:
                break
        else:
            raise RuntimeError("Chinese explanation does not fit below the chart")
    temporary = destination.with_name(destination.stem + ".tmp.pdf")
    output.save(temporary, garbage=4, deflate=True)
    output.close()
    source.close()
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    machines = np.asarray([int(row["machine_count"]) for row in rows], dtype=float)
    qps = np.asarray([float(row["qps_mean"]) for row in rows], dtype=float)
    qps_std = np.asarray([float(row["qps_stdev"]) for row in rows], dtype=float)
    if not np.array_equal(machines, np.asarray([1.0, 2.0, 3.0, 4.0])):
        raise ValueError("expected M=1,2,3,4")

    dataset_size = 1_000_000.0
    baseline = float(qps[0])
    k_inverse_log = baseline * math.log(dataset_size)
    dense_m = np.linspace(1.0, 4.0, 301)
    linear_dense = baseline * dense_m
    inverse_log_dense = k_inverse_log / np.log(dataset_size / dense_m)
    linear_points = baseline * machines
    inverse_log_points = k_inverse_log / np.log(dataset_size / machines)

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
            "HashAll：QPS 随真实机器数 M 的变化（SIFT1M）",
            fontproperties=font,
            fontsize=16,
            fontweight="bold",
        )

        observed_color = "#1769aa"
        linear_color = "#2e8b57"
        log_color = "#d97706"
        left, right = axes
        left.errorbar(
            machines,
            qps,
            yerr=qps_std,
            color=observed_color,
            marker="o",
            markersize=7,
            linewidth=2.5,
            capsize=4,
            label="实测饱和 QPS",
        )
        left.plot(
            dense_m,
            linear_dense,
            color=linear_color,
            linestyle="--",
            linewidth=2.0,
            label="理想线性：QPS(1)×M",
        )
        left.plot(
            dense_m,
            inverse_log_dense,
            color=log_color,
            linestyle=":",
            linewidth=2.4,
            label=f"逆对数复杂度：k/ln(1,000,000/M)，k={k_inverse_log:.1f}",
        )
        for machine, value in zip(machines, qps, strict=True):
            left.annotate(
                f"{value / 1000:.1f}k",
                (machine, value),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
                color=observed_color,
            )
        left.set_xlabel("真实机器数 / shard 数 M", fontproperties=font, fontsize=11)
        left.set_ylabel("QPS", fontproperties=font, fontsize=11)
        left.set_xticks(machines)
        left.set_ylim(50_000, 290_000)
        left.legend(prop=font, loc="upper left", framealpha=0.92)
        left.set_title("绝对 QPS", fontproperties=font, fontsize=12)

        right.errorbar(
            machines,
            qps / baseline,
            yerr=qps_std / baseline,
            color=observed_color,
            marker="o",
            markersize=7,
            linewidth=2.5,
            capsize=4,
            label="实测 / QPS(1)",
        )
        right.plot(
            dense_m,
            linear_dense / baseline,
            color=linear_color,
            linestyle="--",
            linewidth=2.0,
            label="线性辅助线",
        )
        right.plot(
            dense_m,
            inverse_log_dense / baseline,
            color=log_color,
            linestyle=":",
            linewidth=2.4,
            label="k/ln(N/M) 辅助线",
        )
        for machine, value in zip(machines, qps / baseline, strict=True):
            right.annotate(
                f"{value:.3f}×",
                (machine, value),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
                color=observed_color,
            )
        right.set_xlabel("真实机器数 / shard 数 M", fontproperties=font, fontsize=11)
        right.set_ylabel("相对 M=1 的倍数", fontproperties=font, fontsize=11)
        right.set_xticks(machines)
        right.set_ylim(0.84, 4.18)
        right.legend(prop=font, loc="upper left", framealpha=0.92)
        right.set_title("归一化扩展倍数", fontproperties=font, fontsize=12)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        png_path = args.output_dir / "hashall-qps-vs-m-reference-curves.png"
        base_pdf = args.output_dir / "hashall-qps-vs-m-reference-curves.base.pdf"
        pdf_path = args.output_dir / "hashall-qps-vs-m-reference-curves.pdf"
        figure.savefig(png_path, dpi=200, bbox_inches="tight")
        figure.savefig(base_pdf, bbox_inches="tight")
        plt.close(figure)
        annotate_pdf(base_pdf, pdf_path)
        base_pdf.unlink()
    finally:
        Path(temporary_font).unlink(missing_ok=True)

    curve_csv = args.output_dir / "hashall-qps-reference-curves.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "machine_count",
                "observed_qps",
                "observed_qps_stdev",
                "linear_qps",
                "k_over_log_1000000_over_m_qps",
                "observed_relative_to_m1",
                "linear_relative_to_m1",
                "inverse_log_relative_to_m1",
            ]
        )
        for index, machine in enumerate(machines.astype(int)):
            writer.writerow(
                [
                    machine,
                    qps[index],
                    qps_std[index],
                    linear_points[index],
                    inverse_log_points[index],
                    qps[index] / baseline,
                    linear_points[index] / baseline,
                    inverse_log_points[index] / baseline,
                ]
            )
    metadata = {
        "dataset": "SIFT1M",
        "dataset_size": int(dataset_size),
        "baseline_qps": baseline,
        "linear_formula": "QPS_linear(M) = QPS_observed(1) * M",
        "inverse_log_formula": "QPS_inverse_log_reference(M) = k / ln(1000000 / M)",
        "log_base": "natural",
        "k": k_inverse_log,
        "start_points_coincide": True,
        "interpretation": "if local HNSW cost is proportional to ln(N/M), ideal per-query QPS is proportional to 1/ln(N/M)",
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "curve_csv": str(curve_csv),
        },
    }
    (args.output_dir / "hashall-qps-reference-curves.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
