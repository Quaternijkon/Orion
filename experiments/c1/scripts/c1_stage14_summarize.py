#!/usr/bin/env python3
"""Summarize the Stage 14 physical-core-isolated QPS retest.

Physical QPS comes only from Stage 14.  Fixed-ef fan-out, finite-ef
sensitivity, and partition evidence are reused from Stage 13 as declared by
the Stage 14 protocol.  Every PDF receives a Chinese explanatory footer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt

import c1_stage13_summarize as base


SCRIPT_DIR = Path(__file__).resolve().parent
C1_DIR = SCRIPT_DIR.parent
STAGE14_DIR = C1_DIR / "retests" / "stage14-physical-core-isolated"
SUMMARY_DIR = STAGE14_DIR / "summary"
FIGURES_DIR = SUMMARY_DIR / "figures"
STAGE = "stage14-physical-core-isolated"

DATASETS = base.DATASETS
DATASET_LABELS = base.DATASET_LABELS
METHODS = base.METHODS
METHOD_LABELS = base.METHOD_LABELS
COLORS = base.COLORS
MARKERS = base.MARKERS
PHYSICAL_COUNTS = base.PHYSICAL_COUNTS
LOGICAL_COUNTS = base.LOGICAL_COUNTS
FIXED_EF = base.FIXED_EF
EF_GRIDS = base.EF_GRIDS
TARGET_RECALL = base.TARGET_RECALL
CV_THRESHOLD = base.CV_THRESHOLD

base.FOOTER_TITLE = "Stage 14 中文图解"
base.PDF_CREATOR = "Orion C1 Stage 14 summarizer"


def e1_path(dataset: str, method: str, machines: int, baseline: str = "a") -> Path:
    directory = STAGE14_DIR / "e1"
    if machines == 1:
        return directory / f"stage14-e1-{dataset}-random-m1-baseline-{baseline}.json"
    return directory / f"stage14-e1-{dataset}-{method}-m{machines}-primary.json"


def load_physical_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    baseline_b_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method in METHODS:
            rows.append(
                base.e1_row(
                    e1_path(dataset, method, 1, "a"),
                    display_method=method,
                    stage=STAGE,
                    baseline_label="A",
                )
            )
            for machines in PHYSICAL_COUNTS[1:]:
                rows.append(
                    base.e1_row(
                        e1_path(dataset, method, machines),
                        display_method=method,
                        stage=STAGE,
                    )
                )
        baseline_b_rows.append(
            base.e1_row(
                e1_path(dataset, "random", 1, "b"),
                display_method="random",
                stage=STAGE,
                baseline_label="B",
            )
        )
    base.add_normalization(rows)
    return rows, baseline_b_rows


def row_for(
    rows: Iterable[dict[str, Any]], dataset: str, method: str, machines: int
) -> dict[str, Any]:
    return base.row_for(rows, dataset, method, machines)


def audit_cleanup() -> dict[str, Any]:
    paths = sorted((STAGE14_DIR / "cleanup").glob("*.json"))
    if len(paths) != 16:
        raise ValueError(f"expected 16 Stage 14 cleanup proofs, found {len(paths)}")
    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in paths:
        payload = base.load_json(path)
        peer_codes = {
            host: int(details["http_status"])
            for host, details in payload["peers"].items()
        }
        controller_path = Path(payload["controller_storage_path"])
        valid = (
            payload.get("status") == "VERIFIED_DELETED"
            and payload.get("controller_storage_absent") is True
            and not controller_path.exists()
            and len(peer_codes) == 4
            and set(peer_codes.values()) == {404}
        )
        if not valid:
            failures.append(path.name)
        entries.append(
            {
                "collection": payload["collection"],
                "proof": str(path.resolve()),
                "proof_sha256": base.sha256(path),
                "peer_http_statuses": peer_codes,
                "controller_storage_absent_current": not controller_path.exists(),
                "valid": valid,
            }
        )
    if failures:
        raise ValueError(f"cleanup audit failures: {failures}")
    return {"status": "PASS", "proof_count": len(entries), "entries": entries}


def audit_affinity() -> dict[str, Any]:
    applied_path = STAGE14_DIR / "worker-affinity-applied.json"
    restored_path = STAGE14_DIR / "worker-affinity-restored.json"
    preflight_path = STAGE14_DIR / "preflight.json"
    applied = base.load_json(applied_path)
    restored = base.load_json(restored_path)
    preflight = base.load_json(preflight_path)
    apply_valid = (
        applied.get("status") == "PASS"
        and len(applied.get("nodes") or []) == 4
        and all(
            node["after"]["cpuset"] == "0-7,16-23"
            and node["runtime_qdrant_affinity"] == "0-7,16-23"
            for node in applied["nodes"]
        )
    )
    restore_valid = (
        restored.get("status") == "PASS"
        and len(restored.get("nodes") or []) == 4
        and all(
            node.get("valid") is True
            and node["restored"]["cpuset"] == "0-19"
            and node["runtime_qdrant_affinity"] == "0-19"
            for node in restored["nodes"]
        )
        and str(restored["timestamp"]) > str(applied["timestamp"])
    )
    preflight_valid = (
        preflight.get("status") == "PASS"
        and preflight.get("physical_core_overlap") == []
        and preflight.get("worker_cpu_affinity")
        == [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
        and preflight.get("benchmark_cpu_affinity")
        == [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]
    )
    if not (apply_valid and restore_valid and preflight_valid):
        raise ValueError(
            "affinity audit failed: "
            f"apply={apply_valid} restore={restore_valid} preflight={preflight_valid}"
        )
    return {
        "status": "PASS",
        "worker_test_cpuset": "0-7,16-23",
        "benchmark_test_cpuset": "8-15,24-31",
        "physical_core_overlap": [],
        "restored_cpuset": "0-19",
        "applied": str(applied_path.resolve()),
        "applied_sha256": base.sha256(applied_path),
        "restored": str(restored_path.resolve()),
        "restored_sha256": base.sha256(restored_path),
        "preflight": str(preflight_path.resolve()),
        "preflight_sha256": base.sha256(preflight_path),
    }


def save_figure(
    figure: Any,
    filename: str,
    *,
    purpose: str,
    reading: str,
    conclusion: str,
    boundary: str,
) -> Path:
    return base.save_annotated_figure(
        figure,
        FIGURES_DIR / filename,
        purpose=purpose,
        reading=reading,
        conclusion=conclusion,
        boundary=boundary,
    )


def plot_physical_qps(
    rows: Sequence[dict[str, Any]], baseline_b_rows: Sequence[dict[str, Any]]
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for method in METHODS:
            current = [row_for(rows, dataset, method, m) for m in PHYSICAL_COUNTS]
            y = [row["qps_mean"] for row in current]
            low = [row["qps_mean"] - row["qps_ci95_lower"] for row in current]
            high = [row["qps_ci95_upper"] - row["qps_mean"] for row in current]
            axis.errorbar(
                PHYSICAL_COUNTS,
                y,
                yerr=[low, high],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.2,
                capsize=4,
                label=METHOD_LABELS[method],
            )
        baseline_b = next(row for row in baseline_b_rows if row["dataset"] == dataset)
        axis.errorbar(
            [1.08],
            [baseline_b["qps_mean"]],
            yerr=[
                [baseline_b["qps_mean"] - baseline_b["qps_ci95_lower"]],
                [baseline_b["qps_ci95_upper"] - baseline_b["qps_mean"]],
            ],
            color="#111827",
            marker="x",
            markersize=8,
            capsize=3,
            linestyle="none",
            label="Baseline-B",
        )
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Physical machines M (real hosts)")
        axis.set_ylabel("Completed QPS")
        axis.set_xticks(PHYSICAL_COUNTS)
        base.style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Physical-core-isolated QPS on exactly 1-4 real hosts", y=0.985)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    sift_r4 = row_for(rows, "sift1m", "random", 4)["qps_mean"]
    sift_k4 = row_for(rows, "sift1m", "kmeans", 4)["qps_mean"]
    glove_r4 = row_for(rows, "glove-200-angular", "random", 4)["qps_mean"]
    glove_k4 = row_for(rows, "glove-200-angular", "kmeans", 4)["qps_mean"]
    return save_figure(
        figure,
        "stage14_fig1_physical_qps.pdf",
        purpose="在 worker 与压测客户端完全不共享物理核后，重新测量恰好 1、2、3、4 台真实机器的端到端 QPS。",
        reading="横轴 M 是真实主机数，同时每台主机放一个逻辑分片；点为重复均值，误差条为 95% 置信区间，M=1 旁叉号是全部分布式测点结束后的 baseline-B。",
        conclusion=(
            f"M=4 时 SIFT Random/K-Means 分别为 {sift_r4:,.0f}/{sift_k4:,.0f} QPS；"
            f"GloVe 分别为 {glove_r4:,.0f}/{glove_k4:,.0f} QPS。增加机器没有产生接近线性的四倍吞吐。"
        ),
        boundary="本图只使用 Stage 14 物理核隔离后的正式测量；旧 Stage 13 曲线因 worker/client SMT 同核而不再作为 QPS 主证据。",
    )


def plot_normalized_scaling(rows: Sequence[dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for axis, dataset in zip(axes, DATASETS, strict=True):
        axis.plot(
            PHYSICAL_COUNTS,
            PHYSICAL_COUNTS,
            color="#4b5563",
            linestyle="--",
            marker=".",
            label="Ideal linear",
        )
        for method in METHODS:
            current = [row_for(rows, dataset, method, m) for m in PHYSICAL_COUNTS]
            axis.plot(
                PHYSICAL_COUNTS,
                [row["normalized_qps_to_m1"] for row in current],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.2,
                label=METHOD_LABELS[method],
            )
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Physical machines M (real hosts)")
        axis.set_ylabel("QPS / M=1 QPS")
        axis.set_xticks(PHYSICAL_COUNTS)
        axis.set_ylim(bottom=0)
        base.style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Normalized physical scaling after core isolation", y=0.985)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    ratios = {
        (dataset, method): row_for(rows, dataset, method, 4)["normalized_qps_to_m1"]
        for dataset in DATASETS
        for method in METHODS
    }
    return save_figure(
        figure,
        "stage14_fig2_normalized_scaling.pdf",
        purpose="把单机 QPS 归一化为 1，直接比较增加真实机器后的扩展倍数与理想线性扩展。",
        reading="灰色虚线 y=M 是理想值；M=4 的曲线点若为 2，表示四台机器只有单机两倍吞吐；低于 1 表示加机器后反而更慢。",
        conclusion=(
            f"M=4 相对单机：SIFT Random {ratios[('sift1m','random')]:.2f}×、K-Means {ratios[('sift1m','kmeans')]:.2f}×；"
            f"GloVe Random {ratios[('glove-200-angular','random')]:.2f}×、K-Means {ratios[('glove-200-angular','kmeans')]:.2f}×，均明显低于理想 4×。"
        ),
        boundary="Random 与 K-Means 的 M=1 共用相同单分片基线；归一化描述曲线形状，不意味着两种路由具有相同工作量。",
    )


def plot_fanout(rows: Sequence[dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    positions = list(range(len(LOGICAL_COUNTS)))
    for axis, dataset in zip(axes, DATASETS, strict=True):
        current = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: row["logical_shard_count"],
        )
        selected = [row["holdout_selected_fanout"] for row in current]
        axis.plot(positions, LOGICAL_COUNTS, "--", color="#6b7280", label="Broadcast P=M")
        axis.plot(positions, selected, "s-", color="#d97706", linewidth=2.4, label=f"Selected P (ef={FIXED_EF[dataset]})")
        axis.plot(positions, [row["oracle_fanout_mean"] for row in current], "o-.", color="#059669", label="Oracle mean")
        axis.plot(positions, [row["oracle_fanout_p95"] for row in current], "^-.", color="#0f766e", label="Oracle p95")
        for x, y in zip(positions, selected, strict=True):
            axis.annotate(str(y), (x, y), xytext=(0, 7), textcoords="offset points", ha="center")
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Logical shards M (M>4 stays on four real hosts)")
        axis.set_ylabel("Visited shards / fan-out P")
        axis.set_xticks(positions, [str(value) for value in LOGICAL_COUNTS])
        axis.set_ylim(bottom=0)
        base.style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("K-Means fan-out at dataset-fixed finite ef", y=0.985)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.83))
    return save_figure(
        figure,
        "stage14_fig3_fixed_ef_fanout.pdf",
        purpose="在 ef 不再随点增大的条件下，测量 K-Means 为保证 Recall@10≥0.90 必须访问的逻辑分片数。",
        reading="橙线是所有查询统一采用、并经 9,000 条独立留出查询验证的服务 fan-out；灰线为全广播，绿线为逐查询 oracle 参考。M=8/16/32 是逻辑分片，不是真实机器数。",
        conclusion="SIFT 在 M=1/2/3/4/8/16/32 的 P 为 1/1/2/2/2/3/4；GloVe 为 1/2/3/3/5/7/8。GloVe 的 fan-out 并不极低。",
        boundary="fan-out 与 ef 证据来自 Stage 13 的独立调优/留出实验；Stage 14 复用其通过召回门槛的 P，仅重测物理 QPS。",
    )


def sensitivity_matrix(rows: Sequence[dict[str, Any]], dataset: str, field: str) -> np.ndarray:
    return base.sensitivity_matrix(rows, dataset, field)


def plot_sensitivity(rows: Sequence[dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.1))
    for column, dataset in enumerate(DATASETS):
        fanout = sensitivity_matrix(rows, dataset, "selected_minimum_fanout")
        recall = sensitivity_matrix(rows, dataset, "full_fanout_recall_at_10")
        cmap = plt.get_cmap("YlOrBr").copy()
        cmap.set_bad("#d1d5db")
        axes[0, column].imshow(np.ma.masked_invalid(fanout), aspect="auto", cmap=cmap, vmin=1)
        base.annotate_heatmap(axes[0, column], fanout, integer=True)
        axes[1, column].imshow(recall, aspect="auto", cmap="YlGn", vmin=0.80, vmax=1.0)
        base.annotate_heatmap(axes[1, column], recall, integer=False)
        fixed_column = EF_GRIDS[dataset].index(FIXED_EF[dataset])
        for axis in (axes[0, column], axes[1, column]):
            axis.add_patch(plt.Rectangle((fixed_column - 0.5, -0.5), 1, len(LOGICAL_COUNTS), fill=False, edgecolor="#dc2626", linewidth=2.0))
            axis.set_xticks(range(len(EF_GRIDS[dataset])), [str(value) for value in EF_GRIDS[dataset]])
            axis.set_yticks(range(len(LOGICAL_COUNTS)), [str(value) for value in LOGICAL_COUNTS])
            axis.set_xlabel("efSearch (red box = selected fixed ef)")
            axis.set_ylabel("Logical shards M")
        axes[0, column].set_title(f"{DATASET_LABELS[dataset]}: minimum fan-out (X=infeasible)")
        axes[1, column].set_title(f"{DATASET_LABELS[dataset]}: full-fan-out Recall@10")
    figure.suptitle("Finite ef sensitivity grid", y=1.01)
    figure.tight_layout()
    return save_figure(
        figure,
        "stage14_fig4_ef_sensitivity.pdf",
        purpose="展示有限合理 ef 网格中，ef 对最低 fan-out 和全广播召回的影响，验证为何不能无限增大 ef 来压低 fan-out。",
        reading="上排为达到召回门槛的最小 P，X 表示该 ef 下即使全广播也不合格；下排为全广播 Recall@10；红框是最终固定 ef。",
        conclusion="增大 ef 会提高召回并可能降低所需 P，因此 ef 与 fan-out 必须拆开控制。本轮固定 SIFT ef=24、GloVe ef=192，QPS 测量不再逐点增大 ef。",
        boundary="网格来自 1,000 条调优查询；正式 P 还需通过 9,000 条留出查询，例如 GloVe M=16 的 P=6 召回 0.89932 未通过，最终采用 P=7、召回 0.91148。",
    )


def plot_recall_stability(
    rows: Sequence[dict[str, Any]], baseline_b_rows: Sequence[dict[str, Any]]
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for row_index, dataset in enumerate(DATASETS):
        for method in METHODS:
            current = [row_for(rows, dataset, method, m) for m in PHYSICAL_COUNTS]
            axes[row_index, 0].plot(PHYSICAL_COUNTS, [row["recall_mean"] for row in current], color=COLORS[method], marker=MARKERS[method], linewidth=2, label=METHOD_LABELS[method])
            axes[row_index, 1].plot(PHYSICAL_COUNTS, [100 * row["qps_cv"] for row in current], color=COLORS[method], marker=MARKERS[method], linewidth=2, label=METHOD_LABELS[method])
        baseline_b = next(row for row in baseline_b_rows if row["dataset"] == dataset)
        axes[row_index, 0].scatter([1.08], [baseline_b["recall_mean"]], color="#111827", marker="x", s=60)
        axes[row_index, 1].scatter([1.08], [100 * baseline_b["qps_cv"]], color="#111827", marker="x", s=60)
        axes[row_index, 0].axhline(TARGET_RECALL, color="#dc2626", linestyle="--", label="Recall target")
        axes[row_index, 1].axhline(100 * CV_THRESHOLD, color="#dc2626", linestyle="--", label="5% CV trigger")
        axes[row_index, 0].set_title(f"{DATASET_LABELS[dataset]} Recall@10")
        axes[row_index, 1].set_title(f"{DATASET_LABELS[dataset]} QPS CV")
        axes[row_index, 0].set_ylabel("Recall@10")
        axes[row_index, 1].set_ylabel("CV (%)")
        for axis in axes[row_index]:
            axis.set_xlabel("Physical machines M")
            axis.set_xticks(PHYSICAL_COUNTS)
            base.style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Recall validity and QPS repetition stability", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    all_rows = [*rows, *baseline_b_rows]
    minimum_recall = min(row["recall_min"] for row in all_rows)
    maximum_cv = max(row["qps_cv"] for row in all_rows)
    return save_figure(
        figure,
        "stage14_fig5_recall_stability.pdf",
        purpose="确认所有 QPS 测点均满足 Recall@10 门槛，并检查重复测量波动是否得到足够次数处理。",
        reading="左列所有点必须高于红色 0.90 线；右列高于 5% 表示按协议从三次增加到五次重复，不表示自动丢弃该点。",
        conclusion=f"全部正式重复的最低 Recall@10 为 {minimum_recall:.5f}；最大 QPS CV 为 {100*maximum_cv:.2f}%，所有 CV>5% 的配置均已执行五次重复。",
        boundary="CV 只描述重复波动；系统性偏差由物理核隔离、A/B 基线、资源门槛和清理证据共同约束。",
    )


def plot_baseline_partition(
    rows: Sequence[dict[str, Any]],
    baseline_b_rows: Sequence[dict[str, Any]],
    partition_rows: Sequence[dict[str, Any]],
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.1, 4.9))
    x = np.arange(len(DATASETS))
    width = 0.34
    baseline_a = [row_for(rows, dataset, "random", 1) for dataset in DATASETS]
    baseline_b = [next(row for row in baseline_b_rows if row["dataset"] == dataset) for dataset in DATASETS]
    axes[0].bar(x - width / 2, [row["qps_mean"] for row in baseline_a], width, label="Baseline-A")
    axes[0].bar(x + width / 2, [row["qps_mean"] for row in baseline_b], width, label="Baseline-B")
    drifts: dict[str, float] = {}
    for index, (dataset, a, b) in enumerate(zip(DATASETS, baseline_a, baseline_b, strict=True)):
        drift = 100 * (b["qps_mean"] - a["qps_mean"]) / a["qps_mean"]
        drifts[dataset] = drift
        axes[0].annotate(f"{drift:+.2f}%", (index, max(a["qps_mean"], b["qps_mean"])), xytext=(0, 7), textcoords="offset points", ha="center")
    axes[0].set_xticks(x, [DATASET_LABELS[value] for value in DATASETS])
    axes[0].set_ylabel("Completed QPS")
    axes[0].set_title("Single-host baseline drift")
    axes[0].legend(frameon=False)
    base.style_axis(axes[0])
    positions = list(range(1, len(LOGICAL_COUNTS)))
    axes[1].plot(positions, [1 / value for value in LOGICAL_COUNTS[1:]], "--", color="#6b7280", label="Ideal equal max share=1/M")
    for dataset, color in zip(DATASETS, ("#2563eb", "#d97706"), strict=True):
        current = sorted((row for row in partition_rows if row["dataset"] == dataset), key=lambda row: row["logical_shard_count"])
        axes[1].plot(positions, [row["maximum_shard_fraction"] for row in current], marker="o", linewidth=2.1, color=color, label=DATASET_LABELS[dataset])
    axes[1].set_xticks(positions, [str(value) for value in LOGICAL_COUNTS[1:]])
    axes[1].set_xlabel("Logical shards M")
    axes[1].set_ylabel("Largest K-Means shard / all vectors")
    axes[1].set_title("K-Means partition imbalance")
    axes[1].legend(frameon=False)
    base.style_axis(axes[1])
    figure.suptitle("Stage 14 run stability and reused partition balance", y=1.02)
    figure.tight_layout()
    m3 = {row["dataset"]: row for row in partition_rows if row["logical_shard_count"] == 3}
    return save_figure(
        figure,
        "stage14_fig6_baseline_partition.pdf",
        purpose="左侧检查长时间重测前后的单机漂移，右侧检查 K-Means 分区是否把向量与工作量均匀分配。",
        reading="左侧 A/B 差异越小，时间漂移越难解释主曲线；右侧越接近灰色 1/M 线越均衡，明显高于灰线表示最大分片过重。",
        conclusion=(
            f"SIFT/GloVe baseline 漂移分别为 {drifts['sift1m']:+.2f}%/{drifts['glove-200-angular']:+.2f}%。"
            f"M=3 最大分片分别占 {100*m3['sift1m']['maximum_shard_fraction']:.1f}%/{100*m3['glove-200-angular']['maximum_shard_fraction']:.1f}%，且均在 12 次迭代内未收敛。"
        ),
        boundary="分区证据来自 Stage 13 的同一确定性分区产物，Stage 14 QPS 直接复用这些产物；偏斜是解释线索，不单独构成吞吐因果证明。",
    )


def plot_latency_resources(rows: Sequence[dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.8))
    for row_index, dataset in enumerate(DATASETS):
        for method in METHODS:
            current = [row_for(rows, dataset, method, m) for m in PHYSICAL_COUNTS]
            x = PHYSICAL_COUNTS
            axes[row_index, 0].plot(x, [row["p95_latency_us"] / 1000 for row in current], color=COLORS[method], marker=MARKERS[method], linewidth=2, label=METHOD_LABELS[method])
            axes[row_index, 1].plot(x, [row["max_worker_cpu_utilization_pct"] for row in current], color=COLORS[method], marker=MARKERS[method], linewidth=2, label=METHOD_LABELS[method])
            axes[row_index, 2].plot(x, [row["max_network_utilization_pct"] for row in current], color=COLORS[method], marker=MARKERS[method], linewidth=2, label=METHOD_LABELS[method])
        for column, title in enumerate(("p95 latency (ms)", "Max worker CPU (%)", "Max 25GbE utilization (%)")):
            axes[row_index, column].set_title(f"{DATASET_LABELS[dataset]}: {title}")
            axes[row_index, column].set_xlabel("Physical machines M")
            axes[row_index, column].set_xticks(PHYSICAL_COUNTS)
            base.style_axis(axes[row_index, column])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Latency and resource diagnostics for Stage 14", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    maximum_network = max(row["max_network_utilization_pct"] for row in rows)
    maximum_worker = max(row["max_worker_cpu_utilization_pct"] for row in rows)
    return save_figure(
        figure,
        "stage14_fig7_latency_resources.pdf",
        purpose="展示各正式 QPS 点的 p95 延迟、最忙 worker CPU 和网络利用率，用于定位偏离线性扩展的资源线索。",
        reading="左列看 scatter-gather 延迟，中列看最忙 worker 是否接近其保留核心容量，右列看 25GbE 是否接近饱和；必须与 QPS、召回和分区偏斜联合解释。",
        conclusion=f"观测到的最大 worker CPU 为 {maximum_worker:.1f}%，最大 25GbE 利用率仅 {maximum_network:.2f}%；网络带宽不是主瓶颈，CPU/分片不均衡与多分片聚合开销更值得关注。",
        boundary="worker CPU 高利用率本身不违反 E1 门槛；所有正式重复仍要求客户端、协调器、队列、路由占比和网络门槛无违规标志。",
    )


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def report_markdown(
    rows: Sequence[dict[str, Any]],
    baseline_b_rows: Sequence[dict[str, Any]],
    fanout_rows: Sequence[dict[str, Any]],
    partition_rows: Sequence[dict[str, Any]],
    cleanup: dict[str, Any],
    affinity: dict[str, Any],
    figures: Sequence[Path],
) -> str:
    lines = [
        "# Stage 14：1～4 台真实机器物理核隔离重测",
        "",
        "状态：`MEASUREMENTS_COMPLETE`。QPS 采用 Stage 14；SIFT1M 固定 `efSearch=24`，GloVe-200-angular 固定 `efSearch=192`。",
        "",
        "worker 使用物理核 0～7 的双 SMT 线程（`0-7,16-23`）；客户端使用物理核 8～15（`8-15,24-31`），物理核交集为空。M=1～4 均是对应数量的真实机器和逻辑分片。",
        "",
        "## 真实物理机器 QPS",
        "",
        "| 数据集 | 方法 | 固定 ef | M=1 | M=2 | M=3 | M=4 | M=4/M=1 | 四机效率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in METHODS:
            current = [row_for(rows, dataset, method, m) for m in PHYSICAL_COUNTS]
            lines.append(
                "| "
                + " | ".join(
                    [
                        DATASET_LABELS[dataset],
                        METHOD_LABELS[method],
                        str(FIXED_EF[dataset]),
                        *[fmt(row["qps_mean"]) for row in current],
                        f"{current[-1]['normalized_qps_to_m1']:.3f}×",
                        f"{current[-1]['scaling_efficiency']:.3f}",
                    ]
                )
                + " |"
            )
    lines.extend(["", "尾部单机基线漂移：", "", "| 数据集 | baseline-A | baseline-B | 漂移 |", "|---|---:|---:|---:|"])
    for dataset in DATASETS:
        a = row_for(rows, dataset, "random", 1)
        b = next(row for row in baseline_b_rows if row["dataset"] == dataset)
        drift = 100 * (b["qps_mean"] - a["qps_mean"]) / a["qps_mean"]
        lines.append(f"| {DATASET_LABELS[dataset]} | {fmt(a['qps_mean'])} | {fmt(b['qps_mean'])} | {drift:+.2f}% |")
    lines.extend(
        [
            "",
            "## 固定 ef 的 K-Means fan-out",
            "",
            "M=8、16、32 仅表示四台物理机上的逻辑分片数。P 均通过 9,000 条独立留出查询的 Recall@10≥0.90 门槛。",
            "",
            "| 数据集 | 固定 ef | M=1 | M=2 | M=3 | M=4 | M=8 | M=16 | M=32 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        current = sorted((row for row in fanout_rows if row["dataset"] == dataset), key=lambda row: row["logical_shard_count"])
        lines.append("| " + " | ".join([DATASET_LABELS[dataset], str(FIXED_EF[dataset]), *[str(row["holdout_selected_fanout"]) for row in current]]) + " |")
    lines.extend(
        [
            "",
            "GloVe M=16：P=6 在留出集 Recall=0.89932，未通过；P=7 时 Recall=0.91148，最终采用 P=7。",
            "",
            "## 结论",
            "",
            "1. Stage 13 的 worker `0-19` 与客户端 `20-31` 实际共享物理核 4～15 的 SMT 线程，因此旧 QPS 曲线不再作为正式结论。Stage 14 消除了该混杂因素。",
            "2. 物理核隔离后，1～4 台真实机器仍未表现线性扩展；Random 在 SIFT 上随机器数增加明显退化，K-Means 在 SIFT 上能超过单机但在 M=3/4 不再继续增长。",
            "3. K-Means fan-out 并非普遍极低。GloVe 的固定 ef 序列是 1/2/3/3/5/7/8；SIFT 的 1/1/2/2/2/3/4 则是在 ef=24 和独立召回门槛下仍成立的实测结果。",
            "4. M=3 的 K-Means 分区明显偏斜且未在 12 次迭代内收敛；物理机器数不能直接等同于均匀增加的有效计算能力。",
            "",
            "## PDF 图表",
            "",
            "每张 PDF 页面下方都内嵌中文“用途、如何理解、最终结论、证据边界”。",
            "",
        ]
    )
    lines.extend(f"- [{figure.name}](figures/{figure.name})" for figure in figures)
    nonconverged = [row for row in partition_rows if row["logical_shard_count"] == 3 and not row["kmeans_converged"]]
    lines.extend(
        [
            "",
            "## 完整性",
            "",
            f"- Stage 14 collection 清理证明：{cleanup['proof_count']}/16 通过，四个 peer 均为 HTTP 404。",
            f"- 绑核应用、无物理核重叠和实验后恢复：`{affinity['status']}`；Qdrant 已恢复到 `0-19`。",
            "- 所有 E1 文件均为 `VALID_E1`，每次正式重复 Recall@10≥0.90 且 `bottleneck_flags=[]`；CV>5% 时均有五次重复。",
            f"- M=3 未收敛分区：{len(nonconverged)}/2；异常被保留并写入图表与 CSV。",
            "- 原始逐查询 CSV、重复摘要、调优与留出证据均保留；未覆盖 Stage 13 或更早结果。",
            "",
        ]
    )
    return "\n".join(lines)


def summarize() -> dict[str, Any]:
    execution_path = STAGE14_DIR / "execution-complete.json"
    execution = base.load_json(execution_path)
    if execution.get("status") != "MEASUREMENTS_COMPLETE":
        raise ValueError("Stage 14 execution is not complete")
    rows, baseline_b_rows = load_physical_rows()
    fanout_rows = base.load_fanout_rows()
    sensitivity_rows = base.load_sensitivity_rows()
    partition_rows = base.load_partition_rows()
    cleanup = audit_cleanup()
    affinity = audit_affinity()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = [
        base.write_csv(SUMMARY_DIR / "physical-qps.csv", base.physical_csv_rows(rows, baseline_b_rows, [])),
        base.write_csv(SUMMARY_DIR / "fanout-fixed-ef.csv", fanout_rows),
        base.write_csv(SUMMARY_DIR / "ef-sensitivity.csv", sensitivity_rows),
        base.write_csv(SUMMARY_DIR / "partition-balance.csv", partition_rows),
    ]
    drift_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        a = row_for(rows, dataset, "random", 1)
        b = next(row for row in baseline_b_rows if row["dataset"] == dataset)
        drift_rows.append(
            {
                "dataset": dataset,
                "fixed_ef": FIXED_EF[dataset],
                "baseline_a_qps": a["qps_mean"],
                "baseline_b_qps": b["qps_mean"],
                "drift_fraction": (b["qps_mean"] - a["qps_mean"]) / a["qps_mean"],
                "baseline_a_source": a["source"],
                "baseline_b_source": b["source"],
            }
        )
    csv_paths.append(base.write_csv(SUMMARY_DIR / "baseline-drift.csv", drift_rows))
    integrity_path = base.write_json(
        SUMMARY_DIR / "integrity-audit.json",
        {"status": "PASS", "cleanup": cleanup, "affinity": affinity},
    )
    figures = [
        plot_physical_qps(rows, baseline_b_rows),
        plot_normalized_scaling(rows),
        plot_fanout(fanout_rows),
        plot_sensitivity(sensitivity_rows),
        plot_recall_stability(rows, baseline_b_rows),
        plot_baseline_partition(rows, baseline_b_rows, partition_rows),
        plot_latency_resources(rows),
    ]
    base.verify_pdfs(figures)
    report_path = base.atomic_write_text(
        SUMMARY_DIR / "RESULTS_zh.md",
        report_markdown(rows, baseline_b_rows, fanout_rows, partition_rows, cleanup, affinity, figures),
    )
    source_paths = sorted({Path(row["source"]) for row in [*rows, *baseline_b_rows]})
    manifest = {
        "record_type": "stage14_physical_core_isolated_summary",
        "status": "PASS",
        "execution_complete": str(execution_path.resolve()),
        "execution_complete_sha256": base.sha256(execution_path),
        "physical_e1_sources": [{"path": str(path), "sha256": base.sha256(path)} for path in source_paths],
        "csv_files": [{"path": str(path.resolve()), "sha256": base.sha256(path)} for path in csv_paths],
        "figures": [{"path": str(path.resolve()), "sha256": base.sha256(path), "pages": 1} for path in figures],
        "report": str(report_path.resolve()),
        "report_sha256": base.sha256(report_path),
        "integrity_audit": str(integrity_path.resolve()),
        "integrity_audit_sha256": base.sha256(integrity_path),
        "physical_configuration_count": len(rows),
        "distinct_e1_file_count": len(source_paths),
        "baseline_b_count": len(baseline_b_rows),
        "fanout_configuration_count": len(fanout_rows),
        "sensitivity_point_count": len(sensitivity_rows),
        "cleanup_proof_count": cleanup["proof_count"],
    }
    manifest_path = base.write_json(SUMMARY_DIR / "manifest.json", manifest)
    manifest["manifest"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = base.sha256(manifest_path)
    return manifest


def check() -> dict[str, Any]:
    manifest_path = SUMMARY_DIR / "manifest.json"
    manifest = base.load_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise ValueError("summary manifest is not PASS")
    for group in ("physical_e1_sources", "csv_files", "figures"):
        for entry in manifest[group]:
            path = Path(entry["path"])
            if not path.exists() or base.sha256(path) != entry["sha256"]:
                raise ValueError(f"manifest hash mismatch: {path}")
    for path_key, hash_key in (
        ("execution_complete", "execution_complete_sha256"),
        ("report", "report_sha256"),
        ("integrity_audit", "integrity_audit_sha256"),
    ):
        path = Path(manifest[path_key])
        if not path.exists() or base.sha256(path) != manifest[hash_key]:
            raise ValueError(f"manifest hash mismatch: {path}")
    figures = [Path(entry["path"]) for entry in manifest["figures"]]
    base.verify_pdfs(figures)
    audit_cleanup()
    audit_affinity()
    load_physical_rows()
    return {
        "status": "PASS",
        "manifest": str(manifest_path.resolve()),
        "figure_count": len(figures),
        "csv_count": len(manifest["csv_files"]),
        "distinct_e1_file_count": len(manifest["physical_e1_sources"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = check() if args.check else summarize()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
