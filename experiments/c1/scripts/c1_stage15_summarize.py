#!/usr/bin/env python3
"""Summarize the Stage 15 one-HNSW-graph-per-shard physical retest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt

import c1_stage13_summarize as base


SCRIPT_DIR = Path(__file__).resolve().parent
C1_DIR = SCRIPT_DIR.parent
STAGE15_DIR = C1_DIR / "retests" / "stage15-single-hnsw-per-shard"
STAGE14_DIR = C1_DIR / "retests" / "stage14-physical-core-isolated"
SUMMARY_DIR = STAGE15_DIR / "summary"
FIGURES_DIR = SUMMARY_DIR / "figures"
STAGE = "stage15-single-hnsw-per-shard"

DATASETS = ("sift1m", "glove-200-angular")
DATASET_LABELS = {"sift1m": "SIFT1M", "glove-200-angular": "GloVe-200"}
DATASET_SIZES = {"sift1m": 1_000_000, "glove-200-angular": 1_183_514}
METHODS = ("random", "kmeans")
METHOD_LABELS = {"random": "Random broadcast", "kmeans": "K-Means routing"}
COLORS = {"random": "#2563eb", "kmeans": "#d97706"}
MARKERS = {"random": "o", "kmeans": "s"}
PHYSICAL_COUNTS = (1, 2, 3, 4)
FIXED_EF = {"sift1m": 24, "glove-200-angular": 320}
TARGET_RECALL = 0.90

base.FOOTER_TITLE = "Stage 15 中文图解"
base.PDF_CREATOR = "Orion C1 Stage 15 summarizer"


def dataset_token(dataset: str) -> str:
    return "sift1m" if dataset == "sift1m" else "glove"


def e1_path(
    stage_dir: Path,
    stage_number: int,
    dataset: str,
    method: str,
    machines: int,
    baseline: str = "a",
) -> Path:
    if machines == 1:
        return stage_dir / "e1" / (
            f"stage{stage_number}-e1-{dataset}-random-m1-baseline-{baseline}.json"
        )
    return stage_dir / "e1" / (
        f"stage{stage_number}-e1-{dataset}-{method}-m{machines}-primary.json"
    )


def e1_summary_row(
    path: Path,
    *,
    display_method: str,
    stage: str,
    baseline_label: str = "",
) -> dict[str, Any]:
    row = base.e1_row(
        path,
        display_method=display_method,
        stage=stage,
        baseline_label=baseline_label,
    )
    repetitions = base.load_json(path)["repetitions"]
    row["aggregator_cpu_utilization_pct"] = float(
        np.mean(
            [
                repetition["aggregator_cpu_utilization_pct_of_reserved_cores"]
                for repetition in repetitions
            ]
        )
    )
    row["aggregator_max_single_process_cpu_utilization_pct"] = float(
        np.mean(
            [
                repetition["aggregator_max_single_process_cpu_utilization_pct"]
                for repetition in repetitions
            ]
        )
    )
    return row


def load_rows(
    stage_dir: Path, stage_number: int, stage_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    baseline_b: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method in METHODS:
            rows.append(
                e1_summary_row(
                    e1_path(stage_dir, stage_number, dataset, method, 1),
                    display_method=method,
                    stage=stage_name,
                    baseline_label="A",
                )
            )
            for machines in PHYSICAL_COUNTS[1:]:
                rows.append(
                    e1_summary_row(
                        e1_path(stage_dir, stage_number, dataset, method, machines),
                        display_method=method,
                        stage=stage_name,
                    )
                )
        baseline_b.append(
            e1_summary_row(
                e1_path(stage_dir, stage_number, dataset, "random", 1, "b"),
                display_method="random",
                stage=stage_name,
                baseline_label="B",
            )
        )
    base.add_normalization(rows)
    return rows, baseline_b


def row_for(
    rows: Iterable[dict[str, Any]], dataset: str, method: str, machines: int
) -> dict[str, Any]:
    return base.row_for(rows, dataset, method, machines)


def prepare_path(
    stage_dir: Path,
    stage_number: int,
    dataset: str,
    method: str,
    machines: int,
    baseline: str = "a",
) -> Path:
    token = dataset_token(dataset)
    if machines == 1:
        name = f"c1s{stage_number}_{token}_common_m1_baseline-{baseline}-prepare.json"
    else:
        name = f"c1s{stage_number}_{token}_{method}_m{machines}-prepare.json"
    return stage_dir / "logs" / name


def layout_row(
    stage_dir: Path,
    stage_number: int,
    dataset: str,
    method: str,
    machines: int,
    baseline: str = "a",
) -> dict[str, Any]:
    path = prepare_path(
        stage_dir, stage_number, dataset, method, machines, baseline
    )
    payload = base.load_json(path)
    total = int(payload["collection_info"]["segments_count"])
    estimated_nonempty = total - machines
    row = {
        "stage": f"stage{stage_number}",
        "dataset": dataset,
        "method": method,
        "physical_machine_count": machines,
        "logical_shard_count": machines,
        "baseline_label": baseline.upper() if machines == 1 else "",
        "total_segments": total,
        "estimated_empty_appendable_segments": machines,
        "estimated_nonempty_hnsw_segments": estimated_nonempty,
        "estimated_hnsw_segments_per_shard": estimated_nonempty / machines,
        "max_segment_size_kb": payload.get("max_segment_size_kb"),
        "source": str(path.resolve()),
        "source_sha256": base.sha256(path),
    }
    if stage_number == 15:
        gate = payload.get("single_indexed_segment_per_shard_gate") or {}
        row["stage15_gate_status"] = gate.get("status")
        if (
            gate.get("status") != "PASS"
            or int(gate.get("expected_total_segments") or 0) != 2 * machines
            or int(gate.get("observed_total_segments") or 0) != 2 * machines
            or total != 2 * machines
        ):
            raise ValueError(f"Stage 15 single-graph gate failed: {path}")
    else:
        row["stage15_gate_status"] = "NOT_APPLICABLE"
    return row


def load_layout_rows(stage_dir: Path, stage_number: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        rows.append(
            layout_row(stage_dir, stage_number, dataset, "common", 1, "a")
        )
        rows.append(
            layout_row(stage_dir, stage_number, dataset, "common", 1, "b")
        )
        for method in METHODS:
            for machines in PHYSICAL_COUNTS[1:]:
                rows.append(
                    layout_row(
                        stage_dir, stage_number, dataset, method, machines
                    )
                )
    return rows


def layout_for(
    rows: Sequence[dict[str, Any]],
    dataset: str,
    method: str,
    machines: int,
    baseline: str = "A",
) -> dict[str, Any]:
    expected_method = "common" if machines == 1 else method
    matches = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["method"] == expected_method
        and row["physical_machine_count"] == machines
        and (machines != 1 or row["baseline_label"] == baseline)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one layout row for {dataset}/{method}/M={machines}: {matches}"
        )
    return matches[0]


def load_fanout_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for machines in PHYSICAL_COUNTS:
            e1 = row_for(rows, dataset, "kmeans", machines)
            if machines == 1:
                source = Path(e1["source"])
                heldout_recall = e1["recall_mean"]
                selection_kind = "single-shard baseline"
            elif dataset == "sift1m":
                source = (
                    C1_DIR
                    / "retests"
                    / "stage13-physical1to4-ef-controlled"
                    / "fanout"
                    / "holdout"
                    / f"{dataset}-kmeans-m{machines}-ef24-selected.json"
                )
                selected = base.load_json(source)
                heldout_recall = float(selected["achieved_recall"])
                selection_kind = "Stage 13 held-out selection reused"
            else:
                source = (
                    STAGE15_DIR
                    / "fanout"
                    / "holdout"
                    / f"{dataset}-kmeans-m{machines}-ef320-selected.json"
                )
                selected = base.load_json(source)
                heldout_recall = float(selected["achieved_recall"])
                selection_kind = "Stage 15 single-graph held-out selection"
            if e1["fanout"] > machines or heldout_recall < TARGET_RECALL:
                raise ValueError(f"invalid fan-out row: {dataset}/M={machines}")
            output.append(
                {
                    "dataset": dataset,
                    "logical_shard_count": machines,
                    "physical_machine_count": machines,
                    "fixed_ef": FIXED_EF[dataset],
                    "selected_fanout": int(e1["fanout"]),
                    "fanout_fraction": int(e1["fanout"]) / machines,
                    "heldout_recall": heldout_recall,
                    "selection_kind": selection_kind,
                    "source": str(source.resolve()),
                    "source_sha256": base.sha256(source),
                }
            )
    return output


def load_ef_rows() -> list[dict[str, Any]]:
    directory = STAGE15_DIR / "ef-calibration" / "glove-m1-single-graph"
    rows: list[dict[str, Any]] = []
    for ef in (192, 224, 256, 320, 384):
        path = directory / f"glove-m1-ef{ef}-tuning.json"
        payload = base.load_json(path)
        rows.append(
            {
                "dataset": "glove-200-angular",
                "split": "tuning",
                "ef_search": ef,
                "recall_at_10": float(payload["achieved_recall"]),
                "selected": ef == FIXED_EF["glove-200-angular"],
                "source": str(path.resolve()),
                "source_sha256": base.sha256(path),
            }
        )
    holdout_path = directory / "glove-m1-ef320-holdout.json"
    holdout = base.load_json(holdout_path)
    rows.append(
        {
            "dataset": "glove-200-angular",
            "split": "holdout",
            "ef_search": 320,
            "recall_at_10": float(holdout["achieved_recall"]),
            "selected": True,
            "source": str(holdout_path.resolve()),
            "source_sha256": base.sha256(holdout_path),
        }
    )
    return rows


def audit_cleanup() -> dict[str, Any]:
    paths = sorted((STAGE15_DIR / "cleanup").glob("*.json"))
    if len(paths) != 16:
        raise ValueError(f"expected 16 Stage 15 cleanup proofs, found {len(paths)}")
    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in paths:
        payload = base.load_json(path)
        codes = {
            host: int(details["http_status"])
            for host, details in payload["peers"].items()
        }
        controller = Path(payload["controller_storage_path"])
        valid = (
            payload.get("status") == "VERIFIED_DELETED"
            and payload.get("controller_storage_absent") is True
            and not controller.exists()
            and len(codes) == 4
            and set(codes.values()) == {404}
        )
        if not valid:
            failures.append(path.name)
        entries.append(
            {
                "collection": payload["collection"],
                "proof": str(path.resolve()),
                "proof_sha256": base.sha256(path),
                "valid": valid,
            }
        )
    if failures:
        raise ValueError(f"cleanup audit failures: {failures}")
    return {"status": "PASS", "proof_count": len(entries), "entries": entries}


def audit_affinity() -> dict[str, Any]:
    applied_path = STAGE15_DIR / "worker-affinity-applied.json"
    restored_path = STAGE15_DIR / "worker-affinity-restored.json"
    preflight_path = STAGE15_DIR / "preflight.json"
    applied = base.load_json(applied_path)
    restored = base.load_json(restored_path)
    preflight = base.load_json(preflight_path)
    valid = (
        applied.get("status") == "PASS"
        and restored.get("status") == "PASS"
        and preflight.get("status") == "PASS"
        and preflight.get("physical_core_overlap") == []
        and all(
            node["after"]["cpuset"] == "0-7,16-23"
            and node["runtime_qdrant_affinity"] == "0-7,16-23"
            for node in applied["nodes"]
        )
        and all(
            node.get("valid") is True
            and node["restored"]["cpuset"] == "0-19"
            and node["runtime_qdrant_affinity"] == "0-19"
            for node in restored["nodes"]
        )
    )
    if not valid:
        raise ValueError("Stage 15 affinity audit failed")
    return {
        "status": "PASS",
        "worker_test_cpuset": "0-7,16-23",
        "benchmark_test_cpuset": "8-15,24-31",
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


def style_all(axes: Iterable[Any]) -> None:
    for axis in axes:
        base.style_axis(axis)
        axis.set_xticks(PHYSICAL_COUNTS)


def plot_qps(
    rows: Sequence[dict[str, Any]], baseline_b: Sequence[dict[str, Any]]
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for method in METHODS:
            current = [row_for(rows, dataset, method, m) for m in PHYSICAL_COUNTS]
            mean = [row["qps_mean"] for row in current]
            low = [row["qps_mean"] - row["qps_ci95_lower"] for row in current]
            high = [row["qps_ci95_upper"] - row["qps_mean"] for row in current]
            axis.errorbar(
                PHYSICAL_COUNTS,
                mean,
                yerr=[low, high],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.2,
                capsize=4,
                label=METHOD_LABELS[method],
            )
        b = next(row for row in baseline_b if row["dataset"] == dataset)
        axis.scatter([1.08], [b["qps_mean"]], color="#111827", marker="x", s=60, label="Baseline-B")
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Physical machines M")
        axis.set_ylabel("Completed QPS")
    style_all(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Stage 15 end-to-end QPS with one HNSW graph per shard", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    values = {
        (d, m): row_for(rows, d, m, 4)["qps_mean"]
        for d in DATASETS
        for m in METHODS
    }
    return save_figure(
        figure,
        "stage15_fig1_physical_qps.pdf",
        purpose="在每个逻辑 shard 只有一张非空 HNSW 图、worker 与客户端物理核隔离的条件下，重测 1～4 台真实机器的端到端吞吐。",
        reading="横轴 M 同时是真实主机数和逻辑 shard 数；点为重复均值，误差条为 95% 置信区间，M=1 旁的叉号是全部测点后的 baseline-B。",
        conclusion=(
            f"M=4 时 SIFT Random/K-Means 为 {values[('sift1m','random')]:,.0f}/{values[('sift1m','kmeans')]:,.0f} QPS；"
            f"GloVe 为 {values[('glove-200-angular','random')]:,.0f}/{values[('glove-200-angular','kmeans')]:,.0f} QPS。单图控制没有使端到端 QPS 自动按机器数增长。"
        ),
        boundary="这是满足 Recall@10≥0.90 的服务级结果；它包含 fan-out、scatter/gather、合并、负载偏斜和固定 ef 成本，不能只用单张 HNSW 图的局部复杂度解释。",
    )


def plot_scaling(rows: Sequence[dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    for axis, dataset in zip(axes, DATASETS, strict=True):
        n = DATASET_SIZES[dataset]
        local_log = [math.log(n) / math.log(n / m) for m in PHYSICAL_COUNTS]
        axis.plot(PHYSICAL_COUNTS, PHYSICAL_COUNTS, "--", color="#6b7280", label="Ideal linear")
        axis.plot(PHYSICAL_COUNTS, local_log, ":", color="#059669", linewidth=2.2, label="Local log-work only")
        for method in METHODS:
            current = [row_for(rows, dataset, method, m) for m in PHYSICAL_COUNTS]
            axis.plot(
                PHYSICAL_COUNTS,
                [row["normalized_qps_to_m1"] for row in current],
                color=COLORS[method], marker=MARKERS[method], linewidth=2.2,
                label=METHOD_LABELS[method],
            )
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Physical machines M")
        axis.set_ylabel("QPS / M=1 QPS")
        axis.set_ylim(bottom=0)
    style_all(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Observed scaling versus the local HNSW logarithmic-work envelope", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.83))
    ratios = {
        (d, m): row_for(rows, d, m, 4)["normalized_qps_to_m1"]
        for d in DATASETS
        for m in METHODS
    }
    return save_figure(
        figure,
        "stage15_fig2_scaling_model.pdf",
        purpose="区分端到端 QPS 扩展、理想线性扩展，以及仅由局部图规模从 N 降到 N/M 所能提供的对数工作量收益。",
        reading="灰虚线是假设每台机器都贡献独立吞吐的 M 倍；绿点线是 log(N)/log(N/M)，M=4 仅约 1.11 倍；彩线为实测端到端吞吐倍数。",
        conclusion=(
            f"M=4 相对单机：SIFT Random/K-Means 为 {ratios[('sift1m','random')]:.2f}×/{ratios[('sift1m','kmeans')]:.2f}×，"
            f"GloVe 为 {ratios[('glove-200-angular','random')]:.2f}×/{ratios[('glove-200-angular','kmeans')]:.2f}×。局部对数项本来就不是 M 倍吞吐定律。"
        ),
        boundary="绿线只是复杂度直觉，不是 HNSW 的严格性能模型；固定 ef、缓存、图质量和硬件并行会改变局部常数。",
    )


def plot_fanout(fanout_rows: Sequence[dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    for axis, dataset in zip(axes, DATASETS, strict=True):
        current = [row for row in fanout_rows if row["dataset"] == dataset]
        p = [row["selected_fanout"] for row in current]
        axis.plot(PHYSICAL_COUNTS, PHYSICAL_COUNTS, "--", color="#6b7280", label="Broadcast P=M")
        axis.plot(
            PHYSICAL_COUNTS,
            p,
            "s-",
            color="#d97706",
            linewidth=2.4,
            label="Selected P (dataset-fixed ef)",
        )
        for x, row in zip(PHYSICAL_COUNTS, current, strict=True):
            axis.annotate(
                f"P={row['selected_fanout']}\nR={row['heldout_recall']:.3f}",
                (x, row["selected_fanout"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8,
            )
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Physical/logical shards M")
        axis.set_ylabel("Visited shards P")
        axis.set_ylim(0, 4.7)
    style_all(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Held-out K-Means fan-out at dataset-fixed bounded ef", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    sequences = {
        dataset: "/".join(str(row["selected_fanout"]) for row in fanout_rows if row["dataset"] == dataset)
        for dataset in DATASETS
    }
    return save_figure(
        figure,
        "stage15_fig3_fixed_ef_fanout.pdf",
        purpose="在 ef 对每个数据集固定且有硬上限时，测量 K-Means 为达到 Recall@10≥0.90 实际必须访问多少 shard。",
        reading="橙线是统一服务 fan-out P，标注 R 是独立留出集 Recall；灰线是全广播。P 接近 M 表示路由几乎没有减少分片访问。",
        conclusion=f"SIFT M=1/2/3/4 的 P 为 {sequences['sift1m']}；GloVe 为 {sequences['glove-200-angular']}。GloVe 的 fan-out 并不低，不能靠继续增大 ef 人为压低。",
        boundary="SIFT 复用相同分区和固定 ef=24 的既有独立留出选择；GloVe 在单图集合上以固定 ef=320 重新调优并做 9,000-query 留出验证。",
    )


def plot_work(rows: Sequence[dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    for column, dataset in enumerate(DATASETS):
        for method in METHODS:
            current = [row_for(rows, dataset, method, m) for m in PHYSICAL_COUNTS]
            total = [row["mean_distance_computations_per_query"] for row in current]
            local = [value / row["fanout"] for value, row in zip(total, current, strict=True)]
            axes[0, column].plot(PHYSICAL_COUNTS, total, color=COLORS[method], marker=MARKERS[method], linewidth=2.1, label=METHOD_LABELS[method])
            axes[1, column].plot(PHYSICAL_COUNTS, local, color=COLORS[method], marker=MARKERS[method], linewidth=2.1, label=METHOD_LABELS[method])
        axes[0, column].set_title(f"{DATASET_LABELS[dataset]} total distance work/query")
        axes[1, column].set_title(f"{DATASET_LABELS[dataset]} distance work/visited shard")
        axes[0, column].set_ylabel("Distance computations")
        axes[1, column].set_ylabel("Distance computations / P")
        for axis in axes[:, column]:
            axis.set_xlabel("Physical/logical shards M")
    style_all(axes.flat)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Local HNSW work reduction versus aggregate fan-out work", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    sift_r1 = row_for(rows, "sift1m", "random", 1)
    sift_r4 = row_for(rows, "sift1m", "random", 4)
    local_change = (
        (sift_r4["mean_distance_computations_per_query"] / sift_r4["fanout"])
        / sift_r1["mean_distance_computations_per_query"]
    )
    total_change = sift_r4["mean_distance_computations_per_query"] / sift_r1["mean_distance_computations_per_query"]
    return save_figure(
        figure,
        "stage15_fig4_local_vs_aggregate_work.pdf",
        purpose="直接验证分图后单个被访问 shard 的图搜索工作是否下降，以及该下降是否被访问更多 shard 的总工作抵消。",
        reading="下排把每查询距离计算除以实际 P，接近局部单图工作；上排是服务真正承担的每查询总工作。局部下降但总量上升时，QPS 不应上升。",
        conclusion=f"SIFT Random 从 M=1 到 M=4 的每 shard 距离工作变为 {local_change:.2f}×，但每查询总距离工作变为 {total_change:.2f}×；局部收益被 P=4 的广播放大项抵消。",
        boundary="距离计算是硬件计数器推导的工作代理；它不含网络等待和结果合并，但比仅用 wall-clock QPS 更能定位局部图与总服务成本的差异。",
    )


def plot_stage14_comparison(
    rows: Sequence[dict[str, Any]],
    old_rows: Sequence[dict[str, Any]],
    layout15: Sequence[dict[str, Any]],
    layout14: Sequence[dict[str, Any]],
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    for column, dataset in enumerate(DATASETS):
        for method in METHODS:
            ratio = [
                row_for(rows, dataset, method, m)["qps_mean"]
                / row_for(old_rows, dataset, method, m)["qps_mean"]
                for m in PHYSICAL_COUNTS
            ]
            axes[0, column].plot(PHYSICAL_COUNTS, ratio, color=COLORS[method], marker=MARKERS[method], linewidth=2.1, label=METHOD_LABELS[method])
            old_graphs = [
                layout_for(layout14, dataset, method, m)["estimated_nonempty_hnsw_segments"]
                for m in PHYSICAL_COUNTS
            ]
            new_graphs = [
                layout_for(layout15, dataset, method, m)["estimated_nonempty_hnsw_segments"]
                for m in PHYSICAL_COUNTS
            ]
            axes[1, column].plot(PHYSICAL_COUNTS, old_graphs, linestyle="--", color=COLORS[method], marker=MARKERS[method], label=f"Stage14 {METHOD_LABELS[method]}")
            axes[1, column].plot(PHYSICAL_COUNTS, new_graphs, linestyle="-", color=COLORS[method], marker=MARKERS[method], alpha=0.65, label=f"Stage15 {METHOD_LABELS[method]}")
        axes[0, column].axhline(1.0, color="#6b7280", linestyle=":")
        axes[0, column].set_title(f"{DATASET_LABELS[dataset]} Stage15 / Stage14 QPS")
        axes[0, column].set_ylabel("QPS ratio")
        axes[1, column].set_title(f"{DATASET_LABELS[dataset]} non-empty HNSW segments")
        axes[1, column].set_ylabel("Collection-wide graph count")
        for axis in axes[:, column]:
            axis.set_xlabel("Physical/logical shards M")
    style_all(axes.flat)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.suptitle("Why the Stage 14 curve was not a clean graph-partitioning test", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    sift_ratio = row_for(rows, "sift1m", "random", 1)["qps_mean"] / row_for(old_rows, "sift1m", "random", 1)["qps_mean"]
    glove_ratio = row_for(rows, "glove-200-angular", "random", 1)["qps_mean"] / row_for(old_rows, "glove-200-angular", "random", 1)["qps_mean"]
    return save_figure(
        figure,
        "stage15_fig5_stage14_comparison.pdf",
        purpose="对比生产默认多 segment 的 Stage 14 与单 shard 单 HNSW 图的 Stage 15，定位旧曲线中的图数量混杂。",
        reading="上排大于 1 表示单图控制更快；下排虚线是 Stage 14 估计非空图数，实线是 Stage 15 强制的一 shard 一图。M=1 的差异最能说明旧基线被多图搜索压低。",
        conclusion=f"单机基线在单图控制后，SIFT 提升 {sift_ratio:.2f}×，GloVe 提升 {glove_ratio:.2f}×；旧曲线的表观扩展包含基线图数不一致，不能当作纯 HNSW 分图证据。",
        boundary="GloVe Stage 15 同时因单图召回重新固定 ef=320，而 Stage 14 为 ef=192，因此其 QPS 比不能只归因于 segment；SIFT ef=24 未变，是更干净的 A/B。",
    )


def plot_ef(ef_rows: Sequence[dict[str, Any]]) -> Path:
    tuning = [row for row in ef_rows if row["split"] == "tuning"]
    holdout = next(row for row in ef_rows if row["split"] == "holdout")
    figure, axis = plt.subplots(figsize=(7.2, 4.9))
    axis.plot([row["ef_search"] for row in tuning], [row["recall_at_10"] for row in tuning], "o-", color="#2563eb", linewidth=2.2, label="1,000-query tuning")
    axis.scatter([holdout["ef_search"]], [holdout["recall_at_10"]], marker="s", s=75, color="#d97706", label="9,000-query holdout")
    axis.axhline(TARGET_RECALL, color="#dc2626", linestyle="--", label="Recall target")
    axis.axvline(320, color="#059669", linestyle=":", label="Selected ef=320")
    axis.set_xlabel("efSearch")
    axis.set_ylabel("Recall@10")
    axis.set_xticks([192, 224, 256, 320, 384])
    axis.set_ylim(0.85, 0.93)
    axis.legend(frameon=False)
    base.style_axis(axis)
    figure.suptitle("Bounded GloVe ef calibration for a single HNSW graph", y=0.98)
    figure.tight_layout()
    return save_figure(
        figure,
        "stage15_fig6_glove_ef_calibration.pdf",
        purpose="在不无限增大 ef 的前提下，为 GloVe 单图集合选择满足召回门槛的最小固定 ef。",
        reading="蓝线是有限网格 192～384 的调优召回，红线是 0.90 门槛；选择第一个越过门槛的 ef=320，再用不重叠的 9,000 条查询验证。",
        conclusion=f"ef=192 的单图召回仅 {tuning[0]['recall_at_10']:.4f}；最小达标值 ef=320 的调优召回为 {next(row['recall_at_10'] for row in tuning if row['ef_search']==320):.4f}，留出召回为 {holdout['recall_at_10']:.5f}。所有 M 固定使用 320。",
        boundary="384 是预先限定的硬上限；后续 fan-out 失败时只增加 P，不再增加 ef。",
    )


def plot_diagnostics(
    rows: Sequence[dict[str, Any]],
    baseline_b: Sequence[dict[str, Any]],
    partition_rows: Sequence[dict[str, Any]],
) -> Path:
    figure, axes = plt.subplots(2, 4, figsize=(16.4, 8.1))
    metrics = (
        ("recall_mean", "Recall@10"),
        ("qps_cv", "QPS CV"),
        ("p95_latency_us", "p95 latency (us)"),
        ("max_worker_cpu_utilization_pct", "Max worker CPU (%)"),
        ("aggregator_cpu_utilization_pct", "Aggregator worker-pool CPU (%)"),
        ("max_network_utilization_pct", "Max 25GbE utilization (%)"),
        ("mean_routing_latency_us", "Mean routing latency (us)"),
    )
    for axis, (field, title) in zip(axes.flat[:7], metrics, strict=True):
        for dataset, linestyle in zip(DATASETS, ("-", "--"), strict=True):
            for method in METHODS:
                current = [row_for(rows, dataset, method, m) for m in PHYSICAL_COUNTS]
                axis.plot(PHYSICAL_COUNTS, [row[field] for row in current], linestyle=linestyle, color=COLORS[method], marker=MARKERS[method], linewidth=1.8, label=f"{DATASET_LABELS[dataset]} {METHOD_LABELS[method]}")
        axis.set_title(title)
        axis.set_xlabel("M")
    axes[0, 0].axhline(TARGET_RECALL, color="#dc2626", linestyle=":")
    positions = [2, 3, 4]
    axes[1, 3].plot(positions, [1 / m for m in positions], "--", color="#6b7280", label="Ideal 1/M")
    for dataset, color in zip(DATASETS, ("#2563eb", "#d97706"), strict=True):
        current = sorted((row for row in partition_rows if row["dataset"] == dataset), key=lambda row: row["logical_shard_count"])
        axes[1, 3].plot(positions, [row["maximum_shard_fraction"] for row in current], "o-", color=color, label=DATASET_LABELS[dataset])
    axes[1, 3].set_title("Largest K-Means shard fraction")
    axes[1, 3].set_xlabel("M")
    axes[1, 3].legend(frameon=False)
    style_all(axes.flat[:7])
    axes[1, 3].set_xticks(positions)
    base.style_axis(axes[1, 3])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Validity, latency, resources, and partition-balance diagnostics", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    max_network = max(row["max_network_utilization_pct"] for row in rows)
    glove_m3 = next(row for row in partition_rows if row["dataset"] == "glove-200-angular" and row["logical_shard_count"] == 3)
    minimum_recall = min(row["recall_min"] for row in [*rows, *baseline_b])
    sift_random_m1 = row_for(rows, "sift1m", "random", 1)
    sift_random_m4 = row_for(rows, "sift1m", "random", 4)
    sift_kmeans_peak = max(
        row_for(rows, "sift1m", "kmeans", machines)[
            "aggregator_cpu_utilization_pct"
        ]
        for machines in PHYSICAL_COUNTS
    )
    return save_figure(
        figure,
        "stage15_fig7_diagnostics.pdf",
        purpose="确认召回与重复稳定性门槛，并检查延迟、worker/聚合端 CPU、网络、路由和 K-Means 分区偏斜是否解释非线性扩展。",
        reading="Recall 必须高于 0.90；聚合 worker-pool CPU 上升而 Qdrant worker CPU 下降表示瓶颈移向 scatter/gather 客户端；网络远离 100% 表示带宽非主瓶颈。",
        conclusion=(
            f"最低正式 Recall 为 {minimum_recall:.5f}，最大网络仅 {max_network:.2f}%。"
            f"SIFT Random 聚合 worker-pool CPU 从 {sift_random_m1['aggregator_cpu_utilization_pct']:.1f}% 升至 {sift_random_m4['aggregator_cpu_utilization_pct']:.1f}%，"
            f"K-Means 峰值 {sift_kmeans_peak:.1f}%；GloVe M=3 最大 shard 占 {100*glove_m3['maximum_shard_fraction']:.1f}%。"
        ),
        boundary="资源利用率是诊断相关性，不单独证明因果；所有正式 E1 重复仍要求 bottleneck_flags 为空且聚合队列不持续增长。",
    )


def comparison_rows(
    rows: Sequence[dict[str, Any]],
    old_rows: Sequence[dict[str, Any]],
    layout15: Sequence[dict[str, Any]],
    layout14: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method in METHODS:
            for machines in PHYSICAL_COUNTS:
                new = row_for(rows, dataset, method, machines)
                old = row_for(old_rows, dataset, method, machines)
                new_layout = layout_for(layout15, dataset, method, machines)
                old_layout = layout_for(layout14, dataset, method, machines)
                output.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "physical_machine_count": machines,
                        "stage14_ef_search": old["ef_search"],
                        "stage15_ef_search": new["ef_search"],
                        "stage14_qps": old["qps_mean"],
                        "stage15_qps": new["qps_mean"],
                        "stage15_to_stage14_qps_ratio": new["qps_mean"] / old["qps_mean"],
                        "stage14_total_segments": old_layout["total_segments"],
                        "stage15_total_segments": new_layout["total_segments"],
                        "stage14_estimated_nonempty_hnsw_segments": old_layout["estimated_nonempty_hnsw_segments"],
                        "stage15_estimated_nonempty_hnsw_segments": new_layout["estimated_nonempty_hnsw_segments"],
                        "stage14_source": old["source"],
                        "stage15_source": new["source"],
                    }
                )
    return output


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def report_markdown(
    rows: Sequence[dict[str, Any]],
    baseline_b: Sequence[dict[str, Any]],
    fanout_rows: Sequence[dict[str, Any]],
    ef_rows: Sequence[dict[str, Any]],
    cleanup: dict[str, Any],
    affinity: dict[str, Any],
    figures: Sequence[Path],
) -> str:
    lines = [
        "# Stage 15：每个 shard 单 HNSW 图的 1～4 台真实机器重测",
        "",
        "状态：`MEASUREMENTS_COMPLETE`。SIFT1M 固定 `efSearch=24`；GloVe 单图在有限网格中选择最小达标值并固定为 `efSearch=320`（硬上限 384）。",
        "",
        "## 正式 QPS",
        "",
        "| 数据集 | 方法 | ef | P(M=1/2/3/4) | M=1 | M=2 | M=3 | M=4 | M=4/M=1 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in METHODS:
            current = [row_for(rows, dataset, method, m) for m in PHYSICAL_COUNTS]
            lines.append(
                "| " + " | ".join(
                    [
                        DATASET_LABELS[dataset], METHOD_LABELS[method], str(FIXED_EF[dataset]),
                        "/".join(str(row["fanout"]) for row in current),
                        *[fmt(row["qps_mean"]) for row in current],
                        f"{current[-1]['normalized_qps_to_m1']:.3f}×",
                    ]
                ) + " |"
            )
    lines.extend([
        "",
        "尾部单机基线漂移：",
        "",
        "| 数据集 | baseline-A | baseline-B | 漂移 |",
        "|---|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        first = row_for(rows, dataset, "random", 1)
        last = next(row for row in baseline_b if row["dataset"] == dataset)
        drift = 100 * (last["qps_mean"] - first["qps_mean"]) / first["qps_mean"]
        lines.append(
            f"| {DATASET_LABELS[dataset]} | {fmt(first['qps_mean'])} | "
            f"{fmt(last['qps_mean'])} | {drift:+.2f}% |"
        )
    lines.extend([
        "",
        "## 固定 ef 的 K-Means fan-out",
        "",
        "| 数据集 | 固定 ef | M=1 | M=2 | M=3 | M=4 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        current = [row for row in fanout_rows if row["dataset"] == dataset]
        lines.append(f"| {DATASET_LABELS[dataset]} | {FIXED_EF[dataset]} | " + " | ".join(str(row["selected_fanout"]) for row in current) + " |")
    tuning = [row for row in ef_rows if row["split"] == "tuning"]
    holdout = next(row for row in ef_rows if row["split"] == "holdout")
    sift_random_m1 = row_for(rows, "sift1m", "random", 1)
    sift_random_m4 = row_for(rows, "sift1m", "random", 4)
    sift_local_work_ratio = (
        sift_random_m4["mean_distance_computations_per_query"]
        / sift_random_m4["fanout"]
        / sift_random_m1["mean_distance_computations_per_query"]
    )
    glove_random_m1 = row_for(rows, "glove-200-angular", "random", 1)
    glove_random_m4 = row_for(rows, "glove-200-angular", "random", 4)
    glove_local_work_ratio = (
        glove_random_m4["mean_distance_computations_per_query"]
        / glove_random_m4["fanout"]
        / glove_random_m1["mean_distance_computations_per_query"]
    )
    lines.extend([
        "",
        "GloVe 单图有限 ef 网格：" + "，".join(f"{row['ef_search']}→{row['recall_at_10']:.4f}" for row in tuning) + f"；最终 ef=320 的独立留出 Recall={holdout['recall_at_10']:.5f}。",
        "",
        "## 现象是否合理",
        "",
        "合理，而且不能要求端到端 QPS 被调成对数增长。HNSW 的对数项只近似描述一张局部图的搜索工作；在 N≈100 万时，从 M=1 分到 M=4，`log(N)/log(N/4)` 只有约 1.11× 的局部收益。在负载均衡的粗略工作模型中，吞吐上界更接近 `M / (P × D_local + routing + scatter/gather + merge)`。",
        "",
        f"- Random 必须 P=M 全广播，因此 M 与 P 在粗略模型中相互抵消，只剩很小的局部图收益。SIFT 的每-shard 距离工作降至 {sift_local_work_ratio:.2f}×，但聚合 worker-pool CPU 从 {sift_random_m1['aggregator_cpu_utilization_pct']:.1f}% 升到 {sift_random_m4['aggregator_cpu_utilization_pct']:.1f}%，scatter/gather 开销使 QPS 反而下降。",
        f"- GloVe 固定 ef=320 后，每-shard 距离工作从 M=1 到 M=4 为 {glove_local_work_ratio:.2f}×，几乎不降；这说明 ef 下限主导了局部工作，因此 Random 只得到 {glove_random_m4['normalized_qps_to_m1']:.2f}×，而不是四倍。",
        "- K-Means 只有在 P 增长慢于 M、分区均衡且协调端不受限时才可能扩展。GloVe 在 M=2/3/4 的实测 P 接近或等于 M，不具备低 fan-out 前提。",
        "- Stage 14 的单机集合含多张非空 HNSW 图，压低了 M=1 基线；Stage 15 将每 shard 固定为一张图后，消除了这个图数量混杂，但也暴露出真实的 fan-out/总工作瓶颈。",
        "- GloVe M=3 的 K-Means 最大 shard 占约 56.1%，所以三台机器并未提供三份均匀算力。",
        "",
        "结论不是“修正后符合预设曲线”，而是：局部 HNSW 工作可随 shard 变小而缓慢下降；端到端 QPS 是否上升由 P 和系统开销决定。本轮数据不支持把 HNSW 分图直接表述成 QPS 的对数扩展定律。",
        "",
        "## PDF 图表",
        "",
        "每张 PDF 页面下方都内嵌中文“用途、如何理解、最终结论、证据边界”。",
        "",
    ])
    lines.extend(f"- [{path.name}](figures/{path.name})" for path in figures)
    lines.extend([
        "",
        "## 完整性",
        "",
        f"- collection 清理证明：{cleanup['proof_count']}/16 通过；四个 peer 均返回 HTTP 404。",
        f"- 绑核、物理核无重叠、实验后恢复：`{affinity['status']}`；Qdrant 已恢复到 `{affinity['restored_cpuset']}`。",
        "- 16 个独立 E1 文件均为 `VALID_E1`；所有正式重复 Recall@10≥0.90、`bottleneck_flags=[]`。",
        "- 16 个 prepare gate 均满足总 segment 数 `2×M`，即每 shard 一张非空 HNSW 图加一个空 appendable segment。",
        "- Stage 14 原证据未覆盖，现重分类为生产默认多-segment 行为。",
        "",
    ])
    return "\n".join(lines)


def summarize() -> dict[str, Any]:
    execution_path = STAGE15_DIR / "execution-complete.json"
    execution = base.load_json(execution_path)
    if execution.get("status") != "MEASUREMENTS_COMPLETE":
        raise ValueError("Stage 15 execution is not complete")
    rows, baseline_b = load_rows(STAGE15_DIR, 15, STAGE)
    old_rows, _ = load_rows(STAGE14_DIR, 14, "stage14-production-default-multisegment")
    layout15 = load_layout_rows(STAGE15_DIR, 15)
    layout14 = load_layout_rows(STAGE14_DIR, 14)
    fanout_rows = load_fanout_rows(rows)
    ef_rows = load_ef_rows()
    partition_rows = [row for row in base.load_partition_rows() if row["logical_shard_count"] in (2, 3, 4)]
    cleanup = audit_cleanup()
    affinity = audit_affinity()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = [
        base.write_csv(SUMMARY_DIR / "physical-qps.csv", base.physical_csv_rows(rows, baseline_b, [])),
        base.write_csv(SUMMARY_DIR / "fanout-fixed-ef.csv", fanout_rows),
        base.write_csv(SUMMARY_DIR / "ef-calibration.csv", ef_rows),
        base.write_csv(SUMMARY_DIR / "segment-layout-stage14-vs-stage15.csv", [*layout14, *layout15]),
        base.write_csv(SUMMARY_DIR / "stage14-vs-stage15.csv", comparison_rows(rows, old_rows, layout15, layout14)),
        base.write_csv(SUMMARY_DIR / "partition-balance.csv", partition_rows),
    ]
    integrity_path = base.write_json(
        SUMMARY_DIR / "integrity-audit.json",
        {"status": "PASS", "cleanup": cleanup, "affinity": affinity, "single_graph_prepare_gate_count": 16},
    )
    figures = [
        plot_qps(rows, baseline_b),
        plot_scaling(rows),
        plot_fanout(fanout_rows),
        plot_work(rows),
        plot_stage14_comparison(rows, old_rows, layout15, layout14),
        plot_ef(ef_rows),
        plot_diagnostics(rows, baseline_b, partition_rows),
    ]
    base.verify_pdfs(figures)
    report_path = base.atomic_write_text(
        SUMMARY_DIR / "RESULTS_zh.md",
        report_markdown(rows, baseline_b, fanout_rows, ef_rows, cleanup, affinity, figures),
    )
    e1_sources = sorted({Path(row["source"]) for row in [*rows, *baseline_b]})
    manifest = {
        "record_type": "stage15_single_hnsw_per_shard_summary",
        "status": "PASS",
        "execution_complete": str(execution_path.resolve()),
        "execution_complete_sha256": base.sha256(execution_path),
        "physical_e1_sources": [{"path": str(path), "sha256": base.sha256(path)} for path in e1_sources],
        "csv_files": [{"path": str(path.resolve()), "sha256": base.sha256(path)} for path in csv_paths],
        "figures": [{"path": str(path.resolve()), "sha256": base.sha256(path), "pages": 1} for path in figures],
        "report": str(report_path.resolve()),
        "report_sha256": base.sha256(report_path),
        "integrity_audit": str(integrity_path.resolve()),
        "integrity_audit_sha256": base.sha256(integrity_path),
        "distinct_e1_file_count": len(e1_sources),
        "cleanup_proof_count": cleanup["proof_count"],
        "single_graph_prepare_gate_count": 16,
    }
    manifest_path = base.write_json(SUMMARY_DIR / "manifest.json", manifest)
    manifest["manifest"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = base.sha256(manifest_path)
    return manifest


def check() -> dict[str, Any]:
    manifest_path = SUMMARY_DIR / "manifest.json"
    manifest = base.load_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise ValueError("Stage 15 summary manifest is not PASS")
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
    base.verify_pdfs([Path(entry["path"]) for entry in manifest["figures"]])
    audit_cleanup()
    audit_affinity()
    load_rows(STAGE15_DIR, 15, STAGE)
    load_layout_rows(STAGE15_DIR, 15)
    return {
        "status": "PASS",
        "manifest": str(manifest_path.resolve()),
        "figure_count": len(manifest["figures"]),
        "csv_count": len(manifest["csv_files"]),
        "distinct_e1_file_count": len(manifest["physical_e1_sources"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = check() if args.check else summarize()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
