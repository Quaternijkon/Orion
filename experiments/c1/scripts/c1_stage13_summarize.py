#!/usr/bin/env python3
"""Summarize and plot the Stage 13 physical-scale/fixed-ef retest.

All outputs stay below the Stage 13 retest directory.  Every generated PDF is
extended with a Chinese footer that states its purpose, reading guidance,
conclusion, and evidence boundary.
"""

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


SCRIPT_DIR = Path(__file__).resolve().parent
C1_DIR = SCRIPT_DIR.parent
REPO_ROOT = C1_DIR.parent.parent
STAGE13_DIR = C1_DIR / "retests" / "stage13-physical1to4-ef-controlled"
SUMMARY_DIR = STAGE13_DIR / "summary"
FIGURES_DIR = SUMMARY_DIR / "figures"

DATASETS = ("sift1m", "glove-200-angular")
DATASET_LABELS = {
    "sift1m": "SIFT1M",
    "glove-200-angular": "GloVe-200-angular",
}
METHODS = ("random", "kmeans")
METHOD_LABELS = {"random": "Random", "kmeans": "K-Means"}
COLORS = {"random": "#2563eb", "kmeans": "#d97706"}
MARKERS = {"random": "o", "kmeans": "s"}
PHYSICAL_COUNTS = (1, 2, 3, 4)
LOGICAL_COUNTS = (1, 2, 3, 4, 8, 16, 32)
FIXED_EF = {"sift1m": 24, "glove-200-angular": 192}
EF_GRIDS = {
    "sift1m": (16, 24, 32, 48, 64),
    "glove-200-angular": (64, 96, 128, 192),
}
TARGET_RECALL = 0.90
CV_THRESHOLD = 0.05
FOOTER_HEIGHT = 242.0
FOOTER_TITLE = "Stage 13 中文图解"
PDF_CREATOR = "Orion C1 Stage 13 summarizer"

OLD_E1_FILES = {
    "sift1m": {
        ("random", 1): "stage6-e1-corrected-sift1m-common-m1.json",
        ("kmeans", 1): "stage6-e1-corrected-sift1m-common-m1.json",
        ("random", 2): "stage6-e1-corrected-sift1m-random-m2.json",
        ("kmeans", 2): "stage6-e1-corrected-sift1m-kmeans-m2.json",
        ("random", 4): "stage6-e1-corrected-sift1m-random-m4.json",
        ("kmeans", 4): "stage6-e1-corrected-sift1m-kmeans-m4.json",
    },
    "glove-200-angular": {
        ("random", 1): "stage8-e1-corrected-glove-200-angular-common-m1.json",
        ("kmeans", 1): "stage8-e1-corrected-glove-200-angular-common-m1.json",
        ("random", 2): "stage8-e1-corrected-glove-200-angular-random-m2.json",
        ("kmeans", 2): "stage8-e1-corrected-glove-200-angular-kmeans-m2.json",
        ("random", 4): "stage8-e1-corrected-glove-200-angular-random-m4.json",
        ("kmeans", 4): "stage8-e1-corrected-glove-200-angular-kmeans-m4.json",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def write_json(path: Path, payload: Any) -> Path:
    return atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0])
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if list(row) != fields:
                raise ValueError(f"inconsistent CSV fields for {path}")
            writer.writerow(row)
    os.replace(temporary, path)
    return path


def repetition_mean(payload: dict[str, Any], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in payload["repetitions"])


def new_e1_path(dataset: str, method: str, machines: int, baseline: str = "a") -> Path:
    directory = STAGE13_DIR / "e1"
    if machines == 1:
        return directory / (
            f"stage13-e1-{dataset}-random-m1-baseline-{baseline}.json"
        )
    return directory / f"stage13-e1-{dataset}-{method}-m{machines}-primary.json"


def e1_row(
    path: Path,
    *,
    display_method: str,
    stage: str,
    baseline_label: str = "",
) -> dict[str, Any]:
    payload = load_json(path)
    repetitions = list(payload.get("repetitions") or [])
    if payload.get("status") != "VALID_E1":
        raise ValueError(f"invalid E1 status: {path}")
    if len(repetitions) != int(payload["repetition_count"]):
        raise ValueError(f"repetition-count mismatch: {path}")
    if len(repetitions) < 3:
        raise ValueError(f"fewer than three repetitions: {path}")
    recalls = [float(rep["achieved_recall"]) for rep in repetitions]
    if min(recalls) < TARGET_RECALL:
        raise ValueError(f"recall below target: {path}")
    if any(rep.get("bottleneck_flags") for rep in repetitions):
        raise ValueError(f"bottleneck flags present: {path}")
    if any(rep.get("persistent_aggregator_queue_growth") for rep in repetitions):
        raise ValueError(f"persistent queue growth: {path}")
    cv = float(payload["qps_coefficient_of_variation"])
    if cv > CV_THRESHOLD and len(repetitions) < 5:
        raise ValueError(f"CV above threshold without five repetitions: {path}")
    qps = payload["qps_confidence_interval_95"]
    completed = [float(rep["completed_qps"]) for rep in repetitions]
    if not math.isclose(statistics.fmean(completed), float(qps["mean"]), rel_tol=1e-12):
        raise ValueError(f"QPS mean mismatch: {path}")
    resource = [rep["resource_utilization"] for rep in repetitions]
    return {
        "stage": stage,
        "dataset": str(payload["dataset"]),
        "method": display_method,
        "physical_machine_count": int(payload["physical_machine_count"]),
        "logical_shard_count": int(payload["logical_shard_count"]),
        "fanout": int(payload["fanout"]),
        "ef_search": int(payload["ef_search"]),
        "selected_concurrency": int(payload["selected_concurrency"]),
        "repetition_count": len(repetitions),
        "qps_mean": float(qps["mean"]),
        "qps_sample_std": float(qps["sample_std"]),
        "qps_ci95_lower": float(qps["lower"]),
        "qps_ci95_upper": float(qps["upper"]),
        "qps_cv": cv,
        "recall_mean": statistics.fmean(recalls),
        "recall_min": min(recalls),
        "p95_latency_us": repetition_mean(payload, "p95_latency_us"),
        "mean_latency_us": repetition_mean(payload, "mean_latency_us"),
        "mean_routing_latency_us": repetition_mean(payload, "mean_routing_latency_us"),
        "mean_distance_computations_per_query": repetition_mean(
            payload, "mean_distance_computations_per_query"
        ),
        "mean_nodes_visited_per_query": repetition_mean(
            payload, "mean_nodes_visited_per_query"
        ),
        "mean_worker_cpu_time_us_per_query": repetition_mean(
            payload, "mean_worker_cpu_time_us_per_query"
        ),
        "max_worker_cpu_utilization_pct": max(
            float(item["max_worker_cpu_utilization_pct"]) for item in resource
        ),
        "max_network_utilization_pct": max(
            float(item["max_network_utilization_pct"]) for item in resource
        ),
        "baseline_label": baseline_label,
        "source": str(path.resolve()),
        "source_sha256": sha256(path),
    }


def load_e1_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    new_rows: list[dict[str, Any]] = []
    baseline_b_rows: list[dict[str, Any]] = []
    old_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method in METHODS:
            new_rows.append(
                e1_row(
                    new_e1_path(dataset, method, 1, "a"),
                    display_method=method,
                    stage="stage13-fixed-ef",
                    baseline_label="A",
                )
            )
            for machines in PHYSICAL_COUNTS[1:]:
                new_rows.append(
                    e1_row(
                        new_e1_path(dataset, method, machines),
                        display_method=method,
                        stage="stage13-fixed-ef",
                    )
                )
        baseline_b_rows.append(
            e1_row(
                new_e1_path(dataset, "random", 1, "b"),
                display_method="random",
                stage="stage13-fixed-ef",
                baseline_label="B",
            )
        )
        for (method, machines), filename in OLD_E1_FILES[dataset].items():
            old_rows.append(
                e1_row(
                    C1_DIR / "runs" / filename,
                    display_method=method,
                    stage="stage6/8-variable-ef",
                )
            )
    return new_rows, baseline_b_rows, old_rows


def row_for(
    rows: Iterable[dict[str, Any]], dataset: str, method: str, machines: int
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["method"] == method
        and row["physical_machine_count"] == machines
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {dataset}/{method}/M={machines}: {matches}")
    return matches[0]


def add_normalization(rows: list[dict[str, Any]]) -> None:
    for stage in sorted({str(row["stage"]) for row in rows}):
        stage_rows = [row for row in rows if row["stage"] == stage]
        for dataset in DATASETS:
            for method in METHODS:
                relevant = [
                    row
                    for row in stage_rows
                    if row["dataset"] == dataset and row["method"] == method
                ]
                if not relevant:
                    continue
                baseline = row_for(relevant, dataset, method, 1)["qps_mean"]
                for row in relevant:
                    row["normalized_qps_to_m1"] = float(row["qps_mean"]) / float(baseline)
                    row["scaling_efficiency"] = (
                        float(row["normalized_qps_to_m1"])
                        / int(row["physical_machine_count"])
                    )


def load_fanout_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    holdout_dir = STAGE13_DIR / "fanout" / "holdout"
    tuning_dir = STAGE13_DIR / "fanout" / "tuning"
    for dataset in DATASETS:
        ef = FIXED_EF[dataset]
        for logical in LOGICAL_COUNTS:
            selected_path = (
                holdout_dir
                / f"{dataset}-kmeans-m{logical}-ef{ef}-selected.json"
            )
            selected = load_json(selected_path)
            fanout = int(selected["fanout"])
            attempt_path = holdout_dir / f"{dataset}-kmeans-m{logical}-ef{ef}-p{fanout}.json"
            attempt = load_json(attempt_path)
            if selected.get("selected_after_holdout") is not True:
                raise ValueError(f"holdout selection did not pass: {selected_path}")
            if float(selected["achieved_recall"]) < TARGET_RECALL:
                raise ValueError(f"holdout recall below target: {selected_path}")
            tuning = load_json(tuning_dir / f"{dataset}-kmeans-m{logical}.json")
            attempts = list(selected.get("holdout_attempts") or [])
            rows.append(
                {
                    "dataset": dataset,
                    "logical_shard_count": logical,
                    "physical_machine_count": int(tuning["physical_machine_count"]),
                    "fixed_ef": ef,
                    "tuning_selected_fanout": int(tuning["fixed_ef_selected_fanout"]),
                    "holdout_selected_fanout": fanout,
                    "fanout_fraction": fanout / logical,
                    "holdout_recall": float(selected["achieved_recall"]),
                    "oracle_fanout_mean": float(attempt["oracle_fanout_mean"]),
                    "oracle_fanout_median": float(attempt["oracle_fanout_median"]),
                    "oracle_fanout_p95": float(attempt["oracle_fanout_p95"]),
                    "holdout_attempt_count": len(attempts),
                    "initial_holdout_status": str(attempts[0]["result_status"]),
                    "source": str(selected_path.resolve()),
                    "source_sha256": sha256(selected_path),
                }
            )
    return rows


def load_sensitivity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tuning_dir = STAGE13_DIR / "fanout" / "tuning"
    for dataset in DATASETS:
        for logical in LOGICAL_COUNTS:
            path = tuning_dir / f"{dataset}-kmeans-m{logical}.json"
            payload = load_json(path)
            sensitivity = payload["ef_sensitivity"]
            for ef in EF_GRIDS[dataset]:
                point = sensitivity[str(ef)]
                rows.append(
                    {
                        "dataset": dataset,
                        "logical_shard_count": logical,
                        "physical_machine_count": int(payload["physical_machine_count"]),
                        "ef_search": ef,
                        "is_primary_fixed_ef": ef == FIXED_EF[dataset],
                        "status": str(point["status"]),
                        "selected_minimum_fanout": point["selected_minimum_fanout"],
                        "selected_recall_at_10": point["selected_recall_at_10"],
                        "full_fanout_recall_at_10": float(point["full_fanout_recall_at_10"]),
                        "source": str(path.resolve()),
                        "source_sha256": sha256(path),
                    }
                )
    return rows


def partition_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(np.asarray(data["metadata_json"]).item()))


def load_partition_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for logical in LOGICAL_COUNTS[1:]:
            tuning = load_json(
                STAGE13_DIR / "fanout" / "tuning" / f"{dataset}-kmeans-m{logical}.json"
            )
            artifact = Path(tuning["partition_artifact"])
            if sha256(artifact) != str(tuning["partition_sha256"]):
                raise ValueError(f"partition hash mismatch: {artifact}")
            metadata = partition_metadata(artifact)
            counts = [int(value) for value in metadata["shard_counts"]]
            kmeans = metadata.get("kmeans") or {}
            history = list(kmeans.get("history") or [])
            tolerance = float(kmeans.get("tolerance") or 0.0)
            final_movement = (
                float(history[-1]["max_centroid_movement"]) if history else float("nan")
            )
            converged = bool(history and final_movement <= tolerance)
            rows.append(
                {
                    "dataset": dataset,
                    "logical_shard_count": logical,
                    "physical_machine_count": int(tuning["physical_machine_count"]),
                    "point_count": sum(counts),
                    "minimum_shard_count": min(counts),
                    "maximum_shard_count": max(counts),
                    "maximum_shard_fraction": max(counts) / sum(counts),
                    "max_to_min_ratio": max(counts) / min(counts),
                    "kmeans_iterations": len(history),
                    "kmeans_maximum_iterations": int(kmeans.get("maximum_iterations") or 0),
                    "kmeans_final_centroid_movement": final_movement,
                    "kmeans_tolerance": tolerance,
                    "kmeans_converged": converged,
                    "shard_counts": ";".join(map(str, counts)),
                    "partition_artifact": str(artifact),
                    "partition_artifact_sha256": sha256(artifact),
                }
            )
    return rows


def audit_cleanup() -> dict[str, Any]:
    directory = STAGE13_DIR / "cleanup"
    paths = sorted(directory.glob("*.json"))
    if len(paths) != 22:
        raise ValueError(f"expected 22 cleanup proofs, found {len(paths)}")
    failures: list[str] = []
    entries: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        peer_codes = {
            host: int(details["http_status"])
            for host, details in payload["peers"].items()
        }
        controller_path = Path(payload["controller_storage_path"])
        current_absent = not controller_path.exists()
        valid = (
            payload.get("status") == "VERIFIED_DELETED"
            and payload.get("controller_storage_absent") is True
            and current_absent
            and set(peer_codes.values()) == {404}
            and len(peer_codes) == 4
        )
        if not valid:
            failures.append(path.name)
        entries.append(
            {
                "collection": payload["collection"],
                "proof": str(path.resolve()),
                "proof_sha256": sha256(path),
                "peer_http_statuses": peer_codes,
                "controller_storage_absent_recorded": payload[
                    "controller_storage_absent"
                ],
                "controller_storage_absent_current": current_absent,
                "valid": valid,
            }
        )
    if failures:
        raise ValueError(f"cleanup audit failures: {failures}")
    return {
        "status": "PASS",
        "proof_count": len(entries),
        "all_four_peer_statuses_404": True,
        "all_controller_storage_paths_absent": True,
        "entries": entries,
    }


def insert_footer(
    source_pdf: Path,
    destination: Path,
    *,
    purpose: str,
    reading: str,
    conclusion: str,
    boundary: str,
) -> None:
    source = pymupdf.open(source_pdf)
    output = pymupdf.open()
    font_buffer = pymupdf.Font(fontname="china-s").buffer
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
            pymupdf.Point(25, rectangle.height + 14),
            pymupdf.Point(rectangle.width - 25, rectangle.height + 14),
            color=(0.70, 0.75, 0.82),
            width=0.8,
        )
        page.insert_font(fontname="stage13-cjk", fontbuffer=font_buffer)
        title_result = page.insert_textbox(
            pymupdf.Rect(
                30, rectangle.height + 24, rectangle.width - 30, rectangle.height + 49
            ),
            FOOTER_TITLE,
            fontname="stage13-cjk",
            fontsize=13,
            lineheight=1.0,
            color=(0.06, 0.22, 0.48),
        )
        if title_result < 0:
            raise RuntimeError(f"footer title does not fit: {destination}")
        body_rectangle = pymupdf.Rect(
            30,
            rectangle.height + 54,
            rectangle.width - 30,
            rectangle.height + FOOTER_HEIGHT - 14,
        )
        for fontsize in (10.8, 10.4, 10.0, 9.6, 9.2):
            result = page.insert_textbox(
                body_rectangle,
                body,
                fontname="stage13-cjk",
                fontsize=fontsize,
                lineheight=1.22,
                color=(0.07, 0.07, 0.08),
            )
            if result >= 0:
                break
        else:
            raise RuntimeError(f"footer body does not fit: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.pdf")
    output.save(temporary, garbage=4, deflate=True)
    output.close()
    source.close()
    os.replace(temporary, destination)


def save_annotated_figure(
    figure: Any,
    destination: Path,
    *,
    purpose: str,
    reading: str,
    conclusion: str,
    boundary: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.stem + "-", suffix=".pdf", dir=destination.parent, delete=False
    ) as handle:
        base = Path(handle.name)
    try:
        figure.savefig(
            base,
            format="pdf",
            bbox_inches="tight",
            metadata={
                "Title": destination.stem,
                "Creator": PDF_CREATOR,
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(figure)
        insert_footer(
            base,
            destination,
            purpose=purpose,
            reading=reading,
            conclusion=conclusion,
            boundary=boundary,
        )
    finally:
        base.unlink(missing_ok=True)
    return destination


def style_axis(axis: Any) -> None:
    axis.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)


def plot_physical_qps(
    new_rows: Sequence[dict[str, Any]],
    old_rows: Sequence[dict[str, Any]],
    baseline_b_rows: Sequence[dict[str, Any]],
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for method in METHODS:
            current = [
                row
                for row in new_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            current.sort(key=lambda row: row["physical_machine_count"])
            x = [row["physical_machine_count"] for row in current]
            y = [row["qps_mean"] for row in current]
            low = [row["qps_mean"] - row["qps_ci95_lower"] for row in current]
            high = [row["qps_ci95_upper"] - row["qps_mean"] for row in current]
            axis.errorbar(
                x,
                y,
                yerr=[low, high],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.2,
                capsize=4,
                label=f"Stage 13 {METHOD_LABELS[method]} (fixed ef)",
            )
            previous = [
                row
                for row in old_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            previous.sort(key=lambda row: row["physical_machine_count"])
            axis.plot(
                [row["physical_machine_count"] for row in previous],
                [row["qps_mean"] for row in previous],
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=":",
                linewidth=1.6,
                alpha=0.6,
                label=f"Old {METHOD_LABELS[method]} (variable ef)",
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
            label="Stage 13 baseline-B",
        )
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Physical machines M (real hosts)")
        axis.set_ylabel("Completed QPS")
        axis.set_xticks(PHYSICAL_COUNTS)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("Physical 1-4 host QPS: Stage 13 retest vs old curve", y=1.04)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return save_annotated_figure(
        figure,
        FIGURES_DIR / "stage13_fig1_physical_qps.pdf",
        purpose="重新检验真实物理机器数 M=1、2、3、4 时的端到端 QPS，并把固定 ef 的 Stage 13 曲线与旧的逐点变化 ef 曲线并列。",
        reading="横轴只表示真实主机数；实线及误差条是本轮固定 ef 的新结果，虚点线是旧结果，M=1 旁的叉号是尾部 baseline-B。先看同一实线随 M 的变化，再看 95% 置信区间和 A/B 基线是否稳定。",
        conclusion="旧曲线不能继续代表受控 ef 的 1～4 机扩展。SIFT Random 从约 12.5k 降至 4.67k QPS，K-Means 在 M=4 约 10.5k；GloVe Random 在 M=4 约 4.48k、K-Means 约 3.27k，但均远低于理想四倍扩展。",
        boundary="旧结果的 ef 随配置变化，故新旧绝对 QPS 差异不能只归因于机器数；Stage 13 新曲线才是本次问题的主证据。M=8、16、32 不在本图，因为它们只有四台物理机。",
    )


def plot_normalized_scaling(
    new_rows: Sequence[dict[str, Any]], old_rows: Sequence[dict[str, Any]]
) -> Path:
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
            current = sorted(
                (
                    row
                    for row in new_rows
                    if row["dataset"] == dataset and row["method"] == method
                ),
                key=lambda row: row["physical_machine_count"],
            )
            axis.plot(
                [row["physical_machine_count"] for row in current],
                [row["normalized_qps_to_m1"] for row in current],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.2,
                label=f"Stage 13 {METHOD_LABELS[method]}",
            )
            previous = sorted(
                (
                    row
                    for row in old_rows
                    if row["dataset"] == dataset and row["method"] == method
                ),
                key=lambda row: row["physical_machine_count"],
            )
            axis.plot(
                [row["physical_machine_count"] for row in previous],
                [row["normalized_qps_to_m1"] for row in previous],
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=":",
                linewidth=1.5,
                alpha=0.6,
                label=f"Old {METHOD_LABELS[method]}",
            )
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Physical machines M (real hosts)")
        axis.set_ylabel("QPS / Stage-specific M=1 QPS")
        axis.set_xticks(PHYSICAL_COUNTS)
        axis.set_ylim(bottom=0)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("Normalized physical scaling", y=1.04)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return save_annotated_figure(
        figure,
        FIGURES_DIR / "stage13_fig2_normalized_scaling.pdf",
        purpose="去除数据集绝对 QPS 量级差异，直接判断增加真实机器是否接近理想线性扩展。",
        reading="纵轴以各阶段自己的 M=1 为 1；灰色 y=M 是理想线。M=4 若只有 2，表示四台机器只达到单机两倍吞吐；低于 1 甚至表示加机器后反而变慢。实线为新曲线，虚点线为旧曲线。",
        conclusion="Stage 13 的 M=4：SIFT Random 仅约 0.37×、K-Means 约 0.84×；GloVe Random 约 2.16×、K-Means约 1.58×。两数据集均未接近 4×，SIFT 还出现明显负扩展。",
        boundary="归一化只能比较曲线形状，不能消除旧实验逐点 ef 不同、运行时环境变化等混杂因素；主结论依据 Stage 13 实线。",
    )


def plot_fanout(fanout_rows: Sequence[dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    positions = list(range(len(LOGICAL_COUNTS)))
    for axis, dataset in zip(axes, DATASETS, strict=True):
        rows = sorted(
            (row for row in fanout_rows if row["dataset"] == dataset),
            key=lambda row: row["logical_shard_count"],
        )
        selected = [row["holdout_selected_fanout"] for row in rows]
        oracle_mean = [row["oracle_fanout_mean"] for row in rows]
        oracle_p95 = [row["oracle_fanout_p95"] for row in rows]
        axis.plot(
            positions,
            LOGICAL_COUNTS,
            color="#6b7280",
            linestyle="--",
            marker=".",
            label="Broadcast P=M",
        )
        axis.plot(
            positions,
            selected,
            color="#d97706",
            marker="s",
            linewidth=2.4,
            label=f"Selected P (ef={FIXED_EF[dataset]})",
        )
        axis.plot(
            positions,
            oracle_mean,
            color="#059669",
            marker="o",
            linestyle="-.",
            linewidth=1.8,
            label="Per-query oracle mean",
        )
        axis.plot(
            positions,
            oracle_p95,
            color="#0f766e",
            marker="^",
            linestyle=":",
            linewidth=1.6,
            label="Per-query oracle p95",
        )
        for x, y in zip(positions, selected, strict=True):
            axis.annotate(str(y), (x, y), xytext=(0, 7), textcoords="offset points", ha="center")
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Logical shards M (M>4 uses four physical hosts)")
        axis.set_ylabel("Visited shards / fan-out P")
        axis.set_xticks(positions, [str(value) for value in LOGICAL_COUNTS])
        axis.set_ylim(bottom=0)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.suptitle("K-Means fan-out with dataset-fixed ef and held-out recall gate", y=1.04)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return save_annotated_figure(
        figure,
        FIGURES_DIR / "stage13_fig3_fixed_ef_fanout.pdf",
        purpose="在不继续增大 ef 的前提下，测量 K-Means 为达到 Recall@10≥0.90 实际必须访问多少逻辑分片。",
        reading="横轴是逻辑分片数；M=8、16、32 仍只运行在四台物理机。橙线是通过独立 9,000 查询留出集的服务 fan-out；灰线是全广播；绿色线是逐查询 oracle 的平均值和 p95。",
        conclusion="SIFT 的受控序列为 1/1/2/2/2/3/4；GloVe 为 1/2/3/3/5/7/8。GloVe 的旧低 fan-out 判断被明显纠正；SIFT 仍可保持较低 fan-out，但该结果是在 ef=24 和独立留出召回门槛下实测得到，并非无限增大 ef。",
        boundary="橙线是每个配置统一使用的固定服务 fan-out，不是每条查询动态变化；绿色 oracle 仅表示理论逐查询下界。",
    )


def sensitivity_matrix(
    rows: Sequence[dict[str, Any]], dataset: str, field: str
) -> np.ndarray:
    values = np.full((len(LOGICAL_COUNTS), len(EF_GRIDS[dataset])), np.nan)
    for row in rows:
        if row["dataset"] != dataset:
            continue
        row_index = LOGICAL_COUNTS.index(int(row["logical_shard_count"]))
        column_index = EF_GRIDS[dataset].index(int(row["ef_search"]))
        value = row[field]
        if value is not None:
            values[row_index, column_index] = float(value)
    return values


def annotate_heatmap(axis: Any, values: np.ndarray, *, integer: bool) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = "X" if np.isnan(value) else (str(int(value)) if integer else f"{value:.3f}")
            axis.text(column, row, text, ha="center", va="center", fontsize=8.2)


def plot_sensitivity(sensitivity_rows: Sequence[dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.1))
    for column, dataset in enumerate(DATASETS):
        fanout = sensitivity_matrix(sensitivity_rows, dataset, "selected_minimum_fanout")
        recall = sensitivity_matrix(sensitivity_rows, dataset, "full_fanout_recall_at_10")
        masked = np.ma.masked_invalid(fanout)
        cmap = plt.get_cmap("YlOrBr").copy()
        cmap.set_bad("#d1d5db")
        axes[0, column].imshow(masked, aspect="auto", cmap=cmap, vmin=1)
        annotate_heatmap(axes[0, column], fanout, integer=True)
        axes[1, column].imshow(recall, aspect="auto", cmap="YlGn", vmin=0.80, vmax=1.0)
        annotate_heatmap(axes[1, column], recall, integer=False)
        fixed_column = EF_GRIDS[dataset].index(FIXED_EF[dataset])
        for axis in (axes[0, column], axes[1, column]):
            axis.add_patch(
                plt.Rectangle(
                    (fixed_column - 0.5, -0.5),
                    1,
                    len(LOGICAL_COUNTS),
                    fill=False,
                    edgecolor="#dc2626",
                    linewidth=2.0,
                )
            )
            axis.set_xticks(
                range(len(EF_GRIDS[dataset])),
                [str(value) for value in EF_GRIDS[dataset]],
            )
            axis.set_yticks(
                range(len(LOGICAL_COUNTS)),
                [str(value) for value in LOGICAL_COUNTS],
            )
            axis.set_xlabel("efSearch (red box = primary fixed ef)")
            axis.set_ylabel("Logical shards M")
        axes[0, column].set_title(
            f"{DATASET_LABELS[dataset]}: minimum tuning fan-out (X=infeasible)"
        )
        axes[1, column].set_title(
            f"{DATASET_LABELS[dataset]}: full-fan-out Recall@10"
        )
    figure.suptitle("Finite ef sensitivity grid", y=1.01)
    figure.tight_layout()
    return save_annotated_figure(
        figure,
        FIGURES_DIR / "stage13_fig4_ef_sensitivity.pdf",
        purpose="检查有限 ef 范围内，ef 与所需 fan-out 的耦合关系，并验证为何主实验必须固定 ef。",
        reading="上排数字是在 1,000 条调优查询上达到 Recall≥0.90 的最小 fan-out，X 表示即使全 fan-out 也不达标；下排是全 fan-out 的召回。红框列是主实验固定 ef。若向右提高 ef 后上排数字下降，说明增大 ef 会人为压低 fan-out。",
        conclusion="低 ef 下存在大量不可行点；提高 ef 确实可以减少所需 fan-out。因此旧的联合调节 ef 与 fan-out 会混淆两者作用。本轮把 SIFT 固定在 24、GloVe 固定在 192，并禁止继续增大，避免了这种偏差。",
        boundary="热图上排来自独立于正式 QPS 的 1,000 条调优查询；主 fan-out 最终还经过 9,000 条留出验证，M=16 GloVe 因此从调优值 6 上调为 7。",
    )


def plot_recall_stability(
    new_rows: Sequence[dict[str, Any]], baseline_b_rows: Sequence[dict[str, Any]]
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for row_index, dataset in enumerate(DATASETS):
        recall_axis = axes[row_index, 0]
        cv_axis = axes[row_index, 1]
        for method in METHODS:
            rows = sorted(
                (
                    row
                    for row in new_rows
                    if row["dataset"] == dataset and row["method"] == method
                ),
                key=lambda row: row["physical_machine_count"],
            )
            x = [row["physical_machine_count"] for row in rows]
            recall_axis.plot(
                x,
                [row["recall_mean"] for row in rows],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                label=METHOD_LABELS[method],
            )
            cv_axis.plot(
                x,
                [100 * row["qps_cv"] for row in rows],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                label=METHOD_LABELS[method],
            )
        baseline_b = next(row for row in baseline_b_rows if row["dataset"] == dataset)
        recall_axis.scatter(
            [1.08], [baseline_b["recall_mean"]], color="#111827", marker="x", s=60
        )
        cv_axis.scatter([1.08], [100 * baseline_b["qps_cv"]], color="#111827", marker="x", s=60)
        recall_axis.axhline(TARGET_RECALL, color="#dc2626", linestyle="--", label="Recall target")
        cv_axis.axhline(100 * CV_THRESHOLD, color="#dc2626", linestyle="--", label="5% CV trigger")
        recall_axis.set_title(f"{DATASET_LABELS[dataset]} Recall@10")
        cv_axis.set_title(f"{DATASET_LABELS[dataset]} QPS coefficient of variation")
        recall_axis.set_ylabel("Recall@10")
        cv_axis.set_ylabel("CV (%)")
        for axis in (recall_axis, cv_axis):
            axis.set_xlabel("Physical machines M")
            axis.set_xticks(PHYSICAL_COUNTS)
            style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("Recall validity and QPS stability", y=1.01)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return save_annotated_figure(
        figure,
        FIGURES_DIR / "stage13_fig5_recall_stability.pdf",
        purpose="确认每个 QPS 点都满足召回门槛，并检查独立重复之间的波动是否会削弱扩展结论。",
        reading="左列必须全部位于红色 Recall=0.90 线上方；右列红线是 CV=5% 的加测触发线。SIFT M=2 超线后使用了五次重复，其置信区间应比普通三次重复更谨慎解读。叉号为 baseline-B。",
        conclusion="所有正式点均满足 Recall@10≥0.90。GloVe 的 CV 全部较低；SIFT 的 Random M=2 和 K-Means M=2 在五次重复后 CV 仍约 5.7% 和 6.3%，因此这两个点不确定性较大，但 M=3、4 的明显退化趋势不依赖单个 M=2 点。",
        boundary="CV 描述重复波动，不代表系统误差；误差条、baseline A/B 和受控参数需联合判断。",
    )


def plot_baseline_partition(
    new_rows: Sequence[dict[str, Any]],
    baseline_b_rows: Sequence[dict[str, Any]],
    partition_rows: Sequence[dict[str, Any]],
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.1, 4.9))
    x = np.arange(len(DATASETS))
    width = 0.34
    baseline_a = [row_for(new_rows, dataset, "random", 1) for dataset in DATASETS]
    baseline_b = [next(row for row in baseline_b_rows if row["dataset"] == dataset) for dataset in DATASETS]
    axes[0].bar(x - width / 2, [row["qps_mean"] for row in baseline_a], width, label="Baseline-A")
    axes[0].bar(x + width / 2, [row["qps_mean"] for row in baseline_b], width, label="Baseline-B")
    for index, (a, b) in enumerate(zip(baseline_a, baseline_b, strict=True)):
        drift = 100 * (b["qps_mean"] - a["qps_mean"]) / a["qps_mean"]
        axes[0].annotate(
            f"{drift:+.2f}%",
            (index, max(a["qps_mean"], b["qps_mean"])),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )
    axes[0].set_xticks(x, [DATASET_LABELS[value] for value in DATASETS])
    axes[0].set_ylabel("Completed QPS")
    axes[0].set_title("Single-host baseline drift")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    positions = list(range(1, len(LOGICAL_COUNTS)))
    ideal = [1 / value for value in LOGICAL_COUNTS[1:]]
    axes[1].plot(
        positions,
        ideal,
        color="#6b7280",
        linestyle="--",
        label="Ideal equal max share = 1/M",
    )
    for dataset, color in zip(DATASETS, ("#2563eb", "#d97706"), strict=True):
        rows = sorted(
            (row for row in partition_rows if row["dataset"] == dataset),
            key=lambda row: row["logical_shard_count"],
        )
        axes[1].plot(
            positions,
            [row["maximum_shard_fraction"] for row in rows],
            marker="o",
            linewidth=2.1,
            color=color,
            label=DATASET_LABELS[dataset],
        )
    axes[1].set_xticks(positions, [str(value) for value in LOGICAL_COUNTS[1:]])
    axes[1].set_xlabel("Logical shards M")
    axes[1].set_ylabel("Largest K-Means shard / all vectors")
    axes[1].set_title("K-Means partition imbalance")
    axes[1].legend(frameon=False)
    style_axis(axes[1])
    figure.suptitle("Run stability and partition balance", y=1.02)
    figure.tight_layout()
    return save_annotated_figure(
        figure,
        FIGURES_DIR / "stage13_fig6_baseline_partition.pdf",
        purpose="检查长时间重测期间单机性能是否漂移，并揭示 K-Means 分区是否真正把负载平均分给机器。",
        reading="左图比较实验开头和结尾的同一单机配置，柱顶百分比是漂移；右图越接近灰色 1/M 线越均衡，越高表示最大 shard 主导更多工作。",
        conclusion="baseline-B 相对 A：SIFT 约 -3.93%，GloVe 约 -1.09%，不足以解释主要曲线差异。M=3 的 K-Means 明显偏斜：SIFT 最大 shard 约占 51.4%，GloVe 约占 56.1%，且两次 M=3 聚类均达到 12 次迭代上限而未收敛；机器数不能被当作等量负载数。",
        boundary="分区偏斜解释潜在负载瓶颈，但不能单独证明 QPS 因果；QPS 仍以端到端实测为准。",
    )


def plot_latency_resources(new_rows: Sequence[dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.8))
    for row_index, dataset in enumerate(DATASETS):
        for method in METHODS:
            rows = sorted(
                (
                    row
                    for row in new_rows
                    if row["dataset"] == dataset and row["method"] == method
                ),
                key=lambda row: row["physical_machine_count"],
            )
            x = [row["physical_machine_count"] for row in rows]
            axes[row_index, 0].plot(
                x,
                [row["p95_latency_us"] / 1000 for row in rows],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                label=METHOD_LABELS[method],
            )
            axes[row_index, 1].plot(
                x,
                [row["max_worker_cpu_utilization_pct"] for row in rows],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                label=METHOD_LABELS[method],
            )
            axes[row_index, 2].plot(
                x,
                [row["max_network_utilization_pct"] for row in rows],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                label=METHOD_LABELS[method],
            )
        titles = ("p95 latency (ms)", "Max worker CPU (%)", "Max 25GbE utilization (%)")
        for column, title in enumerate(titles):
            axes[row_index, column].set_title(f"{DATASET_LABELS[dataset]}: {title}")
            axes[row_index, column].set_xlabel("Physical machines M")
            axes[row_index, column].set_xticks(PHYSICAL_COUNTS)
            style_axis(axes[row_index, column])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Latency and resource diagnostics for all Stage 13 QPS points", y=1.01)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    maximum_network = max(float(row["max_network_utilization_pct"]) for row in new_rows)
    return save_annotated_figure(
        figure,
        FIGURES_DIR / "stage13_fig7_latency_resources.pdf",
        purpose="为全部正式 QPS 点提供延迟、worker CPU 和网络利用率诊断，帮助判断扩展曲线为何偏离理想值。",
        reading="每行对应一个数据集；左列看 p95 延迟是否随 scatter 增长，中列看 worker CPU 是否接近保留核心上限，右列看 25GbE 是否饱和。资源图用于定位瓶颈线索，不能取代 QPS 与召回主指标。",
        conclusion=f"所有配置的最大 25GbE 利用率不超过约 {maximum_network:.2f}%，网络带宽没有饱和；扩展损失更符合多分片搜索、聚合等待、CPU 与调度开销，而不是链路容量不足。",
        boundary="利用率是各重复期间记录到的最大方向值；短时队列、软件栈和负载偏斜仍可能影响延迟。",
    )


def physical_csv_rows(
    new_rows: Sequence[dict[str, Any]],
    baseline_b_rows: Sequence[dict[str, Any]],
    old_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in [*new_rows, *baseline_b_rows, *old_rows]:
        rows.append(
            {
                "stage": row["stage"],
                "dataset": row["dataset"],
                "method": row["method"],
                "physical_machine_count": row["physical_machine_count"],
                "logical_shard_count": row["logical_shard_count"],
                "fanout": row["fanout"],
                "ef_search": row["ef_search"],
                "baseline_label": row["baseline_label"],
                "selected_concurrency": row["selected_concurrency"],
                "repetition_count": row["repetition_count"],
                "qps_mean": row["qps_mean"],
                "qps_ci95_lower": row["qps_ci95_lower"],
                "qps_ci95_upper": row["qps_ci95_upper"],
                "qps_cv": row["qps_cv"],
                "normalized_qps_to_m1": row.get("normalized_qps_to_m1", ""),
                "scaling_efficiency": row.get("scaling_efficiency", ""),
                "recall_mean": row["recall_mean"],
                "recall_min": row["recall_min"],
                "p95_latency_us": row["p95_latency_us"],
                "max_worker_cpu_utilization_pct": row[
                    "max_worker_cpu_utilization_pct"
                ],
                "max_network_utilization_pct": row["max_network_utilization_pct"],
                "source": row["source"],
                "source_sha256": row["source_sha256"],
            }
        )
    return rows


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def report_markdown(
    new_rows: Sequence[dict[str, Any]],
    baseline_b_rows: Sequence[dict[str, Any]],
    fanout_rows: Sequence[dict[str, Any]],
    partition_rows: Sequence[dict[str, Any]],
    cleanup: dict[str, Any],
    figures: Sequence[Path],
) -> str:
    lines = [
        "# Stage 13：1～4 台真实机器与固定 ef fan-out 重测",
        "",
        "状态：`MEASUREMENTS_COMPLETE`。SIFT1M 固定 `efSearch=24`；GloVe-200-angular 固定 `efSearch=192`；所有正式点 Recall@10≥0.90。",
        "",
        "## 真实物理机器 QPS",
        "",
        "| 数据集 | 方法 | M=1 | M=2 | M=3 | M=4 | M=4 / M=1 | M=4 扩展效率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in METHODS:
            rows = [row_for(new_rows, dataset, method, machines) for machines in PHYSICAL_COUNTS]
            lines.append(
                "| "
                + " | ".join(
                    [
                        DATASET_LABELS[dataset],
                        METHOD_LABELS[method],
                        *[fmt(row["qps_mean"]) for row in rows],
                        f"{rows[-1]['normalized_qps_to_m1']:.3f}×",
                        f"{rows[-1]['scaling_efficiency']:.3f}",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "M=1 是共享单分片 baseline-A。尾部 baseline-B 漂移：",
            "",
            "| 数据集 | baseline-A QPS | baseline-B QPS | 漂移 |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        a = row_for(new_rows, dataset, "random", 1)
        b = next(row for row in baseline_b_rows if row["dataset"] == dataset)
        drift = 100 * (b["qps_mean"] - a["qps_mean"]) / a["qps_mean"]
        lines.append(
            f"| {DATASET_LABELS[dataset]} | {fmt(a['qps_mean'])} | {fmt(b['qps_mean'])} | {drift:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 固定 ef 的 K-Means fan-out",
            "",
            "M=8、16、32 是四台物理机上的逻辑 shard 数，不是 8/16/32 台机器。",
            "",
            "| 数据集 | 固定 ef | M=1 | M=2 | M=3 | M=4 | M=8 | M=16 | M=32 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        rows = sorted(
            (row for row in fanout_rows if row["dataset"] == dataset),
            key=lambda row: row["logical_shard_count"],
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    DATASET_LABELS[dataset],
                    str(FIXED_EF[dataset]),
                    *[str(row["holdout_selected_fanout"]) for row in rows],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "GloVe M=16 的调优值 P=6 在 9,000 条留出查询上 Recall=0.8993，未通过；最终自动上调到 P=7，Recall=0.9115。",
            "",
            "## 结论",
            "",
            "1. 旧的物理 QPS 曲线不适合继续回答本次受控问题：它缺少 M=3，且不同点使用不同 ef。Stage 13 的固定 ef 曲线应作为新的主证据。",
            "2. SIFT Random 随 M 增加显著退化；GloVe 在 M=4 有吞吐提升，但只有单机的约 2.16×，仍非线性扩展。",
            "3. K-Means fan-out 并非普遍极低。GloVe 的最终序列为 1/2/3/3/5/7/8；SIFT 的低 fan-out 序列则在固定 ef=24 和独立留出召回门槛下仍成立。",
            "4. 当前基线漂移较小；M=3 K-Means 分区明显偏斜且未在 12 次迭代内收敛，解释 QPS 时必须把真实机器数与实际负载分布分开。",
            "",
            "## PDF 图表",
            "",
            "每张 PDF 页面下方均内嵌中文“用途、如何理解、最终结论、证据边界”。",
            "",
        ]
    )
    for figure in figures:
        lines.append(f"- [{figure.name}](figures/{figure.name})")
    nonconverged = [
        row
        for row in partition_rows
        if row["logical_shard_count"] == 3 and not row["kmeans_converged"]
    ]
    lines.extend(
        [
            "",
            "## 完整性与清理",
            "",
            f"- 清理证明：{cleanup['proof_count']}/{cleanup['proof_count']} 通过；每个 collection 在四个 peer 上均为 HTTP 404。",
            "- 所有记录的 controller collection 存储目录当前仍不存在。",
            f"- M=3 未收敛 K-Means 分区记录：{len(nonconverged)}/2（两数据集均保留为异常，而非隐藏）。",
            "- 原始逐查询 CSV、三次/五次重复摘要、调优候选和留出尝试均保留在 Stage 13 目录；未覆盖 Stage 12。",
            "",
        ]
    )
    return "\n".join(lines)


def verify_pdfs(figures: Sequence[Path]) -> None:
    required = ("用途：", "如何理解：", "最终结论：", "证据边界：")
    for path in figures:
        document = pymupdf.open(path)
        if document.page_count != 1:
            raise ValueError(f"expected one-page PDF: {path}")
        text = document[0].get_text()
        missing = [label for label in required if label not in text]
        if missing:
            raise ValueError(f"missing Chinese footer labels {missing}: {path}")
        document.close()


def summarize() -> dict[str, Any]:
    execution = load_json(STAGE13_DIR / "execution-complete.json")
    if execution.get("status") != "MEASUREMENTS_COMPLETE":
        raise ValueError("Stage 13 execution is not complete")
    new_rows, baseline_b_rows, old_rows = load_e1_rows()
    add_normalization(new_rows)
    add_normalization(old_rows)
    fanout_rows = load_fanout_rows()
    sensitivity_rows = load_sensitivity_rows()
    partition_rows = load_partition_rows()
    cleanup = audit_cleanup()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = [
        write_csv(
            SUMMARY_DIR / "physical-qps-old-vs-new.csv",
            physical_csv_rows(new_rows, baseline_b_rows, old_rows),
        ),
        write_csv(SUMMARY_DIR / "fanout-fixed-ef.csv", fanout_rows),
        write_csv(SUMMARY_DIR / "ef-sensitivity.csv", sensitivity_rows),
        write_csv(SUMMARY_DIR / "partition-balance.csv", partition_rows),
    ]
    drift_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        a = row_for(new_rows, dataset, "random", 1)
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
    csv_paths.append(write_csv(SUMMARY_DIR / "baseline-drift.csv", drift_rows))
    cleanup_path = write_json(SUMMARY_DIR / "cleanup-audit.json", cleanup)

    figures = [
        plot_physical_qps(new_rows, old_rows, baseline_b_rows),
        plot_normalized_scaling(new_rows, old_rows),
        plot_fanout(fanout_rows),
        plot_sensitivity(sensitivity_rows),
        plot_recall_stability(new_rows, baseline_b_rows),
        plot_baseline_partition(new_rows, baseline_b_rows, partition_rows),
        plot_latency_resources(new_rows),
    ]
    verify_pdfs(figures)
    report_path = atomic_write_text(
        SUMMARY_DIR / "RESULTS_zh.md",
        report_markdown(
            new_rows,
            baseline_b_rows,
            fanout_rows,
            partition_rows,
            cleanup,
            figures,
        ),
    )
    manifest = {
        "record_type": "stage13_retest_summary",
        "status": "PASS",
        "execution_complete": str((STAGE13_DIR / "execution-complete.json").resolve()),
        "execution_complete_sha256": sha256(STAGE13_DIR / "execution-complete.json"),
        "csv_files": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in csv_paths
        ],
        "figures": [
            {"path": str(path.resolve()), "sha256": sha256(path), "pages": 1}
            for path in figures
        ],
        "report": str(report_path.resolve()),
        "report_sha256": sha256(report_path),
        "cleanup_audit": str(cleanup_path.resolve()),
        "cleanup_audit_sha256": sha256(cleanup_path),
        "physical_configuration_count": len(new_rows),
        "baseline_b_count": len(baseline_b_rows),
        "fanout_configuration_count": len(fanout_rows),
        "sensitivity_point_count": len(sensitivity_rows),
        "cleanup_proof_count": cleanup["proof_count"],
    }
    manifest_path = write_json(SUMMARY_DIR / "manifest.json", manifest)
    manifest["manifest"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = sha256(manifest_path)
    return manifest


def check() -> dict[str, Any]:
    manifest_path = SUMMARY_DIR / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise ValueError("summary manifest is not PASS")
    for group in ("csv_files", "figures"):
        for entry in manifest[group]:
            path = Path(entry["path"])
            if not path.exists() or sha256(path) != entry["sha256"]:
                raise ValueError(f"manifest hash mismatch: {path}")
    report = Path(manifest["report"])
    if sha256(report) != manifest["report_sha256"]:
        raise ValueError(f"report hash mismatch: {report}")
    cleanup = Path(manifest["cleanup_audit"])
    if sha256(cleanup) != manifest["cleanup_audit_sha256"]:
        raise ValueError(f"cleanup hash mismatch: {cleanup}")
    figures = [Path(entry["path"]) for entry in manifest["figures"]]
    verify_pdfs(figures)
    return {
        "status": "PASS",
        "manifest": str(manifest_path.resolve()),
        "figure_count": len(figures),
        "csv_count": len(manifest["csv_files"]),
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
