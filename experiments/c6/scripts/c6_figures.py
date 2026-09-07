#!/usr/bin/env python3
"""Generate the complete C6 PDF figure set from recorded run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


POLICY_ORDER = ("P0", "P1", "P2", "P3", "P6")
POLICY_LABELS = {
    "P0": "Static",
    "P1": "Adaptive shards",
    "P2": "Adaptive EF",
    "P3": "Full Orion",
    "P4": "Oracle prefix",
    "P5": "Oracle local",
    "P6": "Full oracle",
}
DATASET_LABELS = {"sift1m": "SIFT1M", "glove-200-angular": "GloVe"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--physical-summary")
    parser.add_argument("--measurement", action="append", default=[])
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(payload: dict[str, Any]) -> tuple[str, int]:
    return str(payload["dataset"]), int(payload["logical_shards"])


def label(dataset: str, logical_shards: int) -> str:
    return f"{DATASET_LABELS.get(dataset, dataset)} M={logical_shards}"


def sorted_payloads(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(values, key=key)


def cdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, len(x) + 1, dtype=float) / len(x)


def save(fig: Any, directory: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(directory / name, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(
    ax: Any,
    groups: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
    *,
    ylabel: str,
) -> None:
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / max(len(series), 1)
    for index, (name, values) in enumerate(series):
        ax.bar(x - 0.4 + width / 2 + index * width, values, width, label=name)
    ax.set_xticks(x, groups, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)


def per_query_delta(path0: Path, path1: Path, field: str) -> list[float]:
    left = read_csv(path0)
    right = read_csv(path1)
    if len(left) != len(right):
        raise ValueError(f"unaligned paired CSVs: {path0} and {path1}")
    return [
        float(a[field]) - float(b[field])
        for a, b in zip(left, right, strict=True)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = Path(args.runs_dir).expanduser().resolve()
    figures = Path(args.figures_dir).expanduser().resolve()
    figures.mkdir(parents=True, exist_ok=True)
    isolated_paths = sorted(runs.glob("*-isolated-e2-e3-summary.json"))
    measurement_paths = (
        [Path(path).expanduser().resolve() for path in args.measurement]
        if args.measurement
        else sorted(runs.glob("*-measurement-completion.json"))
    )
    e1_paths = sorted((runs / "e1").glob("*-e1-summary.json"))
    if not isolated_paths or not measurement_paths or not e1_paths:
        raise RuntimeError("C6 figures require isolated, measurement, and E1 artifacts")
    isolated_loaded = [(load_json(path), path) for path in isolated_paths]
    measurement_loaded = [(load_json(path), path) for path in measurement_paths]
    e1_loaded = [(load_json(path), path) for path in e1_paths]
    isolated = sorted_payloads(payload for payload, _path in isolated_loaded)
    measurements = sorted_payloads(payload for payload, _path in measurement_loaded)
    e1 = sorted_payloads(payload for payload, _path in e1_loaded)
    isolated_path_by_key = {key(payload): path for payload, path in isolated_loaded}
    e1_path_by_key = {key(payload): path for payload, path in e1_loaded}
    measurement_by_key = {key(row): row for row in measurements}

    # Figure 1: oracle shard difficulty CDF.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for payload in e1:
        path = e1_path_by_key[key(payload)]
        rows = read_csv(path.with_name(path.name.replace("-summary.json", "-p4.csv")))
        values = [len(json.loads(row["selected_shards"])) for row in rows]
        x, y = cdf(values)
        ax.step(x, y, where="post", label=label(*key(payload)))
    ax.set(xlabel="Oracle prefix shards", ylabel="Fraction of queries", ylim=(0, 1.01))
    ax.legend(fontsize=7, ncol=2)
    save(fig, figures, "c6_fig1_oracle_shard_difficulty_cdf.pdf")

    # Figure 2: oracle local EF CDF.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for payload in e1:
        path = e1_path_by_key[key(payload)]
        rows = read_csv(path.with_name(path.name.replace("-summary.json", "-p6-per-shard.csv")))
        x, y = cdf([float(row["oracle_ef_search"]) for row in rows])
        ax.step(x, y, where="post", label=label(*key(payload)))
    ax.set_xscale("log", base=2)
    ax.set(xlabel="Oracle local efSearch", ylabel="Fraction of query-shard pairs", ylim=(0, 1.01))
    ax.legend(fontsize=7, ncol=2)
    save(fig, figures, "c6_fig2_oracle_local_budget_cdf.pdf")

    group_labels = [label(*key(row)) for row in measurements]
    # Figure 3: static waste.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    grouped_bars(
        ax,
        group_labels,
        [
            ("Under", [row["static_waste"]["under_searched"] for row in measurements]),
            ("Near", [row["static_waste"]["near_sufficient"] for row in measurements]),
            ("Over", [row["static_waste"]["over_searched"] for row in measurements]),
        ],
        ylabel="Fraction of queries",
    )
    save(fig, figures, "c6_fig3_static_waste_breakdown.pdf")

    isolated_labels = [label(*key(row)) for row in isolated]
    # Figures 4 and 5: fan-out comparisons.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    grouped_bars(
        ax,
        isolated_labels,
        [
            ("P0", [row["E2"]["P0"]["mean_shards_per_query"] for row in isolated]),
            ("P1", [row["E2"]["P1"]["mean_shards_per_query"] for row in isolated]),
        ],
        ylabel="Mean shards/query",
    )
    save(fig, figures, "c6_fig4_static_vs_adaptive_fanout.pdf")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    grouped_bars(
        ax,
        isolated_labels,
        [
            ("P1", [row["E2"]["P1"]["mean_shards_per_query"] for row in isolated]),
            ("P4", [row["E2"]["P4"]["mean_shards_per_query"] for row in isolated]),
        ],
        ylabel="Mean shards/query",
    )
    save(fig, figures, "c6_fig5_adaptive_vs_oracle_fanout.pdf")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for payload in isolated:
        path = isolated_path_by_key[key(payload)]
        prefix = path.name.removesuffix("-e2-e3-summary.json")
        values = per_query_delta(
            runs / f"{prefix}-e2-p0.csv",
            runs / f"{prefix}-e2-p1.csv",
            "selected_shard_count",
        )
        x, y = cdf(values)
        ax.step(x, y, where="post", label=label(*key(payload)))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Static minus adaptive shards/query", ylabel="Fraction of queries", ylim=(0, 1.01))
    ax.legend(fontsize=7, ncol=2)
    save(fig, figures, "c6_fig6_per_query_shard_savings_cdf.pdf")

    def frontier(ax: Any, section: str, policies: Sequence[str]) -> None:
        markers = {"P0": "o", "P1": "s", "P2": "^", "P4": "D", "P5": "X"}
        for payload in isolated:
            for policy in policies:
                row = payload[section][policy]
                ax.scatter(
                    row["aggregate_distance_computations_per_query"],
                    row["recall_at_10"],
                    marker=markers.get(policy, "o"),
                    s=24,
                    label=f"{label(*key(payload))} {policy}",
                )
        ax.axhline(0.9, color="black", linestyle="--", linewidth=0.8)
        ax.set(xlabel="Aggregate distance computations/query", ylabel="Recall@10")

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    frontier(ax, "E2", ("P0", "P1", "P4"))
    ax.legend(fontsize=5.5, ncol=3)
    save(fig, figures, "c6_fig7_recall_work_frontier_shard_adaptation.pdf")

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    frontier(ax, "E3", ("P0", "P2"))
    ax.legend(fontsize=6, ncol=2)
    save(fig, figures, "c6_fig8_uniform_vs_adaptive_ef.pdf")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    grouped_bars(
        ax,
        isolated_labels,
        [
            (policy, [row["E3"][policy]["aggregate_distance_computations_per_query"] for row in isolated])
            for policy in ("P0", "P2", "P5")
        ],
        ylabel="Aggregate distance computations/query",
    )
    save(fig, figures, "c6_fig9_local_budget_vs_oracle.pdf")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    grouped_bars(
        ax,
        isolated_labels,
        [
            (policy, [row["E3"][policy]["p95_max_shard_distance_computations"] for row in isolated])
            for policy in ("P0", "P2", "P5")
        ],
        ylabel="P95 max-shard distance computations",
    )
    save(fig, figures, "c6_fig10_max_shard_work.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    ep_groups = ("1", "2", "3-4", "5-8", ">8")
    for payload in isolated:
        groups = {row["entry_point_group"]: row for row in payload["E3"]["entry_point_calibration"]["groups"]}
        axes[0].plot(ep_groups, [groups[name]["probability_contributes_ground_truth"] for name in ep_groups], marker="o", label=label(*key(payload)))
        axes[1].plot(ep_groups, [groups[name]["mean_oracle_local_distance_computations"] for name in ep_groups], marker="o", label=label(*key(payload)))
    axes[0].set(xlabel="Entry-point count group", ylabel="P(contributes ground truth)")
    axes[1].set(xlabel="Entry-point count group", ylabel="Mean oracle local work")
    axes[1].legend(fontsize=6, ncol=2)
    save(fig, figures, "c6_fig11_entry_point_count_calibration.pdf")

    physical = read_csv(Path(args.physical_summary)) if args.physical_summary else []
    physical_labels = [f"{DATASET_LABELS.get(row['dataset'], row['dataset'])} {row['policy']}" for row in physical]

    def physical_bar(name: str, field: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        colors = ["tab:blue" if row.get("physical_run_valid", "false").lower() == "true" else "tab:red" for row in physical]
        ax.bar(np.arange(len(physical)), [float(row[field]) for row in physical], color=colors)
        ax.set_xticks(np.arange(len(physical)), physical_labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        save(fig, figures, name)

    physical_bar("c6_fig12_physical_tail_latency.pdf", "p99_latency_us", "P99 latency (us)")

    # Full ablation and oracle gap figures.
    summary_maps = [{row["policy"]: row for row in payload["summaries"]} for payload in measurements]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    grouped_bars(
        ax,
        group_labels,
        [
            (
                policy,
                [rows[policy]["aggregate_distance_computations_per_query"] / rows["P0"]["aggregate_distance_computations_per_query"] for rows in summary_maps],
            )
            for policy in POLICY_ORDER
        ],
        ylabel="Normalized work (P0=1)",
    )
    ax.axhline(1.0, color="black", linewidth=0.8)
    save(fig, figures, "c6_fig13_full_ablation.pdf")

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for payload, rows in zip(measurements, summary_maps, strict=True):
        for policy in POLICY_ORDER:
            row = rows[policy]
            ax.scatter(row["aggregate_distance_computations_per_query"], row["recall_at_10"], s=26, label=f"{label(*key(payload))} {policy}")
    ax.axhline(0.9, color="black", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Aggregate distance computations/query", ylabel="Recall@10")
    ax.legend(fontsize=5.2, ncol=3)
    save(fig, figures, "c6_fig14_full_recall_work_frontier.pdf")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    grouped_bars(
        ax,
        group_labels,
        [("P3/P6", [rows["P3"]["aggregate_distance_computations_per_query"] / rows["P6"]["aggregate_distance_computations_per_query"] for rows in summary_maps])],
        ylabel="Work ratio",
    )
    ax.axhline(1.0, color="black", linewidth=0.8)
    save(fig, figures, "c6_fig15_oracle_gap.pdf")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for row in physical:
        ax.scatter(float(row["p99_latency_us"]), float(row["qps_mean"]), label=f"{row['dataset']} {row['policy']}")
    ax.set(xlabel="P99 latency (us)", ylabel="QPS")
    ax.legend(fontsize=7)
    save(fig, figures, "c6_fig16_physical_end_to_end.pdf")
    physical_bar("c6_fig17_physical_qps.pdf", "qps_mean", "QPS")
    physical_bar("c6_fig18_physical_p99_latency.pdf", "p99_latency_us", "P99 latency (us)")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for row in physical:
        ax.scatter(float(row["aggregate_distance_computations_per_query"]), float(row["p99_latency_us"]), label=f"{row['dataset']} {row['policy']}")
    ax.set(xlabel="Aggregate distance computations/query", ylabel="P99 latency (us)")
    ax.legend(fontsize=7)
    save(fig, figures, "c6_fig19_physical_work_vs_latency.pdf")

    # Recommended four-panel combined figure.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for payload in e1:
        path = e1_path_by_key[key(payload)]
        rows = read_csv(path.with_name(path.name.replace("-summary.json", "-p4.csv")))
        x, y = cdf([len(json.loads(row["selected_shards"])) for row in rows])
        axes[0, 0].step(x, y, where="post", label=label(*key(payload)))
    axes[0, 0].set(xlabel="Oracle prefix shards", ylabel="CDF")
    axes[0, 0].legend(fontsize=6, ncol=2)
    grouped_bars(
        axes[0, 1],
        isolated_labels,
        [("P0", [row["E2"]["P0"]["mean_shards_per_query"] for row in isolated]), ("P1", [row["E2"]["P1"]["mean_shards_per_query"] for row in isolated])],
        ylabel="Mean shards/query",
    )
    grouped_bars(
        axes[1, 0],
        isolated_labels,
        [("P0", [row["E3"]["P0"]["aggregate_distance_computations_per_query"] for row in isolated]), ("P2", [row["E3"]["P2"]["aggregate_distance_computations_per_query"] for row in isolated])],
        ylabel="Local-budget work",
    )
    grouped_bars(
        axes[1, 1],
        group_labels,
        [(policy, [rows[policy]["aggregate_distance_computations_per_query"] / rows["P0"]["aggregate_distance_computations_per_query"] for rows in summary_maps]) for policy in POLICY_ORDER],
        ylabel="Normalized work",
    )
    save(fig, figures, "c6_combined_adaptivity.pdf")

    produced = sorted(path.name for path in figures.glob("c6_*.pdf"))
    print(json.dumps({"figure_count": len(produced), "figures": produced}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
