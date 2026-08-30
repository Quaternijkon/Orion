#!/usr/bin/env python3
"""Rank swap_24_21 neighbors with interaction-aware paired-QPS kernels."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import orion_physical_placement_local_refine as local
import orion_physical_placement_neighbor_refine as neighbor
import orion_physical_placement_refine as refine


DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_LAYOUT = Path(
    "/users/dry/orion-distributed/native-20260818-scale4-32-r080-095-v1/"
    "artifacts/scale32/orion-r090-u48-b50-f14-g3248141"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v17-kernel-qps"
BASELINE = "swap_24_21"
KERNELS = ("hamming_rbf", "work_weighted_rbf", "colocation_rbf")
GAMMA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
ALPHA_GRID = np.concatenate(
    (np.asarray([0.0]), np.logspace(-8.0, 2.0, num=50, dtype=np.float64))
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mapping_key(mapping: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(mapping.items()))


def gather_pairs(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[tuple[int, int], ...], dict[int, int]]]:
    pairs = []
    mappings: dict[tuple[tuple[int, int], ...], dict[int, int]] = {}
    for version in range(2, 17):
        output = root / f"online-v{version}"
        summaries = sorted(output.glob("*summary.json"))
        if len(summaries) != 1:
            raise RuntimeError(f"expected one summary in {output}, found {summaries}")
        summary = load_json(summaries[0])
        phases = []
        for row in summary["phases"]:
            path = output / "phases" / row["phase_id"] / "screen.json"
            payload = load_json(path)
            mapping = {
                int(shard): int(peer)
                for shard, peer in payload["ending_placement"]["placement"].items()
            }
            key = mapping_key(mapping)
            mappings[key] = mapping
            phases.append(
                {
                    "phase_id": row["phase_id"],
                    "strategy": row["strategy"],
                    "qps": float(row["qps_mean"]),
                    "mapping": mapping,
                    "key": key,
                    "path": path,
                }
            )
        baseline_strategy = phases[0]["strategy"]
        baseline_rows = [row for row in phases if row["strategy"] == baseline_strategy]
        if len(baseline_rows) != 2:
            raise RuntimeError(f"online-v{version}: expected two baseline phases")
        baseline_qps = float(np.mean([row["qps"] for row in baseline_rows]))
        baseline_mapping = baseline_rows[0]["mapping"]
        baseline_key = baseline_rows[0]["key"]
        for row in phases:
            if row["strategy"] == baseline_strategy:
                continue
            pairs.append(
                {
                    "online_version": version,
                    "phase_id": row["phase_id"],
                    "strategy": row["strategy"],
                    "baseline_strategy": baseline_strategy,
                    "qps": row["qps"],
                    "baseline_qps_mean": baseline_qps,
                    "log_qps_ratio": math.log(row["qps"] / baseline_qps),
                    "candidate_mapping": row["mapping"],
                    "candidate_key": row["key"],
                    "baseline_mapping": baseline_mapping,
                    "baseline_key": baseline_key,
                    "source": {"path": str(row["path"]), "sha256": local.sha256_path(row["path"])},
                }
            )
    if len(pairs) != 52 or len(mappings) != 38:
        raise RuntimeError(
            f"paired input mismatch: pairs={len(pairs)}, placements={len(mappings)}"
        )
    return pairs, mappings


def all_online_mappings(root: Path) -> set[tuple[tuple[int, int], ...]]:
    paths = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in range(2, 17):
        paths.extend((root / f"online-v{version}/phases").glob("*/screen.json"))
    result = set()
    for path in paths:
        payload = load_json(path)
        result.add(
            tuple(
                sorted(
                    (int(shard), int(peer))
                    for shard, peer in payload["ending_placement"]["placement"].items()
                )
            )
        )
    if len(result) != 40:
        raise RuntimeError(f"expected 40 total online placements, got {len(result)}")
    return result


def distance(
    kernel: str,
    left: dict[int, int],
    right: dict[int, int],
    work_by_shard: Sequence[float],
) -> float:
    left_owner = np.asarray([left[shard] for shard in range(32)])
    right_owner = np.asarray([right[shard] for shard in range(32)])
    changed = left_owner != right_owner
    if kernel == "hamming_rbf":
        return float(np.mean(changed))
    if kernel == "work_weighted_rbf":
        weights = np.asarray(work_by_shard, dtype=np.float64)
        return float(np.sum(weights[changed]) / np.sum(weights))
    if kernel == "colocation_rbf":
        left_same = left_owner[:, None] == left_owner[None, :]
        right_same = right_owner[:, None] == right_owner[None, :]
        upper = np.triu_indices(32, k=1)
        return float(np.mean(left_same[upper] != right_same[upper]))
    raise ValueError(f"unknown kernel: {kernel}")


def kernel_value(
    kernel: str,
    gamma: float,
    left: dict[int, int],
    right: dict[int, int],
    work_by_shard: Sequence[float],
) -> float:
    return math.exp(-gamma * distance(kernel, left, right, work_by_shard))


def pair_design(
    pairs: Sequence[dict[str, Any]],
    basis: Sequence[dict[int, int]],
    kernel: str,
    gamma: float,
    work_by_shard: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    target = []
    for pair in pairs:
        candidate = np.asarray(
            [
                kernel_value(
                    kernel,
                    gamma,
                    pair["candidate_mapping"],
                    mapping,
                    work_by_shard,
                )
                for mapping in basis
            ],
            dtype=np.float64,
        )
        baseline = np.asarray(
            [
                kernel_value(
                    kernel,
                    gamma,
                    pair["baseline_mapping"],
                    mapping,
                    work_by_shard,
                )
                for mapping in basis
            ],
            dtype=np.float64,
        )
        rows.append(candidate - baseline)
        target.append(float(pair["log_qps_ratio"]))
    return np.asarray(rows, dtype=np.float64), np.asarray(target, dtype=np.float64)


def fit_ridge(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    if design.shape[0] < design.shape[1]:
        return design.T @ np.linalg.pinv(
            design @ design.T + alpha * np.eye(design.shape[0])
        ) @ target
    return np.linalg.pinv(
        design.T @ design + alpha * np.eye(design.shape[1])
    ) @ design.T @ target


def fit_kernel_model(
    kernel: str,
    pairs: Sequence[dict[str, Any]],
    mappings: dict[tuple[tuple[int, int], ...], dict[int, int]],
    work_by_shard: Sequence[float],
) -> dict[str, Any]:
    candidate_keys = sorted({pair["candidate_key"] for pair in pairs})
    cv_rows = []
    for gamma in GAMMA_GRID:
        folds = []
        for held_out in candidate_keys:
            training = [
                pair
                for pair in pairs
                if pair["candidate_key"] != held_out and pair["baseline_key"] != held_out
            ]
            test = [
                pair
                for pair in pairs
                if pair["candidate_key"] == held_out and pair["baseline_key"] != held_out
            ]
            if not training or not test:
                continue
            basis_keys = sorted(
                {pair["candidate_key"] for pair in training}
                | {pair["baseline_key"] for pair in training}
            )
            basis = [mappings[key] for key in basis_keys]
            train_x, train_y = pair_design(
                training, basis, kernel, gamma, work_by_shard
            )
            test_x, test_y = pair_design(test, basis, kernel, gamma, work_by_shard)
            folds.append((train_x, train_y, test_x, test_y))
        for alpha in ALPHA_GRID:
            predicted = []
            observed = []
            for train_x, train_y, test_x, test_y in folds:
                coefficients = fit_ridge(train_x, train_y, float(alpha))
                predicted.extend((test_x @ coefficients).tolist())
                observed.extend(test_y.tolist())
            pred = np.asarray(predicted, dtype=np.float64)
            obs = np.asarray(observed, dtype=np.float64)
            errors = pred - obs
            nonzero = np.abs(obs) > 1e-12
            sign_accuracy = float(
                np.mean(np.sign(pred[nonzero]) == np.sign(obs[nonzero]))
            )
            cv_rows.append(
                {
                    "gamma": gamma,
                    "alpha": float(alpha),
                    "rmse_log_ratio": float(math.sqrt(np.mean(np.square(errors)))),
                    "rmse_approx_pct": float(100.0 * math.sqrt(np.mean(np.square(errors)))),
                    "sign_accuracy": sign_accuracy,
                    "error_count": len(errors),
                }
            )
    selected = min(
        cv_rows,
        key=lambda row: (
            row["rmse_log_ratio"],
            -row["sign_accuracy"],
            row["gamma"],
            row["alpha"],
        ),
    )
    basis_keys = sorted(mappings)
    basis = [mappings[key] for key in basis_keys]
    design, target = pair_design(
        pairs, basis, kernel, selected["gamma"], work_by_shard
    )
    coefficients = fit_ridge(design, target, selected["alpha"])
    fitted = design @ coefficients
    residual = target - fitted
    return {
        "name": kernel,
        "claim_boundary": "interaction-aware paired-QPS candidate selector only",
        "target": "log(candidate QPS / same-run baseline A/B mean QPS)",
        "pair_observation_count": len(pairs),
        "basis_placement_count": len(basis),
        "gamma": selected["gamma"],
        "ridge_alpha": selected["alpha"],
        "leave_one_candidate_placement_out_rmse_log_ratio": selected["rmse_log_ratio"],
        "leave_one_candidate_placement_out_rmse_approx_pct": selected["rmse_approx_pct"],
        "leave_one_candidate_placement_out_sign_accuracy": selected["sign_accuracy"],
        "cross_validation_top_five": sorted(
            cv_rows,
            key=lambda row: (
                row["rmse_log_ratio"],
                -row["sign_accuracy"],
                row["gamma"],
                row["alpha"],
            ),
        )[:5],
        "basis_keys": [[list(item) for item in key] for key in basis_keys],
        "coefficients": [float(value) for value in coefficients],
        "fit_rmse_log_ratio": float(math.sqrt(np.mean(np.square(residual)))),
        "observed": [float(value) for value in target],
        "fitted": [float(value) for value in fitted],
        "residual": [float(value) for value in residual],
    }


def predict_improvement(
    model: dict[str, Any],
    candidate: dict[int, int],
    baseline: dict[int, int],
    mappings: dict[tuple[tuple[int, int], ...], dict[int, int]],
    work_by_shard: Sequence[float],
) -> float:
    basis_keys = [
        tuple((int(shard), int(peer)) for shard, peer in key)
        for key in model["basis_keys"]
    ]
    basis = [mappings[key] for key in basis_keys]
    gamma = float(model["gamma"])
    kernel = str(model["name"])
    candidate_values = np.asarray(
        [kernel_value(kernel, gamma, candidate, mapping, work_by_shard) for mapping in basis]
    )
    baseline_values = np.asarray(
        [kernel_value(kernel, gamma, baseline, mapping, work_by_shard) for mapping in basis]
    )
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    log_ratio = float((candidate_values - baseline_values) @ coefficients)
    return 100.0 * (math.exp(log_ratio) - 1.0)


def build(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    source_plan_path = root / "confirm-v4-plan/placements.json"
    source_audit_path = root / "online-v16/evidence-audit.json"
    trace_path = root / "trace/p24-per-query.json"
    layout_manifest_path = layout_dir / "build-manifest.json"
    source_plan = load_json(source_plan_path)
    source_audit = load_json(source_audit_path)
    trace = load_json(trace_path)
    layout_manifest = load_json(layout_manifest_path)
    if source_audit.get("status") != "PASS":
        raise RuntimeError("latest evidence audit is not PASS")
    peer_order = [int(value) for value in source_plan["peer_order"]]
    baseline = local.normalize_mapping(source_plan["placements"][BASELINE])
    counts = [int(value) for value in layout_manifest["routing"]["shard_counts"]]
    work_by_shard = [1000.0 * row["work_1000"] for row in refine.shard_features(trace, counts)]
    pairs, mappings = gather_pairs(root)
    seen = all_online_mappings(root)
    models = [
        fit_kernel_model(kernel, pairs, mappings, work_by_shard)
        for kernel in KERNELS
    ]
    rows = []
    for left_shard, right_shard, mapping in neighbor.enumerate_neighbors(baseline, peer_order):
        predictions = {
            model["name"]: predict_improvement(
                model, mapping, baseline, mappings, work_by_shard
            )
            for model in models
        }
        rows.append(
            {
                "left_shard": left_shard,
                "right_shard": right_shard,
                "already_online_tested": mapping_key(mapping) in seen,
                "predicted_improvement_pct": predictions,
                "predicted_improvement_mean_pct": float(np.mean(list(predictions.values()))),
                "predicted_improvement_min_pct": min(predictions.values()),
                "moved_physical_vector_copies": counts[left_shard] + counts[right_shard],
            }
        )
    for model in models:
        name = model["name"]
        ranked = sorted(
            rows,
            key=lambda row: (
                -row["predicted_improvement_pct"][name],
                row["left_shard"],
                row["right_shard"],
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row.setdefault("model_ranks", {})[name] = rank
    for row in rows:
        ranks = list(row["model_ranks"].values())
        row["model_worst_rank"] = max(ranks)
        row["model_rank_sum"] = sum(ranks)
    ranked = sorted(
        rows,
        key=lambda row: (
            row["model_worst_rank"],
            row["model_rank_sum"],
            -row["predicted_improvement_min_pct"],
            row["moved_physical_vector_copies"],
            row["left_shard"],
            row["right_shard"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["consensus_rank"] = rank
    selected = [row for row in ranked if not row["already_online_tested"]][:4]
    if len(selected) != 4:
        raise RuntimeError("could not select four kernel-QPS candidates")

    placements = {BASELINE: baseline}
    metrics: dict[str, Any] = {
        BASELINE: {
            "groups": local.placement_groups(baseline, peer_order),
            "selection_reason": "formally confirmed current winner and A/B control",
        }
    }
    names = []
    for row in selected:
        name = f"kernel_neighbor_{row['left_shard']}_{row['right_shard']}"
        names.append(name)
        mapping = local.swap_mapping(baseline, row["left_shard"], row["right_shard"])
        placements[name] = mapping
        metrics[name] = {
            **row,
            "groups": local.placement_groups(mapping, peer_order),
            "selection_reason": "top untested consensus across three interaction-aware paired-QPS kernels",
        }
    phases = [("swap-24-21-kernel-a", BASELINE)] + [
        (name.replace("_", "-"), name) for name in names
    ] + [("swap-24-21-kernel-b", BASELINE)]
    placements_path = output / "placements.json"
    local.write_json_new(
        placements_path,
        {
            "format_version": 1,
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "controller_peer_id": source_plan["controller_peer_id"],
            "peer_order": peer_order,
            "peer_hosts": source_plan["peer_hosts"],
            "screen_phases": [
                {"phase_id": phase_id, "strategy": strategy}
                for phase_id, strategy in phases
            ],
            "placements": {
                name: {str(shard): peer for shard, peer in sorted(mapping.items())}
                for name, mapping in placements.items()
            },
        },
    )
    pairs_path = output / "paired-qps-observations.json"
    local.write_json_new(
        pairs_path,
        {
            "pair_count": len(pairs),
            "unique_paired_placement_count": len(mappings),
            "pairs": pairs,
        },
    )
    models_path = output / "kernel-qps-models.json"
    local.write_json_new(models_path, {"models": models})
    ranking_path = output / "one-swap-kernel-qps-ranking.json"
    local.write_json_new(
        ranking_path,
        {
            "baseline": BASELINE,
            "evaluated_swaps": len(ranked),
            "ranking_rule": [
                "minimize worst rank across Hamming, work-weighted, and co-location kernels",
                "then rank sum",
                "then maximize minimum predicted improvement",
            ],
            "neighbors": ranked,
        },
    )
    metrics_path = output / "candidate-metrics.json"
    local.write_json_new(metrics_path, metrics)
    rationale_path = output / "RATIONALE_zh.md"
    lines = [
        "# interaction-aware paired-QPS kernel 候选",
        "",
        f"纳入 online-v2…v16 的 {len(pairs)} 个 paired-QPS 观测、{len(mappings)} 个 paired placement。",
        "三种核分别使用整体 Hamming、work-weighted Hamming 和 shard 共置关系距离。",
        "",
        "| 候选 | swap | consensus | predicted mean/min | model worst/sum | copies |",
        "|---|---|---:|---:|---:|---:|",
        *[
            (
                f"| `{name}` | {row['left_shard']}↔{row['right_shard']} | "
                f"{row['consensus_rank']} | {row['predicted_improvement_mean_pct']:+.2f}% / "
                f"{row['predicted_improvement_min_pct']:+.2f}% | "
                f"{row['model_worst_rank']}/{row['model_rank_sum']} | "
                f"{row['moved_physical_vector_copies']} |"
            )
            for name, row in metrics.items()
            if name != BASELINE
        ],
        "",
        "模型 LOPO RMSE / sign accuracy：",
        *[
            (
                f"- {model['name']}: {model['leave_one_candidate_placement_out_rmse_approx_pct']:.2f}% / "
                f"{100.0 * model['leave_one_candidate_placement_out_sign_accuracy']:.1f}%"
            )
            for model in models
        ],
        "",
    ]
    with rationale_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    outputs = (placements_path, pairs_path, models_path, ranking_path, metrics_path, rationale_path)
    manifest_path = output / "manifest.json"
    local.write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": "interaction-aware paired-QPS kernel plan; online evidence remains authoritative",
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "pair_observation_count": len(pairs),
            "unique_paired_placement_count": len(mappings),
            "total_online_placement_count": len(seen),
            "one_swap_neighbors_evaluated": len(ranked),
            "selected_candidate_count": len(names),
            "selected_strategies": names,
            "inputs": {
                "source_plan": {"path": str(source_plan_path), "sha256": local.sha256_path(source_plan_path)},
                "source_audit": {"path": str(source_audit_path), "sha256": local.sha256_path(source_audit_path)},
                "layout_manifest": {"path": str(layout_manifest_path), "sha256": local.sha256_path(layout_manifest_path)},
            },
            "outputs": {
                path.name: {"path": str(path), "sha256": local.sha256_path(path)}
                for path in outputs
            },
        },
    )
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build(parse_args(argv))
    print(json.dumps({"manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
