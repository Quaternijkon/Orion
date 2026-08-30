#!/usr/bin/env python3
"""Rank swap_24_21 neighbors with a direct paired-online-QPS surrogate."""

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
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v16-qps-surrogate"
BASELINE = "swap_24_21"
ALPHA_GRID = np.concatenate(
    (np.asarray([0.0]), np.logspace(-8.0, 6.0, num=180, dtype=np.float64))
)
MODEL_SPECS = (
    ("assignment_qps_ridge", False, False),
    ("assignment_loadshape_qps_ridge", True, False),
    ("standardized_assignment_loadshape_qps_ridge", True, True),
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mapping_key(mapping: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(mapping.items()))


def placement_vector(
    mapping: dict[int, int],
    peer_order: Sequence[int],
    work_by_shard: Sequence[float],
    controller_overhead: float,
    *,
    include_load_shape: bool,
) -> np.ndarray:
    features = []
    for shard in range(32):
        owner = mapping[shard]
        features.extend(1.0 if owner == peer else 0.0 for peer in peer_order)
    if include_load_shape:
        loads = np.asarray(
            [
                sum(
                    work_by_shard[shard]
                    for shard in range(32)
                    if mapping[shard] == peer
                )
                + (controller_overhead if index == 0 else 0.0)
                for index, peer in enumerate(peer_order)
            ],
            dtype=np.float64,
        )
        scale = max(float(np.mean(loads)), 1.0)
        normalized = loads / scale
        features.extend(normalized.tolist())
        features.extend(np.square(normalized).tolist())
        features.extend([float(np.max(normalized)), float(np.var(normalized))])
    return np.asarray(features, dtype=np.float64)


def gather_pairs(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[tuple[int, int], ...], dict[int, int]]]:
    pairs = []
    mappings: dict[tuple[tuple[int, int], ...], dict[int, int]] = {}
    for version in range(2, 16):
        output = root / f"online-v{version}"
        summaries = sorted(output.glob("*summary.json"))
        if len(summaries) != 1:
            raise RuntimeError(f"expected one summary in {output}, found {summaries}")
        summary = load_json(summaries[0])
        phase_rows = []
        for row in summary["phases"]:
            path = output / "phases" / row["phase_id"] / "screen.json"
            payload = load_json(path)
            mapping = {
                int(shard): int(peer)
                for shard, peer in payload["ending_placement"]["placement"].items()
            }
            key = mapping_key(mapping)
            mappings[key] = mapping
            phase_rows.append(
                {
                    "phase_id": row["phase_id"],
                    "strategy": row["strategy"],
                    "qps": float(row["qps_mean"]),
                    "mapping": mapping,
                    "mapping_key": key,
                    "path": path,
                }
            )
        baseline_strategy = phase_rows[0]["strategy"]
        baseline_rows = [row for row in phase_rows if row["strategy"] == baseline_strategy]
        if len(baseline_rows) != 2:
            raise RuntimeError(f"online-v{version}: expected two baseline phases")
        baseline_qps = float(np.mean([row["qps"] for row in baseline_rows]))
        baseline_mapping = baseline_rows[0]["mapping"]
        baseline_key = baseline_rows[0]["mapping_key"]
        for row in phase_rows:
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
                    "candidate_key": row["mapping_key"],
                    "baseline_mapping": baseline_mapping,
                    "baseline_key": baseline_key,
                    "source": {"path": str(row["path"]), "sha256": local.sha256_path(row["path"])},
                }
            )
    if len(mappings) != 34:
        raise RuntimeError(f"expected 34 paired-screen placements, got {len(mappings)}")
    return pairs, mappings


def design_matrix(
    pairs: Sequence[dict[str, Any]],
    peer_order: Sequence[int],
    work_by_shard: Sequence[float],
    controller_overhead: float,
    *,
    include_load_shape: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    for pair in pairs:
        candidate = placement_vector(
            pair["candidate_mapping"],
            peer_order,
            work_by_shard,
            controller_overhead,
            include_load_shape=include_load_shape,
        )
        baseline = placement_vector(
            pair["baseline_mapping"],
            peer_order,
            work_by_shard,
            controller_overhead,
            include_load_shape=include_load_shape,
        )
        rows.append(candidate - baseline)
        targets.append(float(pair["log_qps_ratio"]))
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def fit_arrays(
    design: np.ndarray, target: np.ndarray, alpha: float, *, standardize: bool
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.std(design, axis=0) if standardize else np.ones(design.shape[1])
    scale = np.where(scale < 1e-12, 1.0, scale)
    scaled = design / scale
    if scaled.shape[0] < scaled.shape[1]:
        coefficients_scaled = scaled.T @ np.linalg.pinv(
            scaled @ scaled.T + alpha * np.eye(scaled.shape[0])
        ) @ target
    else:
        coefficients_scaled = np.linalg.pinv(
            scaled.T @ scaled + alpha * np.eye(scaled.shape[1])
        ) @ scaled.T @ target
    return coefficients_scaled / scale, scale


def fit_model(
    name: str,
    include_load_shape: bool,
    standardize: bool,
    pairs: Sequence[dict[str, Any]],
    peer_order: Sequence[int],
    work_by_shard: Sequence[float],
    controller_overhead: float,
) -> dict[str, Any]:
    candidate_keys = sorted({pair["candidate_key"] for pair in pairs})
    cv_rows = []
    for alpha in ALPHA_GRID:
        errors = []
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
            train_x, train_y = design_matrix(
                training,
                peer_order,
                work_by_shard,
                controller_overhead,
                include_load_shape=include_load_shape,
            )
            coefficients, _scale = fit_arrays(
                train_x, train_y, float(alpha), standardize=standardize
            )
            test_x, test_y = design_matrix(
                test,
                peer_order,
                work_by_shard,
                controller_overhead,
                include_load_shape=include_load_shape,
            )
            errors.extend((test_x @ coefficients - test_y).tolist())
        cv_rows.append(
            {
                "alpha": float(alpha),
                "rmse_log_ratio": float(math.sqrt(np.mean(np.square(errors)))),
                "rmse_approx_pct": float(100.0 * math.sqrt(np.mean(np.square(errors)))),
                "error_count": len(errors),
            }
        )
    selected = min(cv_rows, key=lambda row: (row["rmse_log_ratio"], row["alpha"]))
    design, target = design_matrix(
        pairs,
        peer_order,
        work_by_shard,
        controller_overhead,
        include_load_shape=include_load_shape,
    )
    coefficients, scale = fit_arrays(
        design, target, selected["alpha"], standardize=standardize
    )
    fitted = design @ coefficients
    residual = target - fitted
    return {
        "name": name,
        "claim_boundary": "paired online-QPS candidate selector only; online confirmation remains authoritative",
        "target": "log(candidate QPS / same-run baseline A/B mean QPS)",
        "pair_observation_count": len(pairs),
        "candidate_placement_count": len(candidate_keys),
        "include_load_shape": include_load_shape,
        "standardize_features": standardize,
        "fit_intercept": False,
        "ridge_alpha": selected["alpha"],
        "leave_one_candidate_placement_out_rmse_log_ratio": selected["rmse_log_ratio"],
        "leave_one_candidate_placement_out_rmse_approx_pct": selected["rmse_approx_pct"],
        "cross_validation_top_five": sorted(
            cv_rows, key=lambda row: (row["rmse_log_ratio"], row["alpha"])
        )[:5],
        "coefficients": [float(value) for value in coefficients],
        "feature_scale": [float(value) for value in scale],
        "fit_rmse_log_ratio": float(math.sqrt(np.mean(np.square(residual)))),
        "observed": [float(value) for value in target],
        "fitted": [float(value) for value in fitted],
        "residual": [float(value) for value in residual],
    }


def predict_ratio(
    model: dict[str, Any],
    candidate: dict[int, int],
    baseline: dict[int, int],
    peer_order: Sequence[int],
    work_by_shard: Sequence[float],
    controller_overhead: float,
) -> float:
    candidate_vector = placement_vector(
        candidate,
        peer_order,
        work_by_shard,
        controller_overhead,
        include_load_shape=bool(model["include_load_shape"]),
    )
    baseline_vector = placement_vector(
        baseline,
        peer_order,
        work_by_shard,
        controller_overhead,
        include_load_shape=bool(model["include_load_shape"]),
    )
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    log_ratio = float((candidate_vector - baseline_vector) @ coefficients)
    return 100.0 * (math.exp(log_ratio) - 1.0)


def build(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    source_plan_path = root / "confirm-v4-plan/placements.json"
    source_audit_path = root / "online-v15/evidence-audit.json"
    trace_path = root / "trace/p24-per-query.json"
    proxy_manifest_path = root / "plan/manifest.json"
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
    controller_overhead = float(
        load_json(proxy_manifest_path)["work_proxy"]["controller_overhead_model"][
            "controller_overhead_work_units"
        ]
    )
    pairs, observed_mappings = gather_pairs(root)
    models = [
        fit_model(
            name,
            include_load_shape,
            standardize,
            pairs,
            peer_order,
            work_by_shard,
            controller_overhead,
        )
        for name, include_load_shape, standardize in MODEL_SPECS
    ]
    seen = set(observed_mappings)
    rows = []
    for left_shard, right_shard, mapping in neighbor.enumerate_neighbors(baseline, peer_order):
        predictions = {
            model["name"]: predict_ratio(
                model,
                mapping,
                baseline,
                peer_order,
                work_by_shard,
                controller_overhead,
            )
            for model in models
        }
        rows.append(
            {
                "left_shard": left_shard,
                "right_shard": right_shard,
                "already_online_tested": mapping_key(mapping) in seen,
                "predicted_improvement_pct": predictions,
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
        row["predicted_improvement_mean_pct"] = float(
            np.mean(list(row["predicted_improvement_pct"].values()))
        )
        row["predicted_improvement_min_pct"] = min(
            row["predicted_improvement_pct"].values()
        )
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
        raise RuntimeError("could not select four QPS-surrogate candidates")

    placements = {BASELINE: baseline}
    metrics: dict[str, Any] = {
        BASELINE: {
            "groups": local.placement_groups(baseline, peer_order),
            "selection_reason": "formally confirmed current winner and A/B control",
        }
    }
    names = []
    for row in selected:
        name = f"qps_neighbor_{row['left_shard']}_{row['right_shard']}"
        names.append(name)
        mapping = local.swap_mapping(baseline, row["left_shard"], row["right_shard"])
        placements[name] = mapping
        metrics[name] = {
            **row,
            "groups": local.placement_groups(mapping, peer_order),
            "selection_reason": "top untested consensus from direct paired-online-QPS surrogate",
        }
    phases = [("swap-24-21-qps-a", BASELINE)] + [
        (name.replace("_", "-"), name) for name in names
    ] + [("swap-24-21-qps-b", BASELINE)]
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
            "claim_boundary": "within-run paired QPS ratios; formal confirmation remains authoritative",
            "pair_count": len(pairs),
            "unique_online_placement_count": len(observed_mappings),
            "pairs": pairs,
        },
    )
    models_path = output / "qps-surrogate-models.json"
    local.write_json_new(models_path, {"models": models})
    neighborhood_path = output / "one-swap-qps-ranking.json"
    local.write_json_new(
        neighborhood_path,
        {
            "baseline": BASELINE,
            "evaluated_swaps": len(ranked),
            "ranking_rule": [
                "minimize worst rank across three paired-QPS ridge variants",
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
        "# paired-online-QPS surrogate 候选",
        "",
        f"从 online-v2…v15 提取 {len(pairs)} 个候选/同轮 baseline QPS 比，覆盖 {len(observed_mappings)} 个唯一在线 placement。",
        "模型直接学习 placement 差分对 log-QPS 比的影响；在线 A/B 与正式确认仍是权威。",
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
        "模型 LOPO RMSE（近似百分比）："
        + " / ".join(
            f"{model['leave_one_candidate_placement_out_rmse_approx_pct']:.2f}%"
            for model in models
        ),
        "",
    ]
    with rationale_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    outputs = (
        placements_path,
        pairs_path,
        models_path,
        neighborhood_path,
        metrics_path,
        rationale_path,
    )
    manifest_path = output / "manifest.json"
    local.write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": "direct paired-online-QPS surrogate plan; online evidence remains authoritative",
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "pair_observation_count": len(pairs),
            "unique_online_placement_count": len(observed_mappings),
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
