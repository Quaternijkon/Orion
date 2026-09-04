#!/usr/bin/env python3
"""Refine the confirmed swap_24_21 winner over its complete one-swap neighborhood."""

from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT = DEFAULT_ROOT / "refine-v12-neighbor"
BASELINE = "swap_24_21"
ALIASES = {"controller_aware_tail": "controller_aware_v1"}
OBSERVED_STRATEGIES = (
    "round_robin",
    "size_balanced",
    "controller_aware_v1",
    "learned_robust_workcap5500",
    "learned_robust",
    "swap_19_15",
    "swap_6_3",
    "swap_c8_0",
    "swap_28_7",
    "swap_29_25",
    "swap_16_18",
    "swap_29_18",
    "swap_27_16",
    "swap_5_8",
    "swap_27_25",
    "swap_30_11",
    "swap_16_10",
    "swap_25_10",
    "swap_16_17",
    "swap_25_17",
    "swap_24_13",
    "swap_24_21",
    "swap_11_10",
    "swap_7_13",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def phase_files(root: Path) -> list[Path]:
    result = list((root / "online-v1/phases").glob("*/benchmark.json"))
    for version in range(2, 12):
        result.extend((root / f"online-v{version}/phases").glob("*/screen.json"))
    return sorted(result)


def gather_observations(
    root: Path, peer_order: Sequence[int], peer_hosts: dict[int, str]
) -> tuple[list[dict[str, Any]], dict[str, dict[int, int]], dict[str, Any]]:
    values: dict[str, list[list[float]]] = {}
    mappings: dict[str, dict[int, int]] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    for path in phase_files(root):
        payload = load_json(path)
        strategy = ALIASES.get(str(payload["strategy"]), str(payload["strategy"]))
        if strategy not in OBSERVED_STRATEGIES:
            continue
        concurrency_one = next(
            row for row in payload["concurrency_sweep"] if int(row["concurrency"]) == 1
        )
        demand = [
            1_000_000.0
            * float(concurrency_one["cpu_average_cores"][peer_hosts[peer_id]])
            / float(concurrency_one["qps"])
            for peer_id in peer_order
        ]
        mapping = {
            int(shard): int(peer)
            for shard, peer in payload["ending_placement"]["placement"].items()
        }
        if strategy in mappings and mapping != mappings[strategy]:
            raise RuntimeError(f"strategy {strategy} changed placement at {path}")
        mappings[strategy] = mapping
        values.setdefault(strategy, []).append(demand)
        sources.setdefault(strategy, []).append(
            {
                "path": str(path),
                "sha256": local.sha256_path(path),
                "node_cpu_microseconds_per_query": demand,
            }
        )
    if set(values) != set(OBSERVED_STRATEGIES):
        raise RuntimeError(
            f"observation strategies mismatch: {sorted(values)} != {sorted(OBSERVED_STRATEGIES)}"
        )
    observations = []
    provenance = {}
    for strategy in OBSERVED_STRATEGIES:
        averaged = np.mean(np.asarray(values[strategy], dtype=np.float64), axis=0)
        provenance[strategy] = {
            "replicate_count": len(values[strategy]),
            "sources": sources[strategy],
            "mean_node_cpu_microseconds_per_query": [float(value) for value in averaged],
        }
        mapping = mappings[strategy]
        for node_index, peer_id in enumerate(peer_order):
            observations.append(
                {
                    "strategy": strategy,
                    "peer_id": peer_id,
                    "host": peer_hosts[peer_id],
                    "controller": node_index == 0,
                    "shards": sorted(shard for shard, owner in mapping.items() if owner == peer_id),
                    "cpu_microseconds_per_query": float(averaged[node_index]),
                }
            )
    if len(observations) != 96:
        raise RuntimeError(f"expected 96 node observations, got {len(observations)}")
    return observations, mappings, provenance


def mapping_key(mapping: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(mapping.items()))


def build(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    source_plan_path = root / "confirm-v4-plan/placements.json"
    source_summary_path = root / "online-v11/formal-v4-summary.json"
    source_audit_path = root / "online-v11/evidence-audit.json"
    trace_path = root / "trace/p24-per-query.json"
    proxy_manifest_path = root / "plan/manifest.json"
    layout_manifest_path = layout_dir / "build-manifest.json"
    source_plan = load_json(source_plan_path)
    source_summary = load_json(source_summary_path)
    source_audit = load_json(source_audit_path)
    trace = load_json(trace_path)
    layout_manifest = load_json(layout_manifest_path)
    if source_summary.get("confirmed_winner") != BASELINE:
        raise RuntimeError("latest formal summary does not confirm swap_24_21")
    if source_audit.get("status") != "PASS":
        raise RuntimeError("latest formal audit is not PASS")

    peer_order = [int(value) for value in source_plan["peer_order"]]
    peer_hosts = {int(peer): str(host) for peer, host in source_plan["peer_hosts"].items()}
    counts = [int(value) for value in layout_manifest["routing"]["shard_counts"]]
    shard_features = refine.shard_features(trace, counts)
    work_by_shard = [1000.0 * row["work_1000"] for row in shard_features]
    controller_overhead = float(
        load_json(proxy_manifest_path)["work_proxy"]["controller_overhead_model"][
            "controller_overhead_work_units"
        ]
    )
    observations, observed_mappings, provenance = gather_observations(
        root, peer_order, peer_hosts
    )
    baseline = observed_mappings[BASELINE]
    if baseline != local.normalize_mapping(source_plan["placements"][BASELINE]):
        raise RuntimeError("confirmed winner placement changed")
    models = [
        neighbor.fit_model(
            name,
            include_proxy,
            standardize,
            observations,
            work_by_shard,
            controller_overhead,
            observed_strategies=OBSERVED_STRATEGIES,
        )
        for name, include_proxy, standardize in neighbor.MODEL_SPECS
    ]

    seen = {mapping_key(mapping) for mapping in observed_mappings.values()}
    rows = []
    for left_shard, right_shard, mapping in neighbor.enumerate_neighbors(baseline, peer_order):
        predictions = {
            model["name"]: local.predict_mapping(
                model,
                mapping,
                peer_order,
                peer_hosts,
                work_by_shard,
                controller_overhead,
            )
            for model in models
        }
        groups = local.placement_groups(mapping, peer_order)
        trace_metrics = refine.trace_peak_metrics(trace, counts, groups, controller_overhead)
        rows.append(
            {
                "left_shard": left_shard,
                "right_shard": right_shard,
                "already_online_tested": mapping_key(mapping) in seen,
                "ridge_predictions": predictions,
                "moved_physical_vector_copies": counts[left_shard] + counts[right_shard],
                **trace_metrics,
            }
        )
    for model in models:
        model_name = model["name"]
        ranked = sorted(
            rows,
            key=lambda row: (
                row["ridge_predictions"][model_name]["worst_node_cpu_microseconds_per_query"],
                row["left_shard"],
                row["right_shard"],
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row.setdefault("model_ranks", {})[model_name] = rank
    trace_ranked = sorted(
        rows,
        key=lambda row: (
            row["per_query_node_peak_p95"],
            row["per_query_node_peak_p99"],
            row["left_shard"],
            row["right_shard"],
        ),
    )
    for rank, row in enumerate(trace_ranked, start=1):
        row["trace_p95_rank"] = rank
    for row in rows:
        ranks = list(row["model_ranks"].values())
        row["model_worst_rank"] = max(ranks)
        row["model_rank_sum"] = sum(ranks)
    ranked = sorted(
        rows,
        key=lambda row: (
            row["model_worst_rank"],
            row["model_rank_sum"],
            row["trace_p95_rank"],
            row["moved_physical_vector_copies"],
            row["left_shard"],
            row["right_shard"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["consensus_rank"] = rank
    selected = [row for row in ranked if not row["already_online_tested"]][:4]
    if len(selected) != 4:
        raise RuntimeError("could not select four untested neighbors")

    candidate_mappings = {BASELINE: baseline}
    candidate_metrics: dict[str, Any] = {
        BASELINE: {
            "groups": local.placement_groups(baseline, peer_order),
            "selection_reason": "formally confirmed current winner and A/B control",
        }
    }
    names = []
    for row in selected:
        name = f"neighbor_{row['left_shard']}_{row['right_shard']}"
        names.append(name)
        mapping = local.swap_mapping(baseline, row["left_shard"], row["right_shard"])
        candidate_mappings[name] = mapping
        candidate_metrics[name] = {
            **row,
            "groups": local.placement_groups(mapping, peer_order),
            "selection_reason": "top untested complete-neighborhood consensus after 24 online placements",
        }

    phases = [("swap-24-21-neighbor-a", BASELINE)] + [
        (name.replace("_", "-"), name) for name in names
    ] + [("swap-24-21-neighbor-b", BASELINE)]
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
                for name, mapping in candidate_mappings.items()
            },
        },
    )
    neighborhood_path = output / "one-swap-neighborhood.json"
    local.write_json_new(
        neighborhood_path,
        {
            "baseline": BASELINE,
            "evaluated_swaps": len(ranked),
            "already_online_tested_neighbors": sum(row["already_online_tested"] for row in ranked),
            "ranking_rule": [
                "minimize worst rank across three updated ridge models",
                "then model rank sum",
                "then trace p95 rank",
                "then moved copies",
            ],
            "neighbors": ranked,
        },
    )
    metrics_path = output / "candidate-metrics.json"
    local.write_json_new(metrics_path, candidate_metrics)
    models_path = output / "service-demand-models.json"
    local.write_json_new(
        models_path,
        {
            "claim_boundary": "candidate selector only; online QPS remains authoritative",
            "observed_strategies": list(OBSERVED_STRATEGIES),
            "observations": observations,
            "observation_provenance": provenance,
            "models": models,
        },
    )
    rationale_path = output / "RATIONALE_zh.md"
    lines = [
        "# swap_24_21 完整单交换邻域候选",
        "",
        "基于 24 个唯一在线 placement、96 个节点观测，完整枚举新赢家的 384 个单交换邻居。",
        "选择模型共识最高的四个未测试邻居；离线排名不构成 QPS 结论。",
        "",
        "| 候选 | swap | consensus rank | model worst / sum | trace p95 rank | moved copies |",
        "|---|---|---:|---:|---:|---:|",
        *[
            (
                f"| `{name}` | {row['left_shard']}↔{row['right_shard']} | "
                f"{row['consensus_rank']} | {row['model_worst_rank']} / {row['model_rank_sum']} | "
                f"{row['trace_p95_rank']} | {row['moved_physical_vector_copies']} |"
            )
            for name, row in candidate_metrics.items()
            if name != BASELINE
        ],
        "",
    ]
    with rationale_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    outputs = (placements_path, neighborhood_path, metrics_path, models_path, rationale_path)
    manifest_path = output / "manifest.json"
    local.write_json_new(
        manifest_path,
        {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "OFFLINE_REFINEMENT_ONLY",
            "claim_boundary": (
                "complete one-swap enumeration around swap_24_21; models do not establish "
                "online QPS or global optimality"
            ),
            "physical_machine_count": 4,
            "logical_shard_count": 32,
            "modeling": {
                "unique_online_placement_count": len(OBSERVED_STRATEGIES),
                "node_observation_count": len(observations),
                "one_swap_neighbors_evaluated": len(ranked),
                "already_tested_neighbor_count": sum(row["already_online_tested"] for row in ranked),
                "selected_candidate_count": len(names),
                "model_count": len(models),
            },
            "selected_strategies": names,
            "inputs": {
                "source_plan": {"path": str(source_plan_path), "sha256": local.sha256_path(source_plan_path)},
                "source_summary": {"path": str(source_summary_path), "sha256": local.sha256_path(source_summary_path)},
                "source_audit": {"path": str(source_audit_path), "sha256": local.sha256_path(source_audit_path)},
                "trace": {"path": str(trace_path), "sha256": local.sha256_path(trace_path)},
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
