#!/usr/bin/env python3
"""Online-screen learned Orion physical-placement refinements."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_TOOL = Path(__file__).resolve().with_name("orion_physical_placement_online.py")
DEFAULT_ROOT = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/orion-load-balance-r090-20260825"
)
DEFAULT_PLAN = DEFAULT_ROOT / "refine-v2"
DEFAULT_OUTPUT = DEFAULT_ROOT / "online-v2"
DEFAULT_COLLECTION = "orion_lb_r090_p24_20260825_v2"
PHASES = (
    ("controller-v1-a", "controller_aware_v1"),
    ("learned-workcap5500", "learned_robust_workcap5500"),
    ("learned-robust", "learned_robust"),
    ("controller-v1-b", "controller_aware_v1"),
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


online = load_module(ONLINE_TOOL, "orion_physical_placement_online_for_screen")
hashall = online.hashall
simple = online.simple
fixed = online.fixed


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_placements(payload: dict[str, Any]) -> tuple[list[int], dict[str, dict[int, int]]]:
    peers = [int(value) for value in payload["peer_order"]]
    placements = {
        name: {int(shard): int(peer) for shard, peer in mapping.items()}
        for name, mapping in payload["placements"].items()
    }
    for name, mapping in placements.items():
        if set(mapping) != set(range(32)) or set(mapping.values()) != set(peers):
            raise ValueError(f"invalid placement {name}")
    return peers, placements


def measure_screen(
    exp,
    args: argparse.Namespace,
    phase_id: str,
    strategy: str,
    placement_record: dict[str, Any],
    queries,
    neighbors,
) -> dict[str, Any]:
    phase_root = args.output_root / "phases" / phase_id
    output_path = phase_root / "screen.json"
    if output_path.is_file():
        result = load_json(output_path)
        if result.get("status") != "PASS":
            raise RuntimeError(f"existing screen result is not PASS: {output_path}")
        return result
    parsed = urllib.parse.urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6333
    endpoint = (
        f"/collections/{urllib.parse.quote(args.collection, safe='')}/points/search/batch"
    )
    heldout_queries = queries[1000:10000]
    heldout_neighbors = neighbors[1000:10000]
    recall = simple.recall_probe(
        host,
        port,
        endpoint,
        heldout_queries,
        heldout_neighbors,
        batch_size=200,
        top_k=10,
    )
    if float(recall["recall_at_10"]) < 0.90:
        raise RuntimeError(f"{phase_id} misses recall gate: {recall}")
    bodies = simple.make_simple_bodies(heldout_queries, batch_size=200, top_k=10)
    sweep = [
        hashall.timed_run(
            host,
            port,
            endpoint,
            bodies,
            concurrency=concurrency,
            duration_s=args.sweep_seconds,
            batch_size=200,
        )
        for concurrency in args.concurrency_candidates
    ]
    selection = hashall.select_saturation(sweep)
    if not selection["knee_observed"]:
        raise RuntimeError(f"{phase_id}: saturation knee was not observed")
    selected = int(selection["selected_concurrency"])
    warmup = hashall.timed_run(
        host,
        port,
        endpoint,
        bodies,
        concurrency=selected,
        duration_s=args.warmup_seconds,
        batch_size=200,
    )
    repeats = [
        hashall.timed_run(
            host,
            port,
            endpoint,
            bodies,
            concurrency=selected,
            duration_s=args.measure_seconds,
            batch_size=200,
        )
        for _repeat in range(args.repeats)
    ]
    aggregate = online.aggregate_repeat_metrics(repeats)
    contract = online.verify_collection_contract(exp, args)
    cluster = exp.collection_cluster_info(args.base_url, args.collection)
    expected = {
        int(shard): int(peer)
        for shard, peer in placement_record["proof"]["expected_placement"].items()
    }
    ending = exp.validate_numeric_shard_explicit_placement(
        contract["collection_info"],
        cluster,
        sorted(set(expected.values())),
        32,
        expected,
        include_controller=True,
    )
    formal = bool(args.formal_confirmation)
    result = {
        "timestamp": online.utc_timestamp(),
        "record_type": (
            "orion_physical_placement_online_confirmation"
            if formal
            else "orion_physical_placement_online_screen"
        ),
        "status": "PASS",
        "phase_id": phase_id,
        "strategy": strategy,
        "screen_only": not formal,
        "formal_confirmation": formal,
        "claim_boundary": (
            "formal five-stage confirmation within tested candidates; not a global-optimality proof"
            if formal
            else "screening evidence; finalist requires five-repeat confirmation"
        ),
        "physical_machine_count": 4,
        "logical_shard_count": 32,
        "heldout_recall": recall,
        "placement": placement_record,
        "concurrency_sweep": sweep,
        "saturation_selection": selection,
        "selected_concurrency": selected,
        "warmup": warmup,
        "repeats": repeats,
        "repeat_count": len(repeats),
        **aggregate,
        "ending_placement": ending,
    }
    online.write_json(output_path, result)
    return result


def summarize(args: argparse.Namespace, results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_strategy.setdefault(result["strategy"], []).append(result)
    strategies = {
        strategy: {
            "phase_count": len(rows),
            "qps_mean_across_phases": statistics.fmean(row["qps_mean"] for row in rows),
            "qps_min": min(row["qps_mean"] for row in rows),
            "qps_max": max(row["qps_mean"] for row in rows),
            "recall_min": min(row["heldout_recall"]["recall_at_10"] for row in rows),
            "selected_concurrencies": [row["selected_concurrency"] for row in rows],
            "cpu_cores_by_phase": [
                row["cpu_average_cores_mean_across_repeats"] for row in rows
            ],
        }
        for strategy, rows in by_strategy.items()
    }
    ranking = sorted(
        strategies,
        key=lambda name: strategies[name]["qps_mean_across_phases"],
        reverse=True,
    )
    formal = bool(args.formal_confirmation)
    summary = {
        "timestamp": online.utc_timestamp(),
        "status": "CONFIRMATION_COMPLETE" if formal else "SCREEN_COMPLETE",
        "claim_boundary": (
            "confirmation is limited to the tested five-stage sequence and does not prove global optimality"
            if formal
            else "ranking is provisional until formal five-repeat confirmation"
        ),
        "phases": [
            {
                "phase_id": row["phase_id"],
                "strategy": row["strategy"],
                "recall_at_10": row["heldout_recall"]["recall_at_10"],
                "selected_concurrency": row["selected_concurrency"],
                "qps_mean": row["qps_mean"],
                "qps_cv": row["qps_cv"],
                "cpu": row["cpu_average_cores_mean_across_repeats"],
            }
            for row in results
        ],
        "strategy_summary": strategies,
        "ranking": ranking,
        "screen_winner": ranking[0],
    }
    if formal:
        baseline_rows = [results[0], results[3]]
        finalist_rows = [results[1], results[4]]
        baseline_strategy = baseline_rows[0]["strategy"]
        finalist_strategy = finalist_rows[0]["strategy"]
        all_recall_pass = all(
            float(row["heldout_recall"]["recall_at_10"]) >= args.target_recall
            for row in results
        )
        strict_dominance = min(row["qps_mean"] for row in finalist_rows) > max(
            row["qps_mean"] for row in baseline_rows
        )
        confirmed = all_recall_pass and strict_dominance
        baseline_mean = statistics.fmean(row["qps_mean"] for row in baseline_rows)
        finalist_mean = statistics.fmean(row["qps_mean"] for row in finalist_rows)
        summary["confirmation_rule"] = (
            "both finalist phases exceed both baseline phases and all five held-out "
            "Recall@10 values satisfy the target"
        )
        summary["formal_confirmation_result"] = {
            "baseline_strategy": baseline_strategy,
            "finalist_strategy": finalist_strategy,
            "backup_strategy": results[2]["strategy"],
            "baseline_qps_mean_across_phases": baseline_mean,
            "finalist_qps_mean_across_phases": finalist_mean,
            "finalist_improvement_pct": 100.0 * (finalist_mean / baseline_mean - 1.0),
            "baseline_qps_max": max(row["qps_mean"] for row in baseline_rows),
            "finalist_qps_min": min(row["qps_mean"] for row in finalist_rows),
            "strict_endpoint_dominance": strict_dominance,
            "all_recall_pass": all_recall_pass,
            "confirmed_over_baseline": confirmed,
        }
        summary["confirmed_winner"] = finalist_strategy if confirmed else None
    online.write_json(args.output_root / args.summary_file, summary)
    return summary


def phase_plan(
    payload: dict[str, Any], field: str = "screen_phases"
) -> tuple[tuple[str, str], ...]:
    raw = payload.get(field)
    if raw is None:
        if field == "screen_phases":
            return PHASES
        raise ValueError(f"placement plan is missing required {field}")
    phases = tuple((str(row["phase_id"]), str(row["strategy"])) for row in raw)
    if not phases or len({phase_id for phase_id, _strategy in phases}) != len(phases):
        raise ValueError("screen phases must be non-empty with unique phase IDs")
    known = set(payload["placements"])
    unknown = sorted({strategy for _phase_id, strategy in phases} - known)
    if unknown:
        raise ValueError(f"screen phases reference unknown strategies: {unknown}")
    return phases


def validate_confirmation_phases(phases: Sequence[tuple[str, str]]) -> None:
    if len(phases) != 5:
        raise ValueError("formal confirmation requires exactly five phases")
    strategies = [strategy for _phase_id, strategy in phases]
    if strategies[0] != strategies[3]:
        raise ValueError("formal confirmation phases 1 and 4 must be the same baseline")
    if strategies[1] != strategies[4]:
        raise ValueError("formal confirmation phases 2 and 5 must be the same finalist")
    if len({strategies[0], strategies[1], strategies[2]}) != 3:
        raise ValueError("formal baseline, finalist, and backup must be distinct strategies")


def prepare_screen_collection(args: argparse.Namespace, exp, layout: dict[str, Any]) -> dict[str, Any]:
    manifest_path = args.reuse_preparation_manifest
    if manifest_path is None:
        return online.prepare_collection(args, exp, layout)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"reuse preparation manifest does not exist: {manifest_path}")
    names = online.collection_names(exp, args.base_url)
    unexpected = [name for name in names if name != args.collection]
    if unexpected:
        raise RuntimeError(f"unexpected live collections before screen: {unexpected}")
    if args.collection not in names:
        raise RuntimeError(
            f"cannot reuse preparation manifest because collection is absent: {args.collection}"
        )
    return {
        "status": "REUSED_EXTERNAL_MANIFEST",
        "preparation_manifest": str(manifest_path),
        "preparation_manifest_sha256": online.sha256_path(manifest_path),
        "layout_generation": layout["parameters"]["generation"],
    }


def execute(args: argparse.Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    exp = hashall.load_experiment_module(REPO_ROOT)
    plan_manifest = load_json(args.plan_dir / "manifest.json")
    if plan_manifest.get("status") != "OFFLINE_REFINEMENT_ONLY":
        raise RuntimeError("unexpected refinement-plan status")
    placement_payload = load_json(args.plan_dir / "placements.json")
    peers, placements = normalize_placements(placement_payload)
    phase_field = "confirmation_phases" if args.formal_confirmation else "screen_phases"
    phases = phase_plan(placement_payload, phase_field)
    if args.formal_confirmation:
        validate_confirmation_phases(phases)
    original_path = args.output_root / "original-resources.json"
    if not original_path.is_file():
        online.write_json(
            original_path,
            {"timestamp": online.utc_timestamp(), "containers": hashall.inspect_all_containers()},
        )
    body_error: BaseException | None = None
    try:
        hashall.wait_cluster_ready(exp, args.base_url)
        online.set_full_qdrant_resources(args.output_root / "resource-contract.json")
        layout = load_json(args.layout_dir / "build-manifest.json")
        preparation = prepare_screen_collection(args, exp, layout)
        online.write_json(args.output_root / "preparation.json", preparation)
        online.activate_artifact(args, layout)
        hashall.wait_cluster_ready(exp, args.base_url)
        online.write_json(
            args.output_root / "collection-contract.json",
            online.verify_collection_contract(exp, args),
        )
        queries, neighbors = fixed.load_dataset(args.hdf5_path, 10)
        results = []
        for phase_id, strategy in phases:
            placement = online.apply_placement(
                exp, args, phase_id, strategy, peers, placements[strategy]
            )
            hashall.wait_cluster_ready(exp, args.base_url)
            result = measure_screen(
                exp, args, phase_id, strategy, placement, queries, neighbors
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase_id": phase_id,
                        "strategy": strategy,
                        "qps_mean": result["qps_mean"],
                        "qps_cv": result["qps_cv"],
                        "selected_concurrency": result["selected_concurrency"],
                        "cpu": result["cpu_average_cores_mean_across_repeats"],
                    }
                ),
                flush=True,
            )
        summary = summarize(args, results)
        online.write_json(
            args.output_root / args.completion_file,
            {
                "timestamp": online.utc_timestamp(),
                "status": (
                    "CONFIRMATION_COMPLETE"
                    if args.formal_confirmation
                    else "SCREEN_COMPLETE"
                ),
                "winner": (
                    summary.get("confirmed_winner")
                    if args.formal_confirmation
                    else summary["screen_winner"]
                ),
                (
                    "collection_retained_after_confirmation"
                    if args.formal_confirmation
                    else "collection_retained_for_formal_confirmation"
                ): args.collection,
            },
        )
        return 0
    except BaseException as error:
        body_error = error
        online.write_json(
            args.output_root / "screen-failed.json",
            {
                "timestamp": online.utc_timestamp(),
                "status": "FAILED",
                "error": repr(error),
            },
        )
        raise
    finally:
        original = load_json(original_path)["containers"]
        try:
            hashall.restore_resource_state(
                original, args.output_root / "resource-restored.json"
            )
        except Exception:
            if body_error is None:
                raise


def parse_int_csv(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if values != sorted(set(values)) or not values:
        raise argparse.ArgumentTypeError("concurrency values must be sorted and unique")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=online.DEFAULT_BASE_URL)
    parser.add_argument("--run-id", default=online.DEFAULT_RUN_ID)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--topology", type=Path, default=online.DEFAULT_TOPOLOGY)
    parser.add_argument("--layout-dir", type=Path, default=online.DEFAULT_LAYOUT_DIR)
    parser.add_argument("--hdf5-path", type=Path, default=online.DEFAULT_HDF5)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cargo-target-dir", type=Path, default=online.DEFAULT_CARGO_TARGET)
    parser.add_argument("--importer-binary", type=Path, default=online.DEFAULT_IMPORTER)
    parser.add_argument(
        "--concurrency-candidates", type=parse_int_csv, default=parse_int_csv("1,2,4,8,16,32")
    )
    parser.add_argument("--sweep-seconds", type=float, default=6.0)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--measure-seconds", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--summary-file", default="screen-summary.json")
    parser.add_argument("--completion-file", default="screen-complete.json")
    parser.add_argument("--reuse-preparation-manifest", type=Path)
    parser.add_argument("--formal-confirmation", action="store_true")
    parser.add_argument("--transfer-timeout", type=float, default=10_800.0)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--max-repeats", type=int, default=3)
    parser.add_argument("--max-cv", type=float, default=0.05)
    args = parser.parse_args(argv)
    for name in (
        "topology",
        "layout_dir",
        "hdf5_path",
        "plan_dir",
        "output_root",
        "cargo_target_dir",
        "importer_binary",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.reuse_preparation_manifest is not None:
        args.reuse_preparation_manifest = args.reuse_preparation_manifest.expanduser().resolve()
    if args.repeats < 2:
        raise ValueError("screen requires at least two repeats")
    if args.formal_confirmation:
        if args.repeats < 5:
            raise ValueError("formal confirmation requires at least five repeats")
        if args.measure_seconds < 20.0:
            raise ValueError("formal confirmation requires at least 20 seconds per repeat")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
