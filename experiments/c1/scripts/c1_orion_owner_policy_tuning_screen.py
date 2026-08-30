#!/usr/bin/env python3
"""Tuning-only Recall/QPS screen for one prepared owner-policy candidate.

This runner never issues held-out queries.  It validates the native bundle,
collection, four-host placement and 64-CPU contract, measures Recall@10 on the
first 1,000 GloVe queries, selects saturation concurrency on that same tuning
split, and records three stable tuning-QPS repeats for tournament ranking.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c1.scripts import c1_orion_balance_interleaved_ab as base  # noqa: E402
from experiments.c1.scripts import (  # noqa: E402
    c1_orion_l1_owner_repair_interleaved_ab as ledger,
)


benchmark_lock = base.benchmark_lock


def arm_spec(args: argparse.Namespace) -> base.ArmSpec:
    return base.ArmSpec(
        label="A",
        name=args.combination,
        collection=args.collection,
        artifact=args.artifact.expanduser().resolve(),
        layout_dir=args.layout_dir.expanduser().resolve(),
        prepare_manifest=args.prepare_manifest.expanduser().resolve(),
        allow_balance_layout=False,
        allow_scaling_layout=False,
        allow_l1_partition_layout=True,
        allow_historical_prepare_deployment=False,
        allow_single_assignment_layout=True,
    )


def prepare_binding(
    arm: base.ArmSpec,
    *,
    base_url: str,
    deployment_manifest: Path,
    topology: Path,
) -> dict[str, Any]:
    """Accept only checksum-proven artifact-ledger supersession.

    Installing and activating a routing artifact appends its proof to the
    deployment manifest, so the whole-file digest recorded by the earlier
    prepare step necessarily changes.  This is not a generic historical
    deployment exception: topology must still match byte-for-byte and the
    current manifest must contain the exact four-node, workers-first,
    router-loaded ledger entry for this candidate.
    """

    compatibility = replace(arm, allow_historical_prepare_deployment=True)
    binding = base.validate_prepare_binding(
        compatibility,
        base_url=base_url,
        deployment_manifest=deployment_manifest,
        topology=topology,
    )
    topology_binding = binding.get("prepare_topology_binding")
    if (
        not isinstance(topology_binding, Mapping)
        or topology_binding.get("mode") != "current"
    ):
        raise RuntimeError("owner-policy tournament cannot excuse topology drift")
    ledger_arm = replace(arm, name="C_CNBR")
    current_ledger = ledger.validate_current_deployment_artifact_ledger(
        ledger_arm, binding, deployment_manifest
    )
    deployment_binding = binding.get("prepare_deployment_binding")
    if not isinstance(deployment_binding, Mapping):
        raise RuntimeError("owner-policy prepare deployment binding is missing")
    result = dict(binding)
    result["prepare_deployment_binding"] = {
        **dict(deployment_binding),
        "mode": "current_artifact_ledger_supersession",
        "explicitly_allowed": False,
        "generic_historical_exception_used": False,
        "artifact_ledger": current_ledger,
    }
    result["current_deployment_artifact_ledger"] = current_ledger
    return result


def validate_args(args: argparse.Namespace) -> None:
    if "+" not in args.combination:
        raise ValueError("combination must be OWNER+POLICY")
    if args.target_recall != 0.90 or args.recall_upper != 0.93:
        raise ValueError("tournament recall band is frozen to [0.90,0.93)")
    if args.sweep_seconds <= 0 or args.warmup_seconds <= 0 or args.measure_seconds <= 0:
        raise ValueError("timing durations must be positive")
    if args.repeats != 3:
        raise ValueError("tuning tournament repeat count is frozen to three")
    if len(args.concurrency_candidates) < 2:
        raise ValueError("at least two concurrency candidates are required")
    for name in (
        "hdf5_path",
        "topology",
        "deployment_manifest",
        "artifact",
        "layout_dir",
        "prepare_manifest",
    ):
        path = Path(getattr(args, name)).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    output = Path(args.output_dir).expanduser().resolve()
    preexisted = output.exists()
    with benchmark_lock.hold_from_args(
        args,
        args.deployment_manifest,
        owner={
            "kind": "c1_orion_owner_policy_tuning_screen",
            "collection": args.collection,
            "combination": args.combination,
            "output_dir": str(output),
        },
    ) as held:
        try:
            output_dir = base.native.create_output_directory(output)
            arm = arm_spec(args)
            deployment_path = args.deployment_manifest.expanduser().resolve(strict=True)
            topology = base.native.experiment.load_cluster_topology(args.topology)
            deployment_manifest = base.native.experiment.load_optional_json(deployment_path)
            assert deployment_manifest is not None
            deployment_evidence = base.native.build_deployment_evidence(
                deployment_path, deployment_manifest
            )
            base.native.validate_deployment_topology_transport_binding(
                topology, deployment_evidence
            )
            cluster_preflight = base.native.experiment.validate_cluster_preflight(
                args.base_url, topology
            )
            repository_preflight_start = base.repository_snapshot()
            client_affinity_start = base.hashall.verify_client_affinity()
            resources_start = base.hashall.inspect_all_containers()
            dataset = base.load_protocol_dataset(args.hdf5_path)

            inventory = {
                str(row.get("name") or "")
                for row in base.native.experiment.request_json(
                    args.base_url, "GET", "/collections"
                )["result"]["collections"]
            }
            if arm.collection not in inventory:
                raise RuntimeError(f"prepared collection is missing: {arm.collection}")
            background = sorted(inventory - {arm.collection})
            background_start = base.snapshot_quiescent_background_collections(
                args.base_url, background
            )
            binding = prepare_binding(
                arm,
                base_url=args.base_url,
                deployment_manifest=deployment_path,
                topology=args.topology,
            )
            dataset_binding = base.validate_dataset_binding(binding, dataset)
            live_start = base.validate_live_arm(
                arm,
                binding,
                base_url=args.base_url,
                train_count=dataset.train_shape[0],
                vector_dimension=dataset.train_shape[1],
            )
            placement = base.validate_round_robin_four_host_contract(
                {"A": binding, "B": binding},
                {"A": live_start, "B": live_start},
                cluster_preflight,
            )
            resource_contract = base.validate_64_cpu_resource_contract(
                resources_start, topology, deployment_evidence
            )
            proof = live_start.get("artifact_bundle", {}).get(
                "l1_partition_layout_proof"
            )
            if not isinstance(proof, dict) or proof.get("combination") != args.combination:
                raise RuntimeError("live bundle combination differs from the declaration")

            # Validation and dataset loading may import additional repository-
            # local Python modules.  Freeze the exact loaded-source set only
            # after preflight and before the first query, matching the formal
            # interleaved A/B runner's integrity contract.
            repository_measurement_start = base.repository_snapshot()
            repository_preflight_transition = (
                base.validate_repository_preflight_transition(
                    repository_preflight_start, repository_measurement_start
                )
            )

            tuning_recall = base.measure_recall_split(
                arm,
                dataset.tuning_queries,
                dataset.tuning_neighbors,
                base_url=args.base_url,
                split="tuning",
            )
            recall_value = float(tuning_recall["recall_at_10"])
            recall_band_pass = args.target_recall <= recall_value < args.recall_upper
            tuning_bodies = base.simple.make_simple_bodies(
                dataset.tuning_queries,
                batch_size=base.BATCH_SIZE,
                top_k=base.TOP_K,
            )
            host, port, endpoint = base.endpoint_for(args.base_url, arm.collection)
            sweep = [
                base.hashall.timed_run(
                    host,
                    port,
                    endpoint,
                    tuning_bodies,
                    concurrency=concurrency,
                    duration_s=args.sweep_seconds,
                    batch_size=base.BATCH_SIZE,
                )
                for concurrency in args.concurrency_candidates
            ]
            selection = base.hashall.select_saturation(sweep)
            if not selection["knee_observed"]:
                raise RuntimeError("saturation knee not observed; extend concurrency grid")
            selected_concurrency = int(selection["selected_concurrency"])
            warmup = base.hashall.timed_run(
                host,
                port,
                endpoint,
                tuning_bodies,
                concurrency=selected_concurrency,
                duration_s=args.warmup_seconds,
                batch_size=base.BATCH_SIZE,
            )
            repeats = [
                base.hashall.timed_run(
                    host,
                    port,
                    endpoint,
                    tuning_bodies,
                    concurrency=selected_concurrency,
                    duration_s=args.measure_seconds,
                    batch_size=base.BATCH_SIZE,
                )
                for _ in range(args.repeats)
            ]
            qps_values = [float(row["qps"]) for row in repeats]
            qps_mean = statistics.fmean(qps_values)
            qps_stdev = statistics.stdev(qps_values)
            qps_cv = qps_stdev / qps_mean

            live_end = base.validate_live_arm(
                arm,
                binding,
                base_url=args.base_url,
                train_count=dataset.train_shape[0],
                vector_dimension=dataset.train_shape[1],
            )
            resources_end = base.hashall.inspect_all_containers()
            client_affinity_end = base.hashall.verify_client_affinity()
            repository_end = base.repository_snapshot()
            repository_measurement_transition = (
                base.validate_repository_measurement_transition(
                    repository_measurement_start, repository_end
                )
            )
            ending_inventory = {
                str(row.get("name") or "")
                for row in base.native.experiment.request_json(
                    args.base_url, "GET", "/collections"
                )["result"]["collections"]
            }
            background_end = base.snapshot_quiescent_background_collections(
                args.base_url, background
            )
            checks = {
                "tuning_only_query_split": True,
                "dataset_binding_pass": dataset_binding.get("status") == "PASS",
                "combination_bound": proof.get("combination") == args.combination,
                "round_robin_four_hosts": placement.get("status") == "PASS",
                "resource_contract_64_cpu": resource_contract.get("status") == "PASS",
                "collection_identity_unchanged": live_start["collection_identity"]
                == live_end["collection_identity"],
                "container_resources_unchanged": base.resource_identity(resources_start)
                == base.resource_identity(resources_end),
                "client_affinity_unchanged": client_affinity_start
                == client_affinity_end,
                "repository_unchanged": (
                    repository_measurement_transition.get("status") == "PASS"
                ),
                "collection_inventory_unchanged": inventory == ending_inventory,
                "background_collections_quiescent": background_start == background_end,
                "qps_cv_at_most_5pct": qps_cv <= base.MAX_QPS_CV,
            }
            status = "PASS" if all(checks.values()) else "FAIL"
            result: dict[str, Any] = {
                "created_at": base.utc_timestamp(),
                "record_type": "c1_orion_owner_policy_tuning_screen",
                "status": status,
                "combination": args.combination,
                "collection": arm.collection,
                "protocol": {
                    "dataset": "GloVe-200-angular",
                    "distance": "Cosine",
                    "top_k": base.TOP_K,
                    "batch_size": base.BATCH_SIZE,
                    "query_range": [0, base.TUNING_QUERY_COUNT],
                    "heldout_queries_issued": False,
                    "target_recall_band": [args.target_recall, args.recall_upper],
                    "server_cpu_total": 64,
                    "logical_shards": 32,
                    "physical_hosts": 4,
                },
                "tuning_recall_at_10": recall_value,
                "recall_band_pass": recall_band_pass,
                "selected_concurrency": selected_concurrency,
                "qps": {
                    "mean": qps_mean,
                    "stdev": qps_stdev,
                    "cv": qps_cv,
                    "values": qps_values,
                },
                "assignment": proof.get("assignment"),
                "matrix_row_all_offline_gates_pass": proof.get(
                    "matrix_row_all_offline_gates_pass"
                ),
                "concurrency_sweep": sweep,
                "saturation_selection": selection,
                "warmup": warmup,
                "repeats": repeats,
                "tuning_recall": tuning_recall,
                "repository_preflight_transition": repository_preflight_transition,
                "repository_measurement_transition": repository_measurement_transition,
                "checks": checks,
                "benchmark_lock": held.evidence(),
            }
            base.write_json(output_dir / "screen.json", result)
            completion = {
                "status": status,
                "check_count": len(checks),
                "passed": sum(bool(value) for value in checks.values()),
                "checks": [
                    {"check": key, "status": "PASS" if value else "FAIL"}
                    for key, value in checks.items()
                ],
                "screen_sha256": base.sha256_path(output_dir / "screen.json"),
            }
            base.write_json(output_dir / "completion-audit.json", completion)
            if status != "PASS":
                raise RuntimeError(f"tuning screen completion audit failed: {checks}")
            return output_dir
        except BaseException as exc:
            if not preexisted and output.is_dir():
                try:
                    base.write_json(
                        output / "execution-failed.json",
                        {
                            "created_at": base.utc_timestamp(),
                            "status": "FAIL",
                            "error": repr(exc),
                            "benchmark_lock": held.evidence(),
                        },
                    )
                except Exception:
                    pass
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--hdf5-path", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--combination", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--prepare-manifest", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--recall-upper", type=float, default=0.93)
    parser.add_argument("--concurrency-candidates", default="1,2,4,8,16,32,64")
    parser.add_argument("--sweep-seconds", type=float, default=5.0)
    parser.add_argument("--warmup-seconds", type=float, default=6.0)
    parser.add_argument("--measure-seconds", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    benchmark_lock.add_cli_arguments(parser)
    args = parser.parse_args(argv)
    args.concurrency_candidates = base.parse_int_csv(args.concurrency_candidates)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(json.dumps({"output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
