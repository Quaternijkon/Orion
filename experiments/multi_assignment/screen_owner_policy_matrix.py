#!/usr/bin/env python3
"""Screen load-balance owner x multi-assignment combinations offline.

Every owner is aligned to the same immutable production upper graph and is
evaluated with the same full-dataset attachments, query hits, ground truth,
dynamic-ef contract, and BMR_10 policy.  Artifact owners are extracted from
the first ``shard_membership`` entry of each ordered upper node; this makes
historical multi-membership artifacts explicit new primary-owner candidates
rather than claiming byte-for-byte reproduction of their original layout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance.run_offline_screen import (  # noqa: E402
    checked_binary_matrix,
    load_upper,
    local_hits,
    query_metrics,
    sha256_path,
    topology_metrics,
)
from experiments.multi_assignment import budgeted_policy  # noqa: E402
from experiments.multi_assignment.screen_fixed_owner import (  # noqa: E402
    MAX_COVERAGE_DROP,
    MAX_FULL_COVERAGE_DROP,
    MAX_ROUTE_WORK_RATIO,
    POLICIES,
    _copy_histogram,
    _gate_results,
    _load_json,
    _load_metrics,
    _semantic_sha256,
    build_vote_evidence,
    membership_for_policy,
)


POLICY_NAMES = ("current_all_max", "single_rank", budgeted_policy.CANDIDATE_ID)
EDGE_CUT_MAX_RATIO = 1.03
RETAINED_DEGREE_MIN_RATIO = 0.95
RETAINED_DEGREE_P10_MAX_DROP = 0.025
ISOLATED_FRACTION_MAX_DELTA = 0.01
ISOLATED_FRACTION_ABSOLUTE_MAX = 0.03
LARGEST_COMPONENT_MEAN_MAX_DROP = 0.02
LARGEST_COMPONENT_MIN_FLOOR = 0.25


@dataclass(frozen=True)
class NamedOwnerPath:
    name: str
    path: Path
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-b-screen", required=True)
    parser.add_argument(
        "--artifact-owner",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Extract an owner from a generation/graphless artifact.",
    )
    parser.add_argument(
        "--owner-binary",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Read one zero-based i32le owner per reference upper node. This is "
            "the input boundary for checksum-bound experimental graph partitioners."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-count", type=int, default=0)
    return parser.parse_args()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_named_paths(
    artifact_values: Iterable[str], binary_values: Iterable[str]
) -> list[NamedOwnerPath]:
    result: list[NamedOwnerPath] = []
    names = {"C_CNBR"}
    for values, kind in (
        (artifact_values, "artifact"),
        (binary_values, "binary"),
    ):
        for raw in values:
            name, separator, path_text = raw.partition("=")
            if not separator or not name or not path_text:
                raise ValueError(f"invalid NAME=PATH owner specification: {raw}")
            if name in names:
                raise ValueError(f"duplicate/reserved owner name: {name}")
            path = Path(path_text).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"owner input is missing: {path}")
            names.add(name)
            result.append(NamedOwnerPath(name=name, path=path, kind=kind))
    return result


def binary_owner(
    path: Path,
    *,
    reference_labels: np.ndarray,
    num_partitions: int,
    reference_artifact_sha256: str,
    reference_upper_graph_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected_bytes = len(reference_labels) * np.dtype("<i4").itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"owner binary has {path.stat().st_size} bytes; expected "
            f"{expected_bytes}: {path}"
        )
    owner = np.asarray(
        np.memmap(path, dtype="<i4", mode="r", shape=(len(reference_labels),)),
        dtype=np.int32,
    )
    if np.any(owner < 0) or np.any(owner >= num_partitions):
        raise ValueError(f"owner binary contains an invalid partition: {path}")
    source: dict[str, Any] = {
        "source_kind": "frozen_i32le_owner_binary",
        "owner_binary": file_record(path),
        "upper_node_count": len(owner),
        "historical_layout_reproduction_claim": False,
    }
    sibling_manifest = path.parent / "attraction-weighted-manifest.json"
    if not sibling_manifest.is_file():
        raise ValueError(
            "owner binary lacks attraction-weighted-manifest.json identity binding"
        )
    manifest = _load_json(sibling_manifest)
    owner_record = manifest.get("owner") or {}
    if owner_record.get("sha256") != source["owner_binary"]["sha256"]:
        raise ValueError("owner binary checksum differs from sibling manifest")
    if (manifest.get("source") or {}).get("upper_node_count") != len(owner):
        raise ValueError("owner binary upper-node count differs from sibling manifest")
    if (manifest.get("parameters") or {}).get("num_partitions") != num_partitions:
        raise ValueError("owner binary partition count differs from sibling manifest")
    manifest_source = manifest.get("source") or {}
    if manifest_source.get("artifact_sha256") != reference_artifact_sha256:
        raise ValueError("owner binary source artifact differs from Phase-A")
    if manifest_source.get("upper_graph_sha256") != reference_upper_graph_sha256:
        raise ValueError("owner binary upper graph differs from Phase-A")
    source["owner_manifest"] = file_record(sibling_manifest)
    source["owner_generator_record_type"] = manifest.get("record_type")
    source["owner_generator_contract"] = manifest.get("contract")
    return owner, source


def owner_semantic_sha256(labels: np.ndarray, owner: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(labels, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(owner, dtype="<i4").tobytes(order="C"))
    return digest.hexdigest()


def artifact_owner(
    path: Path,
    *,
    reference_labels: np.ndarray,
    reference_navigator_sha256: str,
    num_partitions: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    artifact, _adj, _vectors, labels, _entry, _left, _right, navigator_sha = (
        load_upper(path)
    )
    if navigator_sha != reference_navigator_sha256:
        raise ValueError(f"owner artifact uses a different upper navigator: {path}")
    if int(artifact["shard_count"]) != num_partitions:
        raise ValueError(f"owner artifact shard count drifted: {path}")

    memberships = artifact.get("upper_nodes")
    if not isinstance(memberships, list) or len(memberships) != len(labels):
        raise ValueError(f"owner artifact upper_nodes are invalid: {path}")
    label_to_owner: dict[int, int] = {}
    multi_membership = 0
    maximum_memberships = 0
    for node in memberships:
        row = node.get("shard_membership")
        if not isinstance(row, list) or not row:
            raise ValueError(f"owner artifact has an empty upper membership: {path}")
        shards = [int(value) for value in row]
        if any(value < 0 or value >= num_partitions for value in shards):
            raise ValueError(f"owner artifact has an invalid upper membership: {path}")
        label = int(node["label"])
        if label in label_to_owner:
            raise ValueError(f"owner artifact repeats upper label {label}: {path}")
        label_to_owner[label] = shards[0]
        multi_membership += int(len(shards) > 1)
        maximum_memberships = max(maximum_memberships, len(shards))
    try:
        owner = np.asarray(
            [label_to_owner[int(label)] for label in reference_labels], dtype=np.int32
        )
    except KeyError as exc:
        raise ValueError(
            f"owner artifact lacks reference upper label {exc.args[0]}: {path}"
        ) from exc
    if len(label_to_owner) != len(reference_labels):
        raise ValueError(f"owner artifact has a different upper-label set: {path}")
    return owner, {
        "source_kind": "artifact_first_shard_membership_primary",
        "artifact": file_record(path),
        "artifact_generation": artifact.get("generation"),
        "artifact_layout_sha256": artifact.get("layout_sha256"),
        "upper_navigator_sha256": navigator_sha,
        "upper_node_count": len(owner),
        "upper_nodes_with_multiple_memberships": multi_membership,
        "upper_nodes_with_multiple_memberships_fraction": multi_membership / len(owner),
        "maximum_source_upper_memberships": maximum_memberships,
        "historical_layout_reproduction_claim": False,
    }


def assignment_metrics(
    *,
    owner_name: str,
    policy_name: str,
    membership: np.ndarray,
    owner: np.ndarray,
    labels: np.ndarray,
    query_local: np.ndarray,
    ground_truth: np.ndarray,
    dynamic_ef_base: int,
    dynamic_ef_factor: int,
    owner_source: dict[str, Any],
    owner_topology: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    copy_count = membership.sum(axis=1, dtype=np.int16)
    loads = membership.sum(axis=0, dtype=np.int64)
    result: dict[str, Any] = {
        "owner": owner_name,
        "policy": policy_name,
        "combination": f"{owner_name}+{policy_name}",
        "logical_point_count": len(membership),
        "physical_point_count": int(copy_count.sum()),
        "expansion_ratio": float(copy_count.mean()),
        "extra_copy_fraction": float(copy_count.mean() - 1.0),
        "copy_count_histogram": _copy_histogram(copy_count),
        "membership_semantic_sha256": _semantic_sha256(membership),
        "owner_semantic_sha256": owner_source["owner_semantic_sha256"],
        "owner_upper_load_min": owner_source["upper_load_min"],
        "owner_upper_load_max": owner_source["upper_load_max"],
        "owner_upper_load_mean": owner_source["upper_load_mean"],
        "owner_upper_load_cv": owner_source["upper_load_cv"],
        "owner_upper_load_max_over_mean": owner_source[
            "upper_load_max_over_mean"
        ],
        **{f"owner_{key}": value for key, value in owner_topology.items()},
        **_load_metrics(loads),
    }
    result.update(
        query_metrics(
            owner,
            query_local,
            membership[labels],
            membership,
            ground_truth,
            dynamic_ef_base,
            dynamic_ef_factor,
        )
    )
    if extra:
        result.update(extra)
    return result


def canonical_cross_owner_gates(
    observed: dict[str, Any],
    reference: dict[str, Any],
    query_count: int,
) -> dict[str, dict[str, Any]]:
    def maximum(
        field: str, threshold: float, *, reference_value: float
    ) -> dict[str, Any]:
        value = float(observed[field])
        return {
            "observed": value,
            "operator": "<=",
            "threshold": threshold,
            "reference": reference_value,
            "pass": value <= threshold,
        }

    def minimum(
        field: str, threshold: float, *, reference_value: float
    ) -> dict[str, Any]:
        value = float(observed[field])
        return {
            "observed": value,
            "operator": ">=",
            "threshold": threshold,
            "reference": reference_value,
            "pass": value >= threshold,
        }

    ref_cut = float(reference["owner_upper_edge_cut_ratio"])
    ref_retained = float(reference["owner_retained_degree_mean"])
    ref_p10 = float(reference["owner_retained_degree_p10"])
    ref_isolated = float(reference["owner_upper_isolated_fraction"])
    ref_component_mean = float(reference["owner_largest_component_fraction_mean"])
    ref_component_min = float(reference["owner_largest_component_fraction_min"])
    gates = {
        "owner_upper_edge_cut_ratio": maximum(
            "owner_upper_edge_cut_ratio",
            ref_cut * EDGE_CUT_MAX_RATIO,
            reference_value=ref_cut,
        ),
        "owner_retained_degree_mean": minimum(
            "owner_retained_degree_mean",
            ref_retained * RETAINED_DEGREE_MIN_RATIO,
            reference_value=ref_retained,
        ),
        "owner_retained_degree_p10": minimum(
            "owner_retained_degree_p10",
            ref_p10 - RETAINED_DEGREE_P10_MAX_DROP,
            reference_value=ref_p10,
        ),
        "owner_upper_isolated_fraction": maximum(
            "owner_upper_isolated_fraction",
            min(
                ISOLATED_FRACTION_ABSOLUTE_MAX,
                ref_isolated + ISOLATED_FRACTION_MAX_DELTA,
            ),
            reference_value=ref_isolated,
        ),
        "owner_largest_component_fraction_mean": minimum(
            "owner_largest_component_fraction_mean",
            ref_component_mean - LARGEST_COMPONENT_MEAN_MAX_DROP,
            reference_value=ref_component_mean,
        ),
        "owner_largest_component_fraction_min": minimum(
            "owner_largest_component_fraction_min",
            LARGEST_COMPONENT_MIN_FLOOR,
            reference_value=ref_component_min,
        ),
        "gt_routing_coverage_mean": minimum(
            "gt_routing_coverage_mean",
            float(reference["gt_routing_coverage_mean"]) - MAX_COVERAGE_DROP,
            reference_value=float(reference["gt_routing_coverage_mean"]),
        ),
        "gt_queries_full_coverage_fraction": minimum(
            "gt_queries_full_coverage_fraction",
            float(reference["gt_queries_full_coverage_fraction"])
            - MAX_FULL_COVERAGE_DROP,
            reference_value=float(reference["gt_queries_full_coverage_fraction"]),
        ),
        "expansion_ratio": maximum(
            "expansion_ratio",
            float(reference["expansion_ratio"]),
            reference_value=float(reference["expansion_ratio"]),
        ),
        "physical_copy_load_max": maximum(
            "physical_copy_load_max",
            float(reference["physical_copy_load_max"]),
            reference_value=float(reference["physical_copy_load_max"]),
        ),
    }
    for field in (
        "query_owner_transitions_mean",
        "routed_shards_mean",
        "route_entry_points_mean",
        "route_ef_sum_mean",
    ):
        reference_value = float(reference[field])
        gates[field] = maximum(
            field,
            reference_value * MAX_ROUTE_WORK_RATIO,
            reference_value=reference_value,
        )
    zero_reference = int(reference["gt_queries_zero_coverage"])
    zero_limit = min(zero_reference + 2, max(1, int(query_count * 0.001)))
    zero_observed = int(observed["gt_queries_zero_coverage"])
    gates["gt_queries_zero_coverage"] = {
        "observed": zero_observed,
        "operator": "<=",
        "threshold": zero_limit,
        "reference": zero_reference,
        "pass": zero_observed <= zero_limit,
    }
    return gates


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    phase_b_path = Path(args.phase_b_screen).expanduser().resolve()
    phase_b_sha256 = sha256_path(phase_b_path)
    phase_b = _load_json(phase_b_path)
    phase_a_path = Path(phase_b["phase_a_manifest"]).expanduser().resolve()
    if sha256_path(phase_a_path) != phase_b["phase_a_manifest_sha256"]:
        raise ValueError("Phase-A manifest checksum drifted")
    phase_a = _load_json(phase_a_path)
    construction = phase_a["construction_inputs"]
    artifact_record = construction["artifact"]
    artifact_path = Path(artifact_record["path"]).expanduser().resolve()
    if sha256_path(artifact_path) != artifact_record["sha256"]:
        raise ValueError("upper artifact checksum drifted")

    artifact, _adj, _vectors, labels, _entry, edge_left, edge_right, navigator_sha = (
        load_upper(artifact_path)
    )
    upper_graph_sha256 = canonical_sha256(artifact["upper_graph"])
    num_partitions = int(artifact["shard_count"])
    if num_partitions != 32:
        raise ValueError("owner-policy matrix currently requires frozen P=32 inputs")
    dynamic_ef_base = int(artifact["dynamic_ef_base"])
    dynamic_ef_factor = int(artifact["dynamic_ef_factor"])

    cnbr_record = phase_a["owners"]["C_CNBR"]["owner"]
    cnbr_path = (phase_a_path.parent / cnbr_record["path"]).resolve()
    if sha256_path(cnbr_path) != cnbr_record["sha256"]:
        raise ValueError("C_CNBR owner checksum drifted")
    cnbr_owner = checked_binary_matrix(
        cnbr_path, rows=int(cnbr_record["row_count"]), width=1, dtype="<i4"
    ).reshape(-1)

    mass_record = phase_a["mass"]["values"]
    mass_path = (phase_a_path.parent / mass_record["path"]).resolve()
    if sha256_path(mass_path) != mass_record["sha256"]:
        raise ValueError("proxy-mass checksum drifted")
    proxy_mass = checked_binary_matrix(
        mass_path, rows=int(mass_record["row_count"]), width=1, dtype="<u8"
    ).reshape(-1)

    inputs = phase_b["post_freeze_evaluation_inputs"]
    for key in (
        "attachments",
        "attachments_manifest",
        "query_hits",
        "query_hits_manifest",
        "ground_truth",
        "ground_truth_manifest",
    ):
        path = Path(inputs[key]).expanduser().resolve()
        expected = inputs.get(f"{key}_sha256")
        if expected is not None and sha256_path(path) != expected:
            raise ValueError(f"bound input checksum drifted: {key}")

    attachment_manifest = _load_json(Path(inputs["attachments_manifest"]))
    logical_count = int(attachment_manifest["row_count"])
    attachment_hits = checked_binary_matrix(
        Path(inputs["attachments"]),
        rows=logical_count,
        width=int(attachment_manifest["top_k"]),
        dtype="<u8",
    )
    attachment_local = local_hits(attachment_hits, labels, logical_count)

    query_manifest = _load_json(Path(inputs["query_hits_manifest"]))
    query_total = int(query_manifest["row_count"])
    query_count = int(args.query_count) if int(args.query_count) > 0 else query_total
    if query_count <= 0 or query_count > query_total:
        raise ValueError("query-count is outside the bound query corpus")
    query_hits = checked_binary_matrix(
        Path(inputs["query_hits"]),
        rows=query_total,
        width=int(query_manifest["top_k"]),
        dtype="<u8",
    )
    query_local = local_hits(query_hits[:query_count], labels, query_count)
    gt_manifest = _load_json(Path(inputs["ground_truth_manifest"]))
    ground_truth = checked_binary_matrix(
        Path(inputs["ground_truth"]),
        rows=query_total,
        width=int(gt_manifest["width"]),
        dtype="<u4",
    )[:query_count]

    owners: list[tuple[str, np.ndarray, dict[str, Any]]] = [
        (
            "C_CNBR",
            np.asarray(cnbr_owner, dtype=np.int32),
            {
                "source_kind": "phase_a_frozen_binary_owner",
                "owner_binary": file_record(cnbr_path),
                "upper_navigator_sha256": navigator_sha,
                "historical_layout_reproduction_claim": True,
            },
        )
    ]
    for item in parse_named_paths(args.artifact_owner, args.owner_binary):
        if item.kind == "artifact":
            owner, source = artifact_owner(
                item.path,
                reference_labels=labels,
                reference_navigator_sha256=navigator_sha,
                num_partitions=num_partitions,
            )
        else:
            owner, source = binary_owner(
                item.path,
                reference_labels=labels,
                num_partitions=num_partitions,
                reference_artifact_sha256=artifact_record["sha256"],
                reference_upper_graph_sha256=upper_graph_sha256,
            )
            source["upper_navigator_sha256"] = navigator_sha
        owners.append((item.name, owner, source))

    rows: list[dict[str, Any]] = []
    owner_records: dict[str, Any] = {}
    fixed_policies = {policy.name: policy for policy in POLICIES}
    for owner_name, owner, source in owners:
        if len(owner) != len(labels) or np.any(owner < 0) or np.any(
            owner >= num_partitions
        ):
            raise ValueError(f"owner contract drifted: {owner_name}")
        upper_loads = np.bincount(owner, minlength=num_partitions)
        upper_mean = float(upper_loads.mean())
        source.update(
            {
                "owner_semantic_sha256": owner_semantic_sha256(labels, owner),
                "upper_load_min": int(upper_loads.min()),
                "upper_load_max": int(upper_loads.max()),
                "upper_load_mean": upper_mean,
                "upper_load_cv": float(upper_loads.std() / upper_mean),
                "upper_load_max_over_mean": float(upper_loads.max() / upper_mean),
            }
        )
        owner_topology = topology_metrics(
            owner, edge_left, edge_right, len(owner), num_partitions
        )
        owner_records[owner_name] = {**source, "topology_metrics": owner_topology}
        hit_owner = owner[attachment_local]
        votes, maximum, first_rank, rank_top, id_top = build_vote_evidence(
            hit_owner, num_partitions
        )
        owner_rows: list[dict[str, Any]] = []
        for policy_name in POLICY_NAMES:
            extra: dict[str, Any] = {}
            if policy_name == budgeted_policy.CANDIDATE_ID:
                assignment = budgeted_policy.build_budgeted_assignment(
                    owner=owner,
                    attachment_local=attachment_local,
                    proxy_mass=proxy_mass,
                    num_partitions=num_partitions,
                )
                membership = assignment.membership
                extra = {
                    "eligible_secondary_count": assignment.eligible_secondary_count,
                    "kept_secondary_count": assignment.kept_secondary_count,
                    "extra_copy_budget_cap": logical_count // 10,
                }
            else:
                membership = membership_for_policy(
                    hit_owner=hit_owner,
                    votes=votes,
                    maximum=maximum,
                    first_rank=first_rank,
                    rank_top=rank_top,
                    id_top=id_top,
                    policy=fixed_policies[policy_name],
                )
            if np.any(membership.sum(axis=1) < 1):
                raise AssertionError(f"{owner_name}+{policy_name} left points unassigned")
            if np.any(membership & ~(votes > 0)):
                raise AssertionError(
                    f"{owner_name}+{policy_name} emitted a copy without evidence"
                )
            row = assignment_metrics(
                owner_name=owner_name,
                policy_name=policy_name,
                membership=membership,
                owner=owner,
                labels=labels,
                query_local=query_local,
                ground_truth=ground_truth,
                dynamic_ef_base=dynamic_ef_base,
                dynamic_ef_factor=dynamic_ef_factor,
                owner_source=source,
                owner_topology=owner_topology,
                extra=extra,
            )
            owner_rows.append(row)
            print(
                f"{row['combination']} expansion={row['expansion_ratio']:.6f} "
                f"coverage={row['gt_routing_coverage_mean']:.6f} "
                f"routes={row['routed_shards_mean']:.6f}",
                flush=True,
            )

        baseline = owner_rows[0]
        for row in owner_rows:
            if row["policy"] == "current_all_max":
                row["gate_results"] = {}
                row["all_offline_gates_pass"] = True
                row["excess_expansion_reduction_fraction"] = 0.0
            else:
                row["gate_results"] = _gate_results(row, baseline, query_count)
                row["all_offline_gates_pass"] = all(
                    bool(gate["pass"]) for gate in row["gate_results"].values()
                )
                baseline_excess = float(baseline["expansion_ratio"]) - 1.0
                candidate_excess = float(row["expansion_ratio"]) - 1.0
                row["excess_expansion_reduction_fraction"] = (
                    float(1.0 - candidate_excess / baseline_excess)
                    if baseline_excess > 0
                    else 0.0
                )
        rows.extend(owner_rows)

    canonical_by_policy = {
        row["policy"]: row for row in rows if row["owner"] == "C_CNBR"
    }
    for row in rows:
        reference = canonical_by_policy[row["policy"]]
        if row["owner"] == "C_CNBR":
            row["canonical_cross_owner_gate_results"] = {}
            row["canonical_cross_owner_gates_pass"] = True
        else:
            cross_gates = canonical_cross_owner_gates(
                row, reference, query_count
            )
            row["canonical_cross_owner_gate_results"] = cross_gates
            row["canonical_cross_owner_gates_pass"] = all(
                bool(gate["pass"]) for gate in cross_gates.values()
            )

    output_dir.mkdir(parents=True)
    manifest = {
        "format_version": 1,
        "record_type": "orion_owner_multi_assignment_matrix_screen",
        "dataset": args.dataset,
        "query_count": query_count,
        "query_scope": "full" if query_count == query_total else "prefix",
        "phase_b_screen": file_record(phase_b_path),
        "source_upper_artifact": file_record(artifact_path),
        "upper_navigator_sha256": navigator_sha,
        "proxy_mass": file_record(mass_path),
        "logical_shard_count": num_partitions,
        "dynamic_ef_base": dynamic_ef_base,
        "dynamic_ef_factor": dynamic_ef_factor,
        "owner_order": [name for name, _owner, _source in owners],
        "policy_order": list(POLICY_NAMES),
        "owner_records": owner_records,
        "gate_contract": {
            "baseline_within_each_owner": "current_all_max",
            "canonical_cross_owner_reference": "C_CNBR with the same policy",
            "maximum_gt_coverage_drop": MAX_COVERAGE_DROP,
            "maximum_full_coverage_drop": MAX_FULL_COVERAGE_DROP,
            "maximum_route_work_ratio": MAX_ROUTE_WORK_RATIO,
            "maximum_edge_cut_ratio": EDGE_CUT_MAX_RATIO,
            "minimum_retained_degree_ratio": RETAINED_DEGREE_MIN_RATIO,
        },
        "interpretation_boundary": {
            "online_qps_inferred_from_offline_metrics": False,
            "online_matched_recall_ab_required": True,
            "historical_first_membership_is_new_research_candidate": True,
        },
        "rows": rows,
    }
    (output_dir / "screen-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scalar_fields = sorted(
        {key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))}
    )
    with (output_dir / "summary.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in scalar_fields} for row in rows)


if __name__ == "__main__":
    main()
