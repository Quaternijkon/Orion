from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "experiments/l1_balance/materialize_candidate.py"
    spec = importlib.util.spec_from_file_location("l1_balance_materializer_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


materializer = load_module()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_compact_membership_is_exact_original_2_0_0_rule() -> None:
    owner = np.asarray([0, 0, 1, 1], dtype=np.int32)
    local_hits = np.asarray(
        [
            [0, 1, 2, 3],  # two votes each: multi-assign both shards
            [0, 1, 2, 0],  # three votes for shard zero
            [2, 3, 0, 2],  # three votes for shard one
            [0, 2, 0, 2],  # tie at two: multi-assign both shards
            [0, 2, 2, 0],  # tie at two: multi-assign both shards
        ],
        dtype=np.int32,
    )

    membership, copies = materializer.compact_membership(owner, local_hits, 2)

    assert membership.tolist() == [
        [True, True],
        [True, False],
        [False, True],
        [True, True],
        [True, True],
    ]
    assert copies.tolist() == [2, 1, 1, 2, 2]


def test_assignment_bytes_match_evaluator_canonical_format() -> None:
    assert materializer.assignment_bytes(17, [0, 3, 31]) == (
        b'{"id":17,"shards":[0,3,31]}\n'
    )


def parity_binding() -> dict:
    return {
        "candidate": {
            "actual_metrics": {
                "physical_point_count": 5,
                "expansion_ratio": 1.25,
                "load_min": 2,
                "load_max": 3,
                "load_mean": 2.5,
                "load_cv": 0.2,
                "load_max_over_mean": 1.2,
                "load_min_over_mean": 0.8,
                "empty_shards": 0,
            }
        },
        "materialization_parity": {
            "canonical_format": materializer.ASSIGNMENT_FORMAT,
            "assignment_bytes_sha256": "a" * 64,
            "logical_point_count": 4,
            "physical_point_count": 5,
            "shard_loads": [3, 2],
            "copy_count_histogram": {"1": 3, "2": 1},
        },
    }


def test_golden_parity_accepts_exact_membership_bytes_loads_and_histogram() -> None:
    materializer.validate_materialization_parity(
        parity_binding(),
        layout_sha256="a" * 64,
        rows=4,
        physical_count=5,
        shard_counts=np.asarray([3, 2], dtype=np.int64),
        copy_histogram=Counter({1: 3, 2: 1}),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("assignment_bytes_sha256", "b" * 64),
        ("shard_loads", [2, 3]),
        ("copy_count_histogram", {"1": 1, "2": 2}),
    ],
)
def test_golden_parity_fails_closed_on_any_drift(field: str, value: object) -> None:
    binding = parity_binding()
    binding["materialization_parity"][field] = value
    with pytest.raises(ValueError, match="golden multi-assignment parity mismatch"):
        materializer.validate_materialization_parity(
            binding,
            layout_sha256="a" * 64,
            rows=4,
            physical_count=5,
            shard_counts=np.asarray([3, 2], dtype=np.int64),
            copy_histogram=Counter({1: 3, 2: 1}),
        )


def topology_candidate() -> dict:
    return {
        "topology_metrics": {"upper_edge_cut_ratio": 0.51},
        "actual_metrics": {"routed_shards_mean": 5.2},
        "topology_gates": {
            "upper_edge_cut_ratio": {
                "observed": 0.51,
                "operator": "<=",
                "threshold": 0.55,
                "pass": True,
            },
            "routed_shards_mean": {
                "category": "search_induced",
                "observed": 5.2,
                "operator": "<=",
                "threshold": 5.5,
                "pass": True,
            },
        },
        "topology_all_pass": True,
        "graph_topology_all_pass": True,
        "topology_gate_inputs_complete": True,
        "identity_all_pass": True,
        "identity_gates": {"upper_graph_sha256": True, "navigator_sha256": True},
    }


def test_topology_gate_replay_includes_search_induced_metrics() -> None:
    materializer.validate_topology_record(
        topology_candidate(),
        {
            "all_gates_required": True,
            "preregistered_before_mass_candidate_results": True,
        },
        {"all_gates_required": True, "inputs_complete": True},
    )


def test_topology_gate_cannot_be_overridden_by_all_pass_boolean() -> None:
    candidate = topology_candidate()
    candidate["topology_gates"]["routed_shards_mean"]["observed"] = 5.6
    candidate["actual_metrics"]["routed_shards_mean"] = 5.6
    with pytest.raises(ValueError, match="pass bit is inconsistent"):
        materializer.validate_topology_record(
            candidate,
            {
                "all_gates_required": True,
                "preregistered_before_mass_candidate_results": True,
            },
            {"all_gates_required": True, "inputs_complete": True},
        )


def vector_source_fixture(tmp_path: Path) -> dict:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    vectors_path = source_dir / "orion_numeric_import.f32le"
    vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.6, 0.8]],
        dtype="<f4",
    )
    vectors.tofile(vectors_path)
    dataset_path = tmp_path / "dataset.hdf5"
    dataset_path.write_bytes(b"source-dataset")
    dataset_sha = materializer.sha256_path(dataset_path)
    vector_sha = materializer.sha256_path(vectors_path)
    build_path = source_dir / "build-manifest.json"
    build = {
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": dataset_sha,
            "dimension": 2,
            "train_rows_total": 4,
            "train_rows_used": 4,
        },
        "parameters": {"vector_distance": "cosine"},
        "outputs": {
            "files": {
                vectors_path.name: {
                    "sha256": vector_sha,
                    "size_bytes": vectors_path.stat().st_size,
                }
            }
        },
    }
    write_json(build_path, build)
    checksums_path = source_dir / "checksums.sha256"
    checksums_path.write_text(
        f"{materializer.sha256_path(build_path)}  {build_path.name}\n"
        f"{vector_sha}  {vectors_path.name}\n",
        encoding="ascii",
    )
    return {
        "artifact": {
            "logical_point_count": 4,
            "vector_schema": {"dimension": 2, "distance": "Cosine"},
        },
        "vectors": vectors_path,
        "vector_sha": vector_sha,
        "build": build_path,
        "dataset": dataset_path,
        "dataset_sha": dataset_sha,
    }


def test_canonical_vectors_are_bound_to_dataset_order_and_cosine_preprocessing(
    tmp_path: Path,
) -> None:
    fixture = vector_source_fixture(tmp_path)
    proof = materializer.validate_vector_source_contract(
        vectors_path=fixture["vectors"],
        vectors_sha256=fixture["vector_sha"],
        source_build_manifest_path=fixture["build"],
        dataset_path=fixture["dataset"],
        dataset_sha256=fixture["dataset_sha"],
        artifact=fixture["artifact"],
        chunk_rows=2,
    )

    assert proof["row_count"] == 4
    assert proof["preprocessing_contract"] == (
        "cosine_l2_normalize_nonzero_rows_float32"
    )
    assert proof["vectors_sha256"] == fixture["vector_sha"]


def test_vector_source_rejects_dataset_row_count_drift(tmp_path: Path) -> None:
    fixture = vector_source_fixture(tmp_path)
    build = json.loads(fixture["build"].read_text(encoding="utf-8"))
    build["dataset"]["train_rows_used"] = 3
    write_json(fixture["build"], build)
    checksums = fixture["build"].with_name("checksums.sha256")
    checksums.write_text(
        f"{materializer.sha256_path(fixture['build'])}  build-manifest.json\n"
        f"{fixture['vector_sha']}  {fixture['vectors'].name}\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="row-order binding mismatch"):
        materializer.validate_vector_source_contract(
            vectors_path=fixture["vectors"],
            vectors_sha256=fixture["vector_sha"],
            source_build_manifest_path=fixture["build"],
            dataset_path=fixture["dataset"],
            dataset_sha256=fixture["dataset_sha"],
            artifact=fixture["artifact"],
            chunk_rows=2,
        )


def test_attachment_export_must_use_exact_hardlinked_vector_rows(tmp_path: Path) -> None:
    artifact_path = tmp_path / "generation-1.json"
    vectors_path = tmp_path / "vectors.f32le"
    attachments_path = tmp_path / "attachments.u64le"
    manifest_path = tmp_path / "attachments.manifest.json"
    vectors = np.arange(8, dtype="<f4")
    vectors.tofile(vectors_path)
    hits = np.tile(np.arange(10, dtype="<u8"), (4, 1))
    hits.tofile(attachments_path)
    artifact_path.write_text("{}\n", encoding="utf-8")
    artifact = {
        "generation": 1,
        "logical_point_count": 4,
        "upper_k": 3,
        "upper_ef_search": 4,
        "vector_schema": {"dimension": 2},
    }
    source_sha = materializer.sha256_path(artifact_path)
    vector_sha = materializer.sha256_path(vectors_path)
    write_json(
        manifest_path,
        {
            "format_version": 1,
            "artifact_path": str(artifact_path.resolve()),
            "artifact_sha256": source_sha,
            "generation": 1,
            "upper_graph_present": True,
            "source_upper_k": 3,
            "source_upper_ef_search": 4,
            "vectors_path": str(vectors_path.resolve()),
            "vectors_sha256": vector_sha,
            "row_count": 4,
            "dimension": 2,
            "top_k": 10,
            "search_ef": 100,
            "hits_path": str(attachments_path.resolve()),
            "hits_sha256": materializer.sha256_path(attachments_path),
            "hits_size_bytes": attachments_path.stat().st_size,
        },
    )

    manifest, mapped = materializer.validate_attachment_contract(
        artifact=artifact,
        source_artifact_path=artifact_path.resolve(),
        source_sha256=source_sha,
        attachments_path=attachments_path.resolve(),
        attachments_manifest_path=manifest_path.resolve(),
        vectors_path=vectors_path.resolve(),
        vectors_sha256=vector_sha,
    )

    assert manifest["row_count"] == artifact["logical_point_count"]
    assert mapped.shape == (4, 10)


def v2_allowlist_records() -> tuple[dict, dict, dict]:
    construction = {
        "artifact_sha256": "a" * 64,
        "partitioner_source_sha256": "b" * 64,
    }
    mass = {
        "format_version": 1,
        "mass_mode": materializer.MASS_MODE,
        "source": materializer.MASS_SOURCE,
        "transform": materializer.MASS_TRANSFORM,
        "estimator_version": materializer.MASS_ESTIMATOR_VERSION,
        "eligible_as_finalist": True,
        "rank_weighting": False,
        "manifest_sha256": "c" * 64,
        "sha256": "d" * 64,
    }
    candidate = {
        "method": materializer.MASS_METHOD,
        "balance_contract": materializer.MASS_BALANCE_CONTRACT,
        "source_artifact_sha256": construction["artifact_sha256"],
        "partitioner_source_sha256": construction["partitioner_source_sha256"],
        "navigation_mass_manifest_sha256": mass["manifest_sha256"],
        "navigation_mass_source": mass["source"],
        "navigation_mass_transform": mass["transform"],
        "navigation_mass_estimator_version": mass["estimator_version"],
        "navigation_mass_sha256": mass["sha256"],
        "mass_mode": mass["mass_mode"],
        "materialization_eligible": True,
    }
    return candidate, mass, construction


def cross_dataset_contract(amendment: dict) -> dict:
    return {
        "format_version": 1,
        "stage": materializer.CROSS_DATASET_STAGE,
        "status": materializer.CROSS_DATASET_STATUS,
        "materialization_eligible": True,
        "cross_dataset_identity_all_pass": True,
        "cross_dataset_topology_all_pass": True,
        "owner_parity_all_pass": True,
        "online_qps_confirmation_required": True,
        "protocol_amendment": amendment,
        "method_contract": {
            "estimator_version": materializer.MASS_ESTIMATOR_VERSION,
            "l0_evaluation_only_after_owner_freeze": True,
            "mass_mode": materializer.MASS_MODE,
            "method": materializer.MASS_METHOD,
            "multi_assignment_unchanged": True,
            "source": materializer.MASS_SOURCE,
            "transform": materializer.MASS_TRANSFORM,
            "upper_only_phase_a": True,
        },
        "confirmation_contract": {
            "alternate_method_or_parameter_scan": False,
            "old_raw_owner_byte_parity_required": True,
            "online_cluster_entered_by_this_stage": False,
            "online_qps_confirmation_completed": False,
            "online_qps_confirmation_required": True,
            "online_qps_only_confirmation": True,
            "post_exploratory_not_preregistered": True,
            "topology_thresholds_unchanged": True,
            "v1_heldout_failure_retained": True,
        },
    }


def test_cross_dataset_contract_keeps_online_qps_as_pending_final_gate() -> None:
    amendment = {"id": materializer.PROTOCOL_AMENDMENT_ID}
    materializer.validate_cross_dataset_contract(
        cross_dataset_contract(amendment), amendment
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("top", "cross_dataset_topology_all_pass", False),
        ("confirmation_contract", "online_qps_confirmation_completed", True),
        ("method_contract", "multi_assignment_unchanged", False),
    ],
)
def test_cross_dataset_contract_fails_closed_on_promotion_drift(
    section: str, field: str, value: object
) -> None:
    amendment = {"id": materializer.PROTOCOL_AMENDMENT_ID}
    confirmation = cross_dataset_contract(amendment)
    if section == "top":
        confirmation[field] = value
    else:
        confirmation[section][field] = value
    with pytest.raises(ValueError, match="cross-dataset confirmation contract mismatch"):
        materializer.validate_cross_dataset_contract(confirmation, amendment)


def test_only_exact_raw_regularized_v2_tuple_is_allowlisted() -> None:
    candidate, mass, construction = v2_allowlist_records()
    materializer.validate_mass_candidate_allowlist(candidate, mass, construction)


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("candidate", "method", "mass-ldg"),
        ("candidate", "mass_mode", "raw"),
        (
            "candidate",
            "navigation_mass_source",
            "production_upper_navigation_top10_raw_hit_frequency",
        ),
        ("candidate", "navigation_mass_estimator_version", 1),
        ("mass", "mass_mode", "self-debiased-floor1"),
        ("mass", "estimator_version", 1),
    ],
)
def test_old_raw_floor1_or_non_bkm_candidates_cannot_use_v2_allowlist(
    target: str,
    field: str,
    value: object,
) -> None:
    candidate, mass, construction = v2_allowlist_records()
    (candidate if target == "candidate" else mass)[field] = value
    with pytest.raises(ValueError, match="candidate navigation-mass|estimator mismatch"):
        materializer.validate_mass_candidate_allowlist(candidate, mass, construction)
