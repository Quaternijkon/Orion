from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "experiments/l1_balance/prepare_native_cnbr_phase_a.py"
SPEC = importlib.util.spec_from_file_location("prepare_native_cnbr_phase_a_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)

PROJECTION_PATH = REPO_ROOT / "experiments/l1_balance/project_upper_only_artifact.py"
PROJECTION_SPEC = importlib.util.spec_from_file_location(
    "project_upper_only_artifact_test", PROJECTION_PATH
)
assert PROJECTION_SPEC is not None and PROJECTION_SPEC.loader is not None
projection = importlib.util.module_from_spec(PROJECTION_SPEC)
sys.modules[PROJECTION_SPEC.name] = projection
PROJECTION_SPEC.loader.exec_module(projection)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def upper_inputs(tmp_path: Path) -> dict[str, Path]:
    node_count = 64
    dimension = 2
    labels = np.arange(100_000, 100_000 + node_count, dtype="<u8")
    vectors = np.asarray(
        [
            [float(cluster * 10), float(cluster * cluster)]
            for cluster in range(32)
            for _duplicate in range(2)
        ],
        dtype="<f4",
    )
    upper_nodes = [
        {
            "label": int(labels[node]),
            "vector": [float(value) for value in vectors[node]],
            "shard_membership": [node // 2],
        }
        for node in range(node_count)
    ]
    graph_nodes = [
        {
            "label": int(labels[node]),
            "neighbors_by_level": [[int(labels[node ^ 1])]],
        }
        for node in range(node_count)
    ]
    artifact = {
        "format_version": 1,
        "generation": 1,
        "logical_point_count": node_count * 32,
        "physical_point_count": node_count * 32,
        "shard_count": 32,
        "upper_k": 48,
        "upper_ef_search": 48,
        "dynamic_ef_base": 50,
        "dynamic_ef_factor": 14,
        "layout_sha256": "0" * 64,
        "vector_schema": {
            "vector_name": "",
            "dimension": dimension,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "upper_nodes": upper_nodes,
        "upper_graph": {
            "entry_point": int(labels[0]),
            "max_level": 0,
            "nodes": graph_nodes,
        },
    }
    source_artifact_path = tmp_path / "source-generation-1.json"
    _write_json(source_artifact_path, artifact)
    source_build_path = tmp_path / "source-build.json"
    _write_json(
        source_build_path,
        {
            "mode": "production_bundle",
            "outputs": {
                "files": {
                    source_artifact_path.name: {
                        "sha256": _sha256(source_artifact_path),
                        "size_bytes": source_artifact_path.stat().st_size,
                    }
                },
                "production_artifact": source_artifact_path.name,
            },
            "parameters": {
                "upper_sample_seed": 101,
                "upper_m": 32,
                "upper_ef_construction": 200,
                "upper_graph_seed": 303,
                "balance_mode": "capacity_constrained",
            }
        },
    )
    artifact_path = tmp_path / "generation-1.upper-only.json"
    projection_manifest_path = tmp_path / "generation-1.upper-only.manifest.json"
    projection.run(
        argparse.Namespace(
            source_artifact=str(source_artifact_path),
            source_build_manifest=str(source_build_path),
            output_artifact=str(artifact_path),
            output_manifest=str(projection_manifest_path),
        )
    )

    vectors_path = tmp_path / "upper-vectors.f32le"
    labels_path = tmp_path / "upper-labels.u64le"
    vectors.tofile(vectors_path)
    labels.tofile(labels_path)
    upper_manifest_path = tmp_path / "upper-input.manifest.json"
    _write_json(
        upper_manifest_path,
        {
            "format_version": 1,
            "artifact": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path),
            "generation": 1,
            "row_count": node_count,
            "dimension": dimension,
            "vectors": str(vectors_path),
            "vectors_sha256": _sha256(vectors_path),
            "vectors_size_bytes": vectors_path.stat().st_size,
            "labels": str(labels_path),
            "labels_sha256": _sha256(labels_path),
            "labels_size_bytes": labels_path.stat().st_size,
        },
    )

    hits = np.empty((node_count, 10), dtype="<u8")
    for node in range(node_count):
        local = [node] + [(node + offset) % node_count for offset in range(1, 10)]
        hits[node] = labels[np.asarray(local, dtype=np.int32)]
    hits_path = tmp_path / "l1-self-hits-top10.u64le"
    hits.tofile(hits_path)
    navigation_manifest_path = tmp_path / "l1-self-hits-top10.manifest.json"
    _write_json(
        navigation_manifest_path,
        {
            "format_version": 1,
            "artifact_path": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path),
            "generation": 1,
            "upper_graph_present": True,
            "source_upper_k": 48,
            "source_upper_ef_search": 48,
            "vectors_path": str(vectors_path),
            "vectors_sha256": _sha256(vectors_path),
            "row_count": node_count,
            "dimension": dimension,
            "top_k": 10,
            "search_ef": 100,
            "hits_path": str(hits_path),
            "hits_sha256": _sha256(hits_path),
            "hits_size_bytes": hits_path.stat().st_size,
        },
    )
    return {
        "source_artifact": source_artifact_path,
        "source_build": source_build_path,
        "artifact": artifact_path,
        "projection_manifest": projection_manifest_path,
        "upper_manifest": upper_manifest_path,
        "hits": hits_path,
        "navigation_manifest": navigation_manifest_path,
    }


def _args(
    inputs: dict[str, Path], output: Path, candidate_set: str = "formal"
) -> argparse.Namespace:
    return argparse.Namespace(
        artifact=str(inputs["artifact"]),
        upper_only_projection_manifest=str(inputs["projection_manifest"]),
        upper_input_manifest=str(inputs["upper_manifest"]),
        upper_navigation_hits=str(inputs["hits"]),
        upper_navigation_manifest=str(inputs["navigation_manifest"]),
        candidate_set=candidate_set,
        output_dir=str(output),
    )


def test_cli_has_only_upper_inputs_and_closed_candidate_set() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    help_text = completed.stdout
    for option in (
        "--attachments",
        "--point-to-l1s",
        "--multi-assignment",
        "--query-hits",
        "--ground-truth",
        "--observed-load",
        "--trigger-ratio",
        "--capacity",
    ):
        assert option not in help_text
    assert "--candidate-set {formal,frozen-grid}" in help_text
    assert "--upper-only-projection-manifest" in help_text


def test_phase_a_rejects_artifact_that_still_contains_historical_memberships(
    upper_inputs: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="non-neutral historical shard membership"):
        prepare._load_production_upper(upper_inputs["source_artifact"])


def test_formal_phase_a_is_byte_deterministic_checksum_bound_and_read_only(
    upper_inputs: dict[str, Path], tmp_path: Path
) -> None:
    first_dir = tmp_path / "formal-a"
    second_dir = tmp_path / "formal-b"
    first_manifest, first_sha = prepare.run(_args(upper_inputs, first_dir))
    second_manifest, second_sha = prepare.run(_args(upper_inputs, second_dir))

    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert first_sha == second_sha == _sha256(first_manifest)
    assert (first_dir / "phase-a.manifest.json.sha256").read_text().strip() == first_sha
    assert (first_dir / "owners/N_native.owner.i32le").read_bytes() == (
        second_dir / "owners/N_native.owner.i32le"
    ).read_bytes()
    assert (first_dir / "owners/C_CNBR.owner.i32le").read_bytes() == (
        second_dir / "owners/C_CNBR.owner.i32le"
    ).read_bytes()

    manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
    assert manifest["stage"] == "upper_only_n_native_cnbr_owners_frozen"
    assert manifest["contract"] == prepare.phase_a_contract()
    assert manifest["fixed_parameters"] == prepare.phase_a_fixed_parameters("formal")
    assert manifest["graph_gate_contract"] == prepare.phase_a_graph_gate_contract()
    assert set(manifest["outputs"]) == set(prepare.phase_a_output_keys("formal"))
    assert set(manifest["owners"]) == {"N_native", "C_CNBR"}
    native = manifest["owners"]["N_native"]
    candidate = manifest["owners"]["C_CNBR"]
    assert native["role"] == "balance_free_orion_core"
    assert candidate["role"] == "replacement_candidate"
    assert candidate["base_owner_sha256"] == native["owner"]["sha256"]
    assert candidate["parent_n_owner_record_sha256"] == native[
        "phase_a_owner_record_sha256"
    ]
    assert "parent_n_manifest_sha256" not in candidate
    assert candidate["trigger_ratio"] == {"numerator": 9, "denominator": 4}
    assert isinstance(candidate["rounds"], list)
    assert candidate["round_trace"]["path"].startswith("traces/")
    assert manifest["mass"]["mode"] == (
        "cnbr-upper-self-navigation-raw-occurrence-v1"
    )
    assert manifest["mass"]["total_mass"] == 640
    assert "estimated_partition_proxy_masses" in native
    assert "estimated_partition_masses" not in native
    assert set(manifest["forbidden_stage_invocations"]) == set(
        prepare.FORBIDDEN_STAGE_INVOCATIONS
    )
    assert all(value == 0 for value in manifest["forbidden_stage_invocations"].values())
    projection_binding = manifest["construction_inputs"]["upper_only_projection"]
    assert set(projection_binding["source_artifact"]) == {
        "sha256",
        "size_bytes",
        "role",
    }
    assert projection_binding["upper_build_provenance"]["parameters"] == {
        "upper_sample_seed": 101,
        "upper_m": 32,
        "upper_ef_construction": 200,
        "upper_graph_seed": 303,
    }

    performance = json.loads(
        (first_dir / "phase-a.performance.json").read_text(encoding="utf-8")
    )
    assert performance["phase_a_manifest_sha256"] == first_sha
    assert (
        "required_input_validation_and_mass_replay_wall_seconds"
        in performance["shared"]
    )
    assert (
        "required_input_validation_and_mass_replay_process_cpu_seconds"
        in performance["shared"]
    )
    assert (
        "self_navigation_input_validation_and_mass_replay_wall_seconds"
        not in performance["shared"]
    )
    assert "upper_self_navigation_wall_seconds" not in performance["shared"]
    assert performance["timing_contract"] == {
        "contract_id": "cnbr-integrated-incremental-cost-v1",
        "cost_gate_includes": [
            "external_upper_self_navigation",
            "required_input_validation_and_mass_replay",
            "formal_cnbr",
        ],
        "required_input_validation_start": (
            "after shared upper artifact deserialization and graph normalization"
        ),
        "required_input_validation_includes": [
            "projected_artifact_sha256",
            "upper_only_projection_provenance_and_redaction",
            "upper_input_manifest_and_exact_byte_parity",
            "self_navigation_manifest_and_row_validation",
            "proxy_mass_replay_and_duplicate_tie_proof",
        ],
        "reported_but_excluded_from_incremental_cost_gate": [
            "evidence_preflight",
            "shared_upper_input_materialization",
            "n_native_kmeans",
            "n_native_reference_topology_validation",
            "post_cnbr_source_stability_validation",
            "evidence_serialization_and_freeze",
        ],
    }
    for field in (
        "evidence_preflight_wall_seconds",
        "shared_upper_input_materialization_wall_seconds",
        "n_native_reference_topology_validation_wall_seconds",
        "post_cnbr_source_stability_validation_wall_seconds",
    ):
        assert performance["shared"][field] >= 0.0

    for path in first_dir.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_grid_phase_a_freezes_exact_six_ratio_family(
    upper_inputs: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "grid"
    manifest_path, _sha = prepare.run(
        _args(upper_inputs, output, candidate_set="frozen-grid")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "N_native",
        "CNBR_9_4",
        "CNBR_2_1",
        "CNBR_7_4",
        "CNBR_8_5",
        "CNBR_3_2",
        "CNBR_7_5",
    }
    assert manifest["stage"] == "upper_only_cnbr_frozen_family_owners_frozen"
    assert set(manifest["owners"]) == expected
    selected = [
        name
        for name, record in manifest["owners"].items()
        if record.get("selected_adoption_candidate")
    ]
    assert selected == ["CNBR_9_4"]
    for name in expected - {"N_native"}:
        record = manifest["owners"][name]
        assert record["role"] == "frozen_cnbr_family_member"
        assert record["base_owner_sha256"] == manifest["owners"]["N_native"][
            "owner"
        ]["sha256"]
        assert record["round_trace"]["sha256"] == _sha256(
            output / record["round_trace"]["path"]
        )


@pytest.mark.parametrize(
    ("candidate_set", "expected_prepared_builds"),
    [("formal", 1), ("frozen-grid", 6)],
)
def test_phase_a_prepares_navigation_once_and_never_uses_legacy_cnbr_paths(
    upper_inputs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_set: str,
    expected_prepared_builds: int,
) -> None:
    original_prepare = prepare.core.prepare_upper_navigation
    original_prepared_builder = prepare.core.build_frozen_cnbr_family_owner_prepared
    calls = {"prepare": 0, "prepared_builder": 0}

    def prepare_spy(*args: object, **kwargs: object) -> object:
        calls["prepare"] += 1
        return original_prepare(*args, **kwargs)

    def prepared_builder_spy(*args: object, **kwargs: object) -> object:
        calls["prepared_builder"] += 1
        return original_prepared_builder(*args, **kwargs)

    def reject_legacy_path(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Phase-A must not fall back to a legacy navigation path")

    monkeypatch.setattr(prepare.core, "prepare_upper_navigation", prepare_spy)
    monkeypatch.setattr(
        prepare.core,
        "build_frozen_cnbr_family_owner_prepared",
        prepared_builder_spy,
    )
    monkeypatch.setattr(
        prepare.core,
        "estimate_upper_self_navigation_hit_frequency_v1",
        reject_legacy_path,
    )
    monkeypatch.setattr(prepare.core, "build_c_cnbr_owner", reject_legacy_path)
    monkeypatch.setattr(
        prepare.core,
        "build_frozen_cnbr_family_owner",
        reject_legacy_path,
    )

    prepare.run(
        _args(
            upper_inputs,
            tmp_path / f"prepared-only-{candidate_set}",
            candidate_set=candidate_set,
        )
    )

    assert calls == {
        "prepare": 1,
        "prepared_builder": expected_prepared_builds,
    }


def test_phase_a_fails_closed_on_input_checksum_drift(
    upper_inputs: dict[str, Path], tmp_path: Path
) -> None:
    original = json.loads(upper_inputs["upper_manifest"].read_text(encoding="utf-8"))
    original["vectors_sha256"] = "f" * 64
    bad_manifest = tmp_path / "upper-input.bad.json"
    _write_json(bad_manifest, original)
    bad_inputs = {**upper_inputs, "upper_manifest": bad_manifest}
    with pytest.raises(ValueError, match="vectors_sha256 mismatch"):
        prepare.run(_args(bad_inputs, tmp_path / "bad-output"))
    assert not (tmp_path / "bad-output").exists()


def test_phase_a_rejects_projection_provenance_with_non_whitelisted_field(
    upper_inputs: dict[str, Path], tmp_path: Path
) -> None:
    projection_manifest = json.loads(
        upper_inputs["projection_manifest"].read_text(encoding="utf-8")
    )
    projection_manifest["upper_build_provenance"]["parameters"][
        "balance_mode"
    ] = "capacity_constrained"
    bad_projection = tmp_path / "upper-only.bad.manifest.json"
    _write_json(bad_projection, projection_manifest)
    bad_projection.with_name(bad_projection.name + ".sha256").write_text(
        _sha256(bad_projection) + "\n", encoding="ascii"
    )
    bad_inputs = {**upper_inputs, "projection_manifest": bad_projection}

    with pytest.raises(ValueError, match="parameter whitelist drifted"):
        prepare.run(_args(bad_inputs, tmp_path / "bad-provenance"))
    assert not (tmp_path / "bad-provenance").exists()
