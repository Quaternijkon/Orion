from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "experiments/l1_balance/materialize_native_cnbr_candidate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_native_cnbr_candidate_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_direct_script_cli_help_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--arm {N_native,C_CNBR}" in completed.stdout
    assert "--validate-only" in completed.stdout
    assert "--construction-cost-audit" in completed.stdout


def test_selection_manifest_is_required_only_for_candidate(monkeypatch) -> None:
    native_args = materializer.parse_args(
        [
            "--phase-b-screen",
            "screen.json",
            "--arm",
            "N_native",
            "--validate-only",
        ]
    )
    assert native_args.selection_manifest is None

    class ValidationProbe(Exception):
        pass

    def probe(screen, arm, selection, construction_cost):
        assert screen == "screen.json"
        assert arm == "N_native"
        assert selection is None
        assert construction_cost is None
        raise ValidationProbe

    monkeypatch.setattr(materializer, "validate_phase_b_screen", probe)
    with pytest.raises(ValidationProbe):
        materializer.run(native_args)

    candidate_args = materializer.parse_args(
        [
            "--phase-b-screen",
            "screen.json",
            "--arm",
            "C_CNBR",
            "--validate-only",
        ]
    )
    with pytest.raises(ValueError, match="requires --selection-manifest"):
        materializer.run(candidate_args)

    candidate_without_cost = materializer.parse_args(
        [
            "--phase-b-screen",
            "screen.json",
            "--selection-manifest",
            "selection.json",
            "--arm",
            "C_CNBR",
            "--validate-only",
        ]
    )
    with pytest.raises(ValueError, match="requires --construction-cost-audit"):
        materializer.run(candidate_without_cost)


def test_arm_enum_and_balance_variants_have_no_aliases() -> None:
    assert set(materializer.ARM_CONTRACTS) == {"N_native", "C_CNBR"}
    assert {
        name: contract.balance_variant
        for name, contract in materializer.ARM_CONTRACTS.items()
    } == {"N_native": "natural_orion", "C_CNBR": "cnbr"}
    with pytest.raises(SystemExit):
        materializer.parse_args(
            [
                "--phase-b-screen",
                "screen.json",
                "--selection-manifest",
                "selection.json",
                "--arm",
                "native_unconstrained_kmeans",
                "--validate-only",
            ]
        )


def test_compact_membership_replays_exact_enabled_2_0_0_rule() -> None:
    owner = np.asarray([0, 0, 1, 1], dtype=np.int32)
    local_hits = np.asarray(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 0],
            [2, 3, 0, 2],
            [0, 2, 0, 2],
        ],
        dtype=np.int32,
    )
    membership, copies, primary = materializer.compact_membership(
        owner, local_hits, 2
    )
    assert membership.tolist() == [
        [True, True],
        [True, False],
        [False, True],
        [True, True],
    ]
    assert copies.tolist() == [2, 1, 1, 2]
    assert primary.tolist() == [0, 0, 1, 0]


def test_v1_screen_is_rejected_without_source_bound_v2_schema(
    tmp_path: Path,
) -> None:
    screen_path = tmp_path / "screen-manifest.json"
    write_json(
        screen_path,
        {
            "format_version": 1,
            "stage": materializer.phase_b.PHASE_B_STAGE,
            "candidate_set": "formal",
            "comparison": "N_native_vs_C_CNBR",
            "outputs": {
                "screen_manifest": "screen-manifest.json",
                "screen_manifest_sidecar": "screen-manifest.json.sha256",
                "summary_csv": "summary.csv",
                "summary_json": "summary.json",
            },
        },
    )
    screen_path.with_name(screen_path.name + ".sha256").write_text(
        sha256(screen_path) + "\n", encoding="ascii"
    )
    for path in (screen_path, screen_path.with_name(screen_path.name + ".sha256")):
        os.chmod(path, 0o444)
    with pytest.raises(ValueError, match="v2 output/source schema drifted"):
        materializer.validate_phase_b_screen(
            screen_path, "N_native", tmp_path / "selection.json", None
        )


def test_source_snapshot_binds_frozen_bytes_to_current_runtime_source(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current.py"
    current.write_text("VALUE = 1\n", encoding="utf-8")
    frozen_dir = tmp_path / "frozen"
    source_dir = frozen_dir / "source"
    source_dir.mkdir(parents=True)
    frozen = source_dir / "current.py"
    frozen.write_bytes(current.read_bytes())
    file_binding = {
        "path": "source/current.py",
        "sha256": sha256(frozen),
        "size_bytes": frozen.stat().st_size,
    }
    record_path = frozen_dir / "source.record.json"
    write_json(
        record_path,
        {
            "format_version": 1,
            "record_type": "fixture_source",
            "files": {"unit": file_binding},
        },
    )
    value = {
        "record": {
            "path": record_path.name,
            "sha256": sha256(record_path),
            "size_bytes": record_path.stat().st_size,
        },
        "files": {"unit": file_binding},
    }
    for path in (frozen, record_path):
        os.chmod(path, 0o444)
    assert materializer._validate_source_snapshot(
        base_dir=frozen_dir,
        value=value,
        label="fixture",
        record_type="fixture_source",
        expected_current_sources={"unit": current},
        runtime_bound=False,
    ) == sha256(record_path)
    current.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the frozen evaluator"):
        materializer._validate_source_snapshot(
            base_dir=frozen_dir,
            value=value,
            label="fixture",
            record_type="fixture_source",
            expected_current_sources={"unit": current},
            runtime_bound=False,
        )


def parity_record(
    owner: np.ndarray, local_hits: np.ndarray
) -> dict[str, object]:
    membership, copy_count, primary = materializer.compact_membership(
        owner, local_hits, 2
    )
    digest = hashlib.sha256()
    for point_id, row in enumerate(membership):
        digest.update(
            materializer.assignment_bytes(point_id, np.flatnonzero(row))
        )
    shard_counts = membership.sum(axis=0, dtype=np.int64)
    primary_loads = np.bincount(primary, minlength=2)
    values, counts = np.unique(copy_count, return_counts=True)
    histogram = {
        str(int(value)): int(count)
        for value, count in zip(values, counts, strict=True)
    }
    return {
        "canonical_format": materializer.ASSIGNMENT_FORMAT,
        "assignment_bytes_sha256": digest.hexdigest(),
        "logical_point_count": len(local_hits),
        "physical_point_count": int(copy_count.sum()),
        "primary_shards_sha256": hashlib.sha256(
            np.ascontiguousarray(primary, dtype="<i4").tobytes()
        ).hexdigest(),
        "primary_shard_loads": [int(value) for value in primary_loads],
        "physical_copy_shard_loads": [int(value) for value in shard_counts],
        "copy_count_histogram": histogram,
    }


@pytest.fixture()
def bundle_fixture(tmp_path: Path) -> dict[str, object]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact_path = source_dir / "generation-1.json"
    artifact = {
        "format_version": 1,
        "generation": 1,
        "logical_point_count": 4,
        "physical_point_count": 4,
        "shard_count": 2,
        "upper_k": 2,
        "upper_ef_search": 4,
        "dynamic_ef_base": 5,
        "dynamic_ef_factor": 2,
        "layout_sha256": "1" * 64,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "upper_nodes": [
            {
                "label": point_id,
                "vector": [float(point_id), float(point_id + 1)],
                "shard_membership": [point_id // 2],
            }
            for point_id in range(4)
        ],
        "upper_graph": {
            "entry_point": 0,
            "max_level": 0,
            "nodes": [
                {"label": point_id, "neighbors_by_level": [[point_id ^ 1]]}
                for point_id in range(4)
            ],
        },
    }
    write_json(artifact_path, artifact)

    vectors_path = source_dir / "vectors.f32le"
    np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype="<f4",
    ).tofile(vectors_path)
    os.chmod(vectors_path, 0o644)
    query_path = source_dir / "queries.f32le"
    np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f4").tofile(query_path)
    dataset_path = source_dir / "dataset.hdf5"
    dataset_path.write_bytes(b"dataset")
    dataset_manifest_path = source_dir / "dataset-manifest.json"
    write_json(
        dataset_manifest_path,
        {
            "dataset": {"name": "fixture", "dimension": 2},
        },
    )
    source_build_path = source_dir / "build-manifest.json"
    write_json(
        source_build_path,
        {
            "parameters": {
                "upper_sample_seed": 100,
                "upper_m": 32,
                "upper_ef_construction": 100,
                "upper_graph_seed": 100,
            }
        },
    )
    attachments_path = source_dir / "attachments.u64le"
    attachments_path.write_bytes(b"attachments")
    attachments_manifest_path = source_dir / "attachments.manifest.json"
    write_json(attachments_manifest_path, {"format_version": 1})
    phase_a_manifest_path = source_dir / "phase-a.manifest.json"
    write_json(phase_a_manifest_path, {"format_version": 1})
    screen_path = source_dir / "screen-manifest.json"
    write_json(screen_path, {"format_version": 1})

    owner_n_path = source_dir / "N_native.owner.i32le"
    owner_c_path = source_dir / "C_CNBR.owner.i32le"
    owner_n = np.asarray([0, 0, 1, 1], dtype="<i4")
    owner_c = np.asarray([0, 1, 0, 1], dtype="<i4")
    owner_n.tofile(owner_n_path)
    owner_c.tofile(owner_c_path)
    owner_n_record_path = source_dir / "N_native.owner-record.json"
    owner_c_record_path = source_dir / "C_CNBR.owner-record.json"
    write_json(owner_n_record_path, {"name": "N_native"})
    write_json(owner_c_record_path, {"name": "C_CNBR"})
    for path in (
        owner_n_path,
        owner_c_path,
        owner_n_record_path,
        owner_c_record_path,
    ):
        os.chmod(path, 0o444)

    local_hits = np.asarray(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 0],
            [2, 3, 0, 2],
            [0, 2, 0, 2],
        ],
        dtype=np.int32,
    )
    owners = {
        "N_native": SimpleNamespace(
            path=owner_n_path,
            values=owner_n,
            sha256=sha256(owner_n_path),
            owner_record_path=owner_n_record_path,
            owner_record_sha256=sha256(owner_n_record_path),
            record={
                "partition_sizes": [2, 2],
                "topology_metrics": {"upper_edge_cut_ratio": 0.5},
            },
        ),
        "C_CNBR": SimpleNamespace(
            path=owner_c_path,
            values=owner_c,
            sha256=sha256(owner_c_path),
            owner_record_path=owner_c_record_path,
            owner_record_sha256=sha256(owner_c_record_path),
            record={
                "partition_sizes": [2, 2],
                "topology_metrics": {"upper_edge_cut_ratio": 0.4},
            },
        ),
    }
    frozen = SimpleNamespace(
        artifact=artifact,
        artifact_path=artifact_path,
        labels=np.arange(4, dtype="<u8"),
        num_partitions=2,
        manifest={
            "fixed_parameters": {"upper_subset_denominator": 32},
            "construction_inputs": {
                "upper_only_projection": {
                    "upper_build_provenance": {
                        "source_manifest_sha256": sha256(source_build_path),
                        "source_manifest_size_bytes": source_build_path.stat().st_size,
                        "parameters": {
                            "upper_sample_seed": 100,
                            "upper_m": 32,
                            "upper_ef_construction": 100,
                            "upper_graph_seed": 100,
                        },
                    }
                }
            },
        },
        manifest_path=phase_a_manifest_path,
        manifest_sha256=sha256(phase_a_manifest_path),
        owners=owners,
    )
    inputs = SimpleNamespace(
        attachment_local=local_hits,
        attachments_sha256=sha256(attachments_path),
        attachments_path=attachments_path,
        attachments_manifest_path=attachments_manifest_path,
    )
    forbidden = materializer.phase_a.phase_a_forbidden_stage_invocations()

    def binding(arm_name: str):
        contract = materializer.ARM_CONTRACTS[arm_name]
        owner = owners[arm_name]
        record = {
            "name": arm_name,
            "identity_all_pass": True,
            "topology_all_pass": True,
            "topology_metrics": owner.record["topology_metrics"],
            "graph_topology_gates": {},
            "query_topology_gates": {},
            "physical_copy_load_improves_over_reference": True,
            "materialization_parity": parity_record(
                owner.values, local_hits
            ),
        }
        return materializer.MaterializationBinding(
            arm=contract,
            screen_path=screen_path,
            screen_sha256=sha256(screen_path),
            screen={"format_version": 1},
            record=record,
            frozen=frozen,
            inputs=inputs,
            owner=owner,
            source_artifact_path=artifact_path,
            source_artifact_sha256=sha256(artifact_path),
            vectors_path=vectors_path,
            vectors_sha256=sha256(vectors_path),
            source_build_manifest_path=source_build_path,
            source_build_manifest_sha256=sha256(source_build_path),
            dataset_manifest_path=dataset_manifest_path,
            dataset_manifest_sha256=sha256(dataset_manifest_path),
            dataset_path=dataset_path,
            dataset_sha256=sha256(dataset_path),
            replay_queries_path=query_path,
            replay_query_rows=2,
            forbidden_stage_invocations=forbidden,
            evaluator_source_code_record_sha256="2" * 64,
            selection_path=screen_path,
            selection_sha256=sha256(screen_path),
            selection_source_code_record_sha256="3" * 64,
            construction_cost_path=(
                screen_path if arm_name == "C_CNBR" else None
            ),
            construction_cost_sha256=(
                sha256(screen_path) if arm_name == "C_CNBR" else None
            ),
            construction_cost_gate=(
                {
                    "status": "PASS",
                    "audit": str(screen_path),
                    "audit_sha256": sha256(screen_path),
                }
                if arm_name == "C_CNBR"
                else None
            ),
        )

    rebind_binary = tmp_path / "orion_rebind_memberships"
    replay_verifier = tmp_path / "orion_verify_upper_replay"
    for path in (rebind_binary, replay_verifier):
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        os.chmod(path, 0o755)
    return {
        "binding": binding,
        "source_vector": vectors_path,
        "rebind_binary": rebind_binary,
        "replay_verifier": replay_verifier,
    }


def fake_subprocess_run(command: list[str], **_kwargs):
    executable = Path(command[0]).name
    if executable == "orion_rebind_memberships":
        source_path = Path(command[1])
        sidecar_path = Path(command[2])
        output_path = Path(command[3])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        source["generation"] = sidecar["generation"]
        source["layout_sha256"] = sidecar["layout_sha256"]
        source["physical_point_count"] = sidecar["physical_point_count"]
        for node, memberships in zip(
            source["upper_nodes"], sidecar["shard_memberships"], strict=True
        ):
            node["shard_membership"] = memberships
        write_json(output_path, source)
        output_path.with_name(output_path.name + ".sha256").write_text(
            sha256(output_path) + "\n", encoding="ascii"
        )
        return subprocess.CompletedProcess(command, 0, "rebind-pass\n", "")
    if executable == "orion_verify_upper_replay":
        source_path = Path(command[1])
        rebound_path = Path(command[2])
        query_path = Path(command[3])
        query_rows = int(command[4])
        dimension = int(command[5])
        replay_path = Path(command[6])
        manifest_path = Path(command[7])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        rebound = json.loads(rebound_path.read_text(encoding="utf-8"))
        replay_path.write_bytes(b"replay")
        replay_sha = sha256(replay_path)
        graph_sha = materializer.canonical_json_sha256(source["upper_graph"])
        node_sha = "a" * 64

        def artifact_record(path: Path, value: dict) -> dict:
            return {
                "path": str(path),
                "file_sha256": sha256(path),
                "canonical_artifact_sha256": "b" * 64,
                "generation": value["generation"],
                "layout_sha256": value["layout_sha256"],
                "shard_count": value["shard_count"],
                "physical_point_count": value["physical_point_count"],
                "canonical_upper_graph_sha256": graph_sha,
                "canonical_upper_graph_size_bytes": 123,
                "ordered_upper_nodes_identity_sha256": node_sha,
            }

        write_json(
            manifest_path,
            {
                "format_version": 1,
                "verdict": "PASS",
                "source": artifact_record(source_path, source),
                "rebound": artifact_record(rebound_path, rebound),
                "query_corpus": {
                    "path": str(query_path),
                    "sha256": sha256(query_path),
                    "row_count": query_rows,
                    "dimension": dimension,
                    "size_bytes": query_rows * dimension * 4,
                },
                "replay": {
                    "path": str(replay_path),
                    "encoding": "fixture",
                    "sha256": replay_sha,
                    "size_bytes": replay_path.stat().st_size,
                    "upper_k": source["upper_k"],
                    "compared_hit_count": 1,
                    "ordered_label_and_distance_bits_sha256": replay_sha,
                },
                "gates": {
                    gate: True for gate in materializer.REPLAY_GATES
                },
                "elapsed_seconds": 0.01,
            },
        )
        return subprocess.CompletedProcess(command, 0, "replay-pass\n", "")
    raise AssertionError(f"unexpected executable {executable}")


@pytest.mark.parametrize(
    ("arm_name", "generation", "variant"),
    [
        ("N_native", 2, "natural_orion"),
        ("C_CNBR", 3, "cnbr"),
    ],
)
def test_fixture_bundle_has_canonical_n_or_c_contract(
    bundle_fixture: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm_name: str,
    generation: int,
    variant: str,
) -> None:
    monkeypatch.setattr(materializer.subprocess, "run", fake_subprocess_run)
    binding = bundle_fixture["binding"](arm_name)
    output = tmp_path / f"bundle-{arm_name}"
    result = materializer.materialize_bundle(
        binding,
        generation=generation,
        rebind_binary=bundle_fixture["rebind_binary"],
        replay_verifier=bundle_fixture["replay_verifier"],
        output_dir=output,
        chunk_size=2,
    )

    manifest = json.loads(
        (output / "build-manifest.json").read_text(encoding="utf-8")
    )
    diagnostics = manifest["routing"]["l1_partition_diagnostics"]
    assert manifest["tool"] == materializer.TOOL_PATH
    assert diagnostics["balance_contract"] == {"variant": variant}
    assert diagnostics["arm"] == arm_name
    assert diagnostics["phase_a_owner_sha256"] == binding.owner.sha256
    assert diagnostics["owner_sha256"] == binding.owner.sha256
    assert diagnostics["phase_a_owner_record_sha256"] == (
        binding.owner.owner_record_sha256
    )
    assert diagnostics["phase_b_manifest_sha256"] == binding.screen_sha256
    assert diagnostics["attachments_sha256"] == (
        binding.inputs.attachments_sha256
    )
    assert diagnostics["forbidden_stage_invocations"] == (
        materializer.phase_a.phase_a_forbidden_stage_invocations()
    )
    assert diagnostics["invocation_counts"] == (
        materializer.MATERIALIZER_INVOCATION_COUNTS
    )
    assert diagnostics["multi_assignment_after_owner_freeze"] is True
    assert manifest["parameters"]["upper_sample_seed"] == 100
    assert manifest["parameters"]["upper_m"] == 32
    assert manifest["parameters"]["upper_ef_construction"] == 100
    assert manifest["parameters"]["upper_graph_seed"] == 100
    assert manifest["provenance"]["upper_build_parameter_provenance"] == (
        binding.frozen.manifest["construction_inputs"][
            "upper_only_projection"
        ]["upper_build_provenance"]
    )
    if arm_name == "C_CNBR":
        assert diagnostics["parent_n_owner_sha256"] == (
            binding.frozen.owners["N_native"].sha256
        )
        assert diagnostics["parent_n_manifest_sha256"] == (
            binding.frozen.manifest_sha256
        )
        assert diagnostics["construction_cost_v4_gate"] == "PASS"
        assert diagnostics["construction_cost_v4_audit_sha256"] == (
            binding.construction_cost_sha256
        )
        assert manifest["provenance"][
            "construction_cost_v4_audit_sha256"
        ] == binding.construction_cost_sha256
        assert manifest["provenance"]["construction_cost_v4_gate"][
            "status"
        ] == "PASS"
    else:
        assert "parent_n_owner_sha256" not in diagnostics
        assert "parent_n_manifest_sha256" not in diagnostics

    assert result["balance_variant"] == variant
    assert os.path.samefile(
        bundle_fixture["source_vector"],
        output / "orion_numeric_import.f32le",
    )
    assert stat.S_IMODE(bundle_fixture["source_vector"].stat().st_mode) == 0o644
    assert (
        stat.S_IMODE(
            (output / "orion_numeric_import.f32le").stat().st_mode
        )
        == 0o644
    )
    for path in output.iterdir():
        if path.is_file() and path.name != "orion_numeric_import.f32le":
            assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_parity_drift_fails_before_rebind(
    bundle_fixture: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = bundle_fixture["binding"]("C_CNBR")
    drifted_record = copy.deepcopy(binding.record)
    drifted_record["materialization_parity"][
        "assignment_bytes_sha256"
    ] = "f" * 64
    binding = materializer.MaterializationBinding(
        **{**binding.__dict__, "record": drifted_record}
    )
    calls: list[list[str]] = []

    def forbidden_run(command: list[str], **_kwargs):
        calls.append(command)
        raise AssertionError("rebind must not run after parity drift")

    monkeypatch.setattr(materializer.subprocess, "run", forbidden_run)
    output = tmp_path / "parity-drift"
    with pytest.raises(
        ValueError, match="golden Phase-B multi-assignment parity mismatch"
    ):
        materializer.materialize_bundle(
            binding,
            generation=2,
            rebind_binary=bundle_fixture["rebind_binary"],
            replay_verifier=bundle_fixture["replay_verifier"],
            output_dir=output,
            chunk_size=2,
        )
    assert calls == []
    assert not (output / "orion_numeric_import.assignments.jsonl").exists()
    assert not (
        output / "orion_numeric_import.assignments.jsonl.tmp"
    ).exists()


def test_existing_output_is_never_overwritten(
    bundle_fixture: dict[str, object], tmp_path: Path
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materializer.materialize_bundle(
            bundle_fixture["binding"]("N_native"),
            generation=2,
            rebind_binary=bundle_fixture["rebind_binary"],
            replay_verifier=bundle_fixture["replay_verifier"],
            output_dir=output,
            chunk_size=2,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_build_parameters_use_only_projection_provenance(
    bundle_fixture: dict[str, object],
) -> None:
    binding = bundle_fixture["binding"]("N_native")
    source_build = json.loads(
        binding.source_build_manifest_path.read_text(encoding="utf-8")
    )
    source_build["parameters"] = {}
    write_json(binding.source_build_manifest_path, source_build)

    parameters = materializer._build_parameters(binding, generation=2)
    assert {
        key: parameters[key]
        for key in materializer.phase_a.UPPER_BUILD_PARAMETER_KEYS
    } == {
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "upper_graph_seed": 100,
    }


@pytest.mark.parametrize("missing_key", materializer.phase_a.UPPER_BUILD_PARAMETER_KEYS)
def test_build_parameters_fail_closed_without_complete_projection_provenance(
    bundle_fixture: dict[str, object], missing_key: str
) -> None:
    binding = bundle_fixture["binding"]("N_native")
    frozen_manifest = copy.deepcopy(binding.frozen.manifest)
    del frozen_manifest["construction_inputs"]["upper_only_projection"][
        "upper_build_provenance"
    ]["parameters"][missing_key]
    frozen = SimpleNamespace(
        **{**binding.frozen.__dict__, "manifest": frozen_manifest}
    )
    drifted = materializer.MaterializationBinding(
        **{**binding.__dict__, "frozen": frozen}
    )

    with pytest.raises(ValueError, match="parameter whitelist drifted"):
        materializer._build_parameters(drifted, generation=2)
