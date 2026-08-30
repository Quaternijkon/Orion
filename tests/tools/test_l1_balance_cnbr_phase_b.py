from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance import evaluate_cnbr_frozen_owners as evaluate
from experiments.l1_balance import freeze_phase_b_source_binding as source_binding
from experiments.l1_balance import prepare_native_cnbr_phase_a as prepare
from experiments.l1_balance import project_upper_only_artifact as projection
from experiments.l1_balance import synthesize_cnbr_dual_dataset_selection as selection


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rewrite_frozen_json(path: Path, value: object, *, sidecar: bool = False) -> None:
    os.chmod(path, 0o644)
    _write_json(path, value)
    os.chmod(path, 0o444)
    if sidecar:
        sidecar_path = path.with_name(path.name + ".sha256")
        os.chmod(sidecar_path, 0o644)
        sidecar_path.write_text(_sha256(path) + "\n", encoding="ascii")
        os.chmod(sidecar_path, 0o444)


def _tiny_case(
    tmp_path: Path,
    candidate_set: str = "formal",
    *,
    dataset_name: str = "tiny",
    vector_mode: str = "canonical-exact",
) -> argparse.Namespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if vector_mode not in {"canonical-exact", "cosine-normalized"}:
        raise ValueError(f"unsupported test vector mode: {vector_mode}")
    node_count = 64
    logical_rows = 2_048
    dimension = 2
    query_rows = 4
    labels = np.arange(node_count, dtype="<u8")
    raw_upper_vectors = np.asarray(
        [
            [float(cluster * 10), float(cluster * cluster)]
            for cluster in range(32)
            for _duplicate in range(2)
        ],
        dtype="<f4",
    )
    canonical_vectors = raw_upper_vectors[np.arange(logical_rows) % node_count]
    if vector_mode == "cosine-normalized":
        norms = np.linalg.norm(canonical_vectors.astype(np.float64), axis=1)
        full_vectors = canonical_vectors.copy()
        nonzero = norms > 1e-12
        full_vectors[nonzero] = (
            canonical_vectors[nonzero].astype(np.float64)
            / norms[nonzero, np.newaxis]
        ).astype("<f4")
        upper_vectors = full_vectors[:node_count].copy()
        distance = "Cosine"
        metric = "cosine"
    else:
        full_vectors = canonical_vectors
        upper_vectors = raw_upper_vectors
        distance = "Euclid"
        metric = "euclid"

    canonical_vectors_path = tmp_path / "vectors.f32le"
    canonical_vectors.astype("<f4", copy=False).tofile(canonical_vectors_path)
    if vector_mode == "cosine-normalized":
        vectors_path = tmp_path / "orion_numeric_import.f32le"
        full_vectors.astype("<f4", copy=False).tofile(vectors_path)
    else:
        vectors_path = canonical_vectors_path

    dataset_path = tmp_path / "tiny-dataset.hdf5"
    dataset_path.write_bytes(
        f"checksum-bound {dataset_name} dataset fixture\n".encode("utf-8")
    )
    dataset_sha256 = _sha256(dataset_path)
    upper_nodes = [
        {
            "label": int(labels[node]),
            "vector": [float(value) for value in upper_vectors[node]],
            "shard_membership": [node // 2],
        }
        for node in range(node_count)
    ]
    graph_nodes = []
    for node in range(node_count):
        neighbors = sorted({node ^ 1, (node + 2) % node_count, (node - 2) % node_count})
        graph_nodes.append(
            {
                "label": int(labels[node]),
                "neighbors_by_level": [[int(labels[value]) for value in neighbors]],
            }
        )
    artifact = {
        "format_version": 1,
        "generation": 1,
        "logical_point_count": logical_rows,
        "physical_point_count": logical_rows,
        "shard_count": 32,
        "upper_k": 48,
        "upper_ef_search": 48,
        "dynamic_ef_base": 50,
        "dynamic_ef_factor": 14,
        "layout_sha256": "0" * 64,
        "vector_schema": {
            "vector_name": "",
            "dimension": dimension,
            "distance": distance,
            "datatype": "float32",
        },
        "upper_nodes": upper_nodes,
        "upper_graph": {
            "entry_point": 0,
            "max_level": 0,
            "nodes": graph_nodes,
        },
    }
    source_artifact_path = tmp_path / "source-generation-1.json"
    _write_json(source_artifact_path, artifact)
    import_manifest_path = tmp_path / "orion_numeric_import.manifest.json"
    if vector_mode == "cosine-normalized":
        _write_json(
            import_manifest_path,
            {
                "format_version": 1,
                "orion_artifact_file": source_artifact_path.name,
                "orion_artifact_sha256": _sha256(source_artifact_path),
                "point_count": logical_rows,
                "dimension": dimension,
                "vectors_file": vectors_path.name,
                "vectors_sha256": _sha256(vectors_path),
            },
        )
    projection_source_build_path = tmp_path / "projection-source-build.json"
    source_build_files = {
        source_artifact_path.name: {
            "path": str(source_artifact_path),
            "sha256": _sha256(source_artifact_path),
            "size_bytes": source_artifact_path.stat().st_size,
        },
        vectors_path.name: {
            "path": str(vectors_path),
            "sha256": _sha256(vectors_path),
            "size_bytes": vectors_path.stat().st_size,
        },
    }
    source_build_outputs = {
        "files": source_build_files,
        "production_artifact": source_artifact_path.name,
    }
    if vector_mode == "cosine-normalized":
        source_build_files[import_manifest_path.name] = {
            "path": str(import_manifest_path),
            "sha256": _sha256(import_manifest_path),
            "size_bytes": import_manifest_path.stat().st_size,
        }
        source_build_outputs["import_manifest"] = import_manifest_path.name
    source_build_parameters = {
        "upper_sample_seed": 101,
        "upper_m": 32,
        "upper_ef_construction": 200,
        "upper_graph_seed": 303,
        "balance_mode": "capacity_constrained",
    }
    if vector_mode == "cosine-normalized":
        source_build_parameters["vector_distance"] = metric
    _write_json(
        projection_source_build_path,
        {
            "format_version": 1,
            "mode": "production_bundle",
            "dataset": {
                "path": str(dataset_path),
                "sha256": dataset_sha256,
                "dimension": dimension,
                "train_rows_total": logical_rows,
                "train_rows_used": logical_rows,
            },
            "outputs": source_build_outputs,
            "parameters": source_build_parameters,
        },
    )
    if vector_mode == "cosine-normalized":
        (tmp_path / "checksums.sha256").write_text(
            f"{_sha256(projection_source_build_path)}  "
            f"{projection_source_build_path.name}\n"
            f"{_sha256(vectors_path)}  {vectors_path.name}\n"
            f"{_sha256(import_manifest_path)}  {import_manifest_path.name}\n",
            encoding="ascii",
        )
    artifact_path = tmp_path / "generation-1.upper-only.json"
    projection_manifest_path = tmp_path / "generation-1.upper-only.manifest.json"
    projection.run(
        argparse.Namespace(
            source_artifact=str(source_artifact_path),
            source_build_manifest=str(projection_source_build_path),
            output_artifact=str(artifact_path),
            output_manifest=str(projection_manifest_path),
        )
    )

    upper_vectors_path = tmp_path / "upper-vectors.f32le"
    upper_labels_path = tmp_path / "upper-labels.u64le"
    upper_vectors.tofile(upper_vectors_path)
    labels.tofile(upper_labels_path)
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
            "vectors": str(upper_vectors_path),
            "vectors_sha256": _sha256(upper_vectors_path),
            "vectors_size_bytes": upper_vectors_path.stat().st_size,
            "labels": str(upper_labels_path),
            "labels_sha256": _sha256(upper_labels_path),
            "labels_size_bytes": upper_labels_path.stat().st_size,
        },
    )

    self_hits = np.empty((node_count, 10), dtype="<u8")
    for node in range(node_count):
        row = [(node + offset) % node_count for offset in range(10)]
        if 10 <= node < 40 and 0 not in row:
            row[-1] = 0
        assert len(set(row)) == 10 and node in row
        self_hits[node] = labels[np.asarray(row, dtype=np.int32)]
    self_hits_path = tmp_path / "l1-self-hits-top10.u64le"
    self_hits.tofile(self_hits_path)
    self_manifest_path = tmp_path / "l1-self-hits-top10.manifest.json"
    _write_json(
        self_manifest_path,
        {
            "format_version": 1,
            "artifact_path": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path),
            "generation": 1,
            "upper_graph_present": True,
            "source_upper_k": 48,
            "source_upper_ef_search": 48,
            "vectors_path": str(upper_vectors_path),
            "vectors_sha256": _sha256(upper_vectors_path),
            "row_count": node_count,
            "dimension": dimension,
            "top_k": 10,
            "search_ef": 100,
            "hits_path": str(self_hits_path),
            "hits_sha256": _sha256(self_hits_path),
            "hits_size_bytes": self_hits_path.stat().st_size,
        },
    )

    phase_a_dir = tmp_path / f"phase-a-{candidate_set}"
    phase_a_manifest, _digest = prepare.run(
        argparse.Namespace(
            artifact=str(artifact_path),
            upper_only_projection_manifest=str(projection_manifest_path),
            upper_input_manifest=str(upper_manifest_path),
            upper_navigation_hits=str(self_hits_path),
            upper_navigation_manifest=str(self_manifest_path),
            candidate_set=candidate_set,
            output_dir=str(phase_a_dir),
        )
    )

    query_vectors = canonical_vectors[:query_rows]
    queries_path = tmp_path / "queries.f32le"
    query_vectors.astype("<f4", copy=False).tofile(queries_path)
    canonical_gt = np.asarray(
        [[(row * 17 + offset) % logical_rows for offset in range(10)] for row in range(query_rows)],
        dtype="<u4",
    )
    dataset_gt_path = tmp_path / "dataset-ground-truth.u32le"
    gt_alias_path = tmp_path / "ground-truth-top10.u32le"
    canonical_gt.tofile(dataset_gt_path)
    canonical_gt.tofile(gt_alias_path)
    dataset_identity = {
        "name": dataset_name,
        "sha256": dataset_sha256,
        "dimension": dimension,
        "expected_train_rows": logical_rows,
        "expected_query_rows": query_rows,
        "metric": metric,
        "hnsw_space": "l2" if metric == "euclid" else metric,
    }
    dataset_manifest_path = tmp_path / "dataset.manifest.json"
    _write_json(
        dataset_manifest_path,
        {
            "dataset": dataset_identity,
            "source_dataset": {
                "dataset": dataset_identity,
                "path": str(dataset_path),
                "sha256": dataset_sha256,
                "size_bytes": dataset_path.stat().st_size,
            },
            "files": {
                "vectors": {
                    "path": str(canonical_vectors_path),
                    "sha256": _sha256(canonical_vectors_path),
                    "size_bytes": canonical_vectors_path.stat().st_size,
                },
                "queries": {
                    "path": str(queries_path),
                    "sha256": _sha256(queries_path),
                    "size_bytes": queries_path.stat().st_size,
                },
                "ground_truth": {
                    "path": str(dataset_gt_path),
                    "sha256": _sha256(dataset_gt_path),
                    "size_bytes": dataset_gt_path.stat().st_size,
                },
            },
        },
    )
    source_build_path = tmp_path / "phase-b-source-binding.json"
    source_binding.run(
        argparse.Namespace(
            dataset_manifest=str(dataset_manifest_path),
            artifact=str(artifact_path),
            vectors=str(vectors_path),
            source_build_manifest=str(projection_source_build_path),
            upper_only_projection_manifest=str(projection_manifest_path),
            output=str(source_build_path),
        )
    )

    attachments = np.empty((logical_rows, 10), dtype="<u8")
    for point in range(logical_rows):
        start = point % node_count
        attachments[point] = labels[
            np.asarray([(start + offset) % node_count for offset in range(10)])
        ]
    attachments_path = tmp_path / "full-attachments-top10.u64le"
    attachments.tofile(attachments_path)
    attachments_manifest_path = tmp_path / "full-attachments-top10.manifest.json"
    _write_json(
        attachments_manifest_path,
        {
            "format_version": 1,
            "artifact_path": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path),
            "generation": 1,
            "vectors_path": str(vectors_path),
            "vectors_sha256": _sha256(vectors_path),
            "row_count": logical_rows,
            "dimension": dimension,
            "top_k": 10,
            "search_ef": 100,
            "hits_path": str(attachments_path),
            "hits_sha256": _sha256(attachments_path),
            "hits_size_bytes": attachments_path.stat().st_size,
        },
    )

    query_hits = np.empty((query_rows, 48), dtype="<u8")
    for row in range(query_rows):
        query_hits[row] = labels[
            np.asarray([(row + offset) % node_count for offset in range(48)])
        ]
    query_hits_path = tmp_path / "query-hits-top48.u64le"
    query_hits.tofile(query_hits_path)
    query_manifest_path = tmp_path / "query-hits-top48.manifest.json"
    _write_json(
        query_manifest_path,
        {
            "format_version": 1,
            "artifact_path": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path),
            "vectors_path": str(queries_path),
            "vectors_sha256": _sha256(queries_path),
            "row_count": query_rows,
            "dimension": dimension,
            "top_k": 48,
            "search_ef": 100,
            "hits_path": str(query_hits_path),
            "hits_sha256": _sha256(query_hits_path),
            "hits_size_bytes": query_hits_path.stat().st_size,
        },
    )
    gt_manifest_path = tmp_path / "ground-truth-top10.manifest.json"
    _write_json(
        gt_manifest_path,
        {
            "format_version": 1,
            "source_hdf5": str(dataset_path),
            "source_dataset": "neighbors",
            "output": str(gt_alias_path),
            "dtype": "<u4",
            "row_count": query_rows,
            "width": 10,
            "sha256": _sha256(gt_alias_path),
            "size_bytes": gt_alias_path.stat().st_size,
        },
    )
    return argparse.Namespace(
        phase_a_manifest=str(phase_a_manifest),
        dataset_manifest=str(dataset_manifest_path),
        source_build_manifest=str(source_build_path),
        attachments=str(attachments_path),
        attachments_manifest=str(attachments_manifest_path),
        row_count=logical_rows,
        attachment_k=10,
        query_hits=str(query_hits_path),
        query_hits_manifest=str(query_manifest_path),
        ground_truth=str(gt_alias_path),
        ground_truth_manifest=str(gt_manifest_path),
        ground_truth_width=10,
        output_dir=str(tmp_path / f"phase-b-{candidate_set}"),
    )


@pytest.mark.parametrize("candidate_set", ["formal", "frozen-grid"])
def test_phase_b_happy_path_freezes_sources_and_outputs(
    tmp_path: Path, candidate_set: str
) -> None:
    args = _tiny_case(tmp_path, candidate_set)
    evaluate.run(args)
    output_dir = Path(args.output_dir)
    screen_path = output_dir / "screen-manifest.json"
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    assert screen["candidate_set"] == candidate_set
    assert screen["reference"]["materialization_eligible"] is True
    assert screen["identity_gates"]["phase_b_evaluator_source_code_sha256"] is True
    source_record = screen["evaluator_source_code"]["record"]
    assert _sha256(output_dir / source_record["path"]) == source_record["sha256"]
    assert set(screen["evaluator_source_code"]["files"]) == {
        "evaluator",
        "offline_screen",
        "phase_a_generator",
        "phase_a_core",
        "source_binding",
    }
    if candidate_set == "frozen-grid":
        assert all(
            candidate["materialization_eligible"] is False
            for name, candidate in screen["candidates"].items()
            if name != "CNBR_9_4"
        )
    for path in output_dir.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluate.run(args)


def test_phase_a_failure_precedes_any_l0_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _tiny_case(tmp_path)
    manifest_path = Path(args.phase_a_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract"]["owners_immutable"] = False
    _rewrite_frozen_json(manifest_path, manifest, sidecar=True)
    opened = False

    def forbidden_open(*_args: object, **_kwargs: object) -> None:
        nonlocal opened
        opened = True
        raise AssertionError("Phase B input opened before Phase A passed")

    monkeypatch.setattr(evaluate, "open_phase_b_inputs", forbidden_open)
    with pytest.raises(ValueError, match="architecture contract"):
        evaluate.run(args)
    assert opened is False
    assert not Path(args.output_dir).exists()


def test_phase_b_rejects_drifted_upper_build_provenance(
    tmp_path: Path,
) -> None:
    args = _tiny_case(tmp_path)
    manifest_path = Path(args.phase_a_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["construction_inputs"]["upper_only_projection"][
        "upper_build_provenance"
    ]["parameters"]["upper_m"] = 16
    _rewrite_frozen_json(manifest_path, manifest, sidecar=True)

    with pytest.raises(ValueError, match="upper-only projection manifest drifted"):
        evaluate.validate_phase_a(args)


@pytest.mark.parametrize(
    "target",
    ["artifact", "source", "owner", "owner_record", "trace", "performance"],
)
def test_phase_a_checksum_drift_fails_closed(tmp_path: Path, target: str) -> None:
    args = _tiny_case(tmp_path)
    manifest_path = Path(args.phase_a_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase_a_dir = manifest_path.parent
    if target == "artifact":
        path = Path(manifest["construction_inputs"]["artifact"]["path"])
    elif target == "source":
        path = phase_a_dir / manifest["source_code"]["core"]["path"]
    elif target == "owner":
        path = phase_a_dir / manifest["owners"]["C_CNBR"]["owner"]["path"]
    elif target == "owner_record":
        path = phase_a_dir / manifest["owners"]["C_CNBR"]["owner_record"]["path"]
    elif target == "trace":
        path = phase_a_dir / manifest["owners"]["C_CNBR"]["round_trace"]["path"]
    else:
        path = phase_a_dir / "phase-a.performance.json"
    os.chmod(path, 0o644)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        evaluate.validate_phase_a(args)


def test_source_binding_and_gt_alias_contract(tmp_path: Path) -> None:
    args = _tiny_case(tmp_path)
    frozen = evaluate.validate_phase_a(args)
    inputs = evaluate.open_phase_b_inputs(args, frozen)
    dataset = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    assert Path(dataset["files"]["ground_truth"]["path"]) != inputs.ground_truth_path
    assert dataset["files"]["ground_truth"]["sha256"] == inputs.ground_truth_sha256

    sidecar = Path(args.source_build_manifest + ".sha256")
    os.chmod(sidecar, 0o644)
    sidecar.write_text("0" * 64 + "\n", encoding="ascii")
    os.chmod(sidecar, 0o444)
    with pytest.raises(ValueError, match="source-build checksum binding"):
        evaluate.open_phase_b_inputs(args, frozen)


@pytest.mark.parametrize(
    ("dataset_name", "vector_mode", "expected_mode", "expected_preprocessing"),
    [
        (
            "sift1m",
            "canonical-exact",
            "canonical_dataset_vectors_exact",
            "identity_float32_row_order",
        ),
        (
            "glove-200-angular",
            "cosine-normalized",
            "source_build_bound_cosine_normalized",
            "cosine_l2_normalize_nonzero_rows_float32",
        ),
    ],
)
def test_source_binding_replays_exact_and_cosine_normalized_lineage(
    tmp_path: Path,
    dataset_name: str,
    vector_mode: str,
    expected_mode: str,
    expected_preprocessing: str,
) -> None:
    args = _tiny_case(
        tmp_path,
        dataset_name=dataset_name,
        vector_mode=vector_mode,
    )
    binding = json.loads(
        Path(args.source_build_manifest).read_text(encoding="utf-8")
    )
    lineage = binding["vector_lineage"]
    assert lineage["mode"] == expected_mode
    assert lineage["preprocessing_contract"] == expected_preprocessing
    assert lineage["canonical_dataset_vectors"][
        "byte_identical_to_build_vectors"
    ] is (vector_mode == "canonical-exact")
    if vector_mode == "canonical-exact":
        assert lineage["checksums"] is None
        assert lineage["import_manifest"] is None
    else:
        assert lineage["checksums"]["role"] == "source_build_checksums"
        assert lineage["import_manifest"]["role"] == (
            "source_build_import_manifest"
        )

    frozen = evaluate.validate_phase_a(args)
    evaluate.open_phase_b_inputs(args, frozen)


@pytest.mark.parametrize(
    ("dataset_name", "vector_mode"),
    [
        ("sift1m", "canonical-exact"),
        ("glove-200-angular", "cosine-normalized"),
    ],
)
def test_source_binding_vector_byte_drift_fails_closed(
    tmp_path: Path, dataset_name: str, vector_mode: str
) -> None:
    args = _tiny_case(
        tmp_path,
        dataset_name=dataset_name,
        vector_mode=vector_mode,
    )
    frozen = evaluate.validate_phase_a(args)
    binding = json.loads(
        Path(args.source_build_manifest).read_text(encoding="utf-8")
    )
    vectors_path = Path(binding["vector_lineage"]["vectors"]["path"])
    vector_bytes = bytearray(vectors_path.read_bytes())
    vector_bytes[0] ^= 1
    vectors_path.write_bytes(vector_bytes)

    with pytest.raises(ValueError, match="source-build vector checksum lineage"):
        evaluate.open_phase_b_inputs(args, frozen)


def test_normalized_source_binding_requires_checksums_and_import_lineage(
    tmp_path: Path,
) -> None:
    args = _tiny_case(
        tmp_path,
        dataset_name="glove-200-angular",
        vector_mode="cosine-normalized",
    )
    frozen = evaluate.validate_phase_a(args)
    binding = json.loads(
        Path(args.source_build_manifest).read_text(encoding="utf-8")
    )
    lineage = binding["vector_lineage"]

    checksums_path = Path(lineage["checksums"]["path"])
    checksums_path.rename(checksums_path.with_name("checksums.removed"))
    with pytest.raises(ValueError, match="lack source-build checksums"):
        evaluate.open_phase_b_inputs(args, frozen)

    args = _tiny_case(
        tmp_path / "import-drift",
        dataset_name="glove-200-angular",
        vector_mode="cosine-normalized",
    )
    frozen = evaluate.validate_phase_a(args)
    binding = json.loads(
        Path(args.source_build_manifest).read_text(encoding="utf-8")
    )
    import_path = Path(binding["vector_lineage"]["import_manifest"]["path"])
    import_manifest = json.loads(import_path.read_text(encoding="utf-8"))
    import_manifest["point_count"] -= 1
    _write_json(import_path, import_manifest)
    with pytest.raises(
        ValueError, match="source-build import manifest checksum lineage"
    ):
        evaluate.open_phase_b_inputs(args, frozen)


def test_phase_b_rejects_legacy_source_build_without_lineage_binding(
    tmp_path: Path,
) -> None:
    args = _tiny_case(tmp_path)
    frozen = evaluate.validate_phase_a(args)
    binding = json.loads(
        Path(args.source_build_manifest).read_text(encoding="utf-8")
    )
    args.source_build_manifest = binding["vector_lineage"][
        "source_build_manifest"
    ]["path"]

    with pytest.raises(ValueError, match="requires a frozen source-build"):
        evaluate.open_phase_b_inputs(args, frozen)


def test_phase_b_replays_binding_instead_of_trusting_claimed_lineage(
    tmp_path: Path,
) -> None:
    args = _tiny_case(
        tmp_path,
        dataset_name="glove-200-angular",
        vector_mode="cosine-normalized",
    )
    frozen = evaluate.validate_phase_a(args)
    binding_path = Path(args.source_build_manifest)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["vector_lineage"]["preprocessing_contract"] = (
        "identity_float32_row_order"
    )
    _rewrite_frozen_json(binding_path, binding, sidecar=True)

    with pytest.raises(ValueError, match="binding vector lineage drifted"):
        evaluate.open_phase_b_inputs(args, frozen)


def test_source_binding_path_and_gt_sha_drift_fail(tmp_path: Path) -> None:
    args = _tiny_case(tmp_path)
    frozen = evaluate.validate_phase_a(args)
    binding_path = Path(args.source_build_manifest)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    artifact_name = frozen.artifact_path.name
    binding["outputs"]["files"][artifact_name]["path"] = str(
        binding_path.parent / "another-generation.json"
    )
    _rewrite_frozen_json(binding_path, binding, sidecar=True)
    with pytest.raises(ValueError, match="artifact path drifted"):
        evaluate.open_phase_b_inputs(args, frozen)

    args = _tiny_case(tmp_path / "gt-drift")
    frozen = evaluate.validate_phase_a(args)
    ground_truth_path = Path(args.ground_truth)
    ground_truth_path.write_bytes(ground_truth_path.read_bytes()[:-4] + b"\xff" * 4)
    with pytest.raises(ValueError, match="ground-truth manifest mismatch"):
        evaluate.open_phase_b_inputs(args, frozen)


def test_primary_and_physical_copy_golden_semantics() -> None:
    owner = np.asarray([0, 1, 2, 3], dtype=np.int32)
    attachments = np.asarray(
        [
            [0, 1, 2, 3],
            [2, 1, 1, 2],
            [3, 3, 1, 2],
        ],
        dtype=np.int64,
    )
    membership, copies, physical, primary, primary_loads = evaluate.assignment_views(
        owner, attachments, 4
    )
    assert primary.tolist() == [0, 1, 3]
    assert membership.tolist() == [
        [True, False, False, False],
        [False, True, True, False],
        [False, False, False, True],
    ]
    assert copies.tolist() == [1, 2, 1]
    assert physical.tolist() == [1, 1, 1, 1]
    assert primary_loads.tolist() == [1, 1, 0, 1]


def test_eligibility_is_fail_closed_for_load_graph_and_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _tiny_case(tmp_path)
    frozen = evaluate.validate_phase_a(args)
    inputs = evaluate.open_phase_b_inputs(args, frozen)
    rows, screen = evaluate.evaluate_phase_b(frozen, inputs)
    assert rows[0]["name"] == "N_native"
    assert screen["reference"]["materialization_eligible"] is True

    owners = dict(frozen.owners)
    owners["C_CNBR"] = replace(
        owners["C_CNBR"], values=owners["N_native"].values.copy()
    )
    equal_load = replace(frozen, owners=owners)
    _rows, equal_screen = evaluate.evaluate_phase_b(equal_load, inputs)
    assert equal_screen["candidate"]["physical_copy_load_gate"]["pass"] is False
    assert equal_screen["candidate"]["materialization_eligible"] is False

    monkeypatch.setattr(
        evaluate,
        "graph_gate_results",
        lambda *_args, **_kwargs: {"forced_graph_failure": {"pass": False}},
    )
    _rows, graph_screen = evaluate.evaluate_phase_b(frozen, inputs)
    assert graph_screen["candidate"]["graph_topology_all_pass"] is False
    assert graph_screen["candidate"]["materialization_eligible"] is False

    monkeypatch.undo()
    monkeypatch.setattr(
        evaluate,
        "query_gate_results",
        lambda *_args, **_kwargs: {"forced_query_failure": {"pass": False}},
    )
    _rows, query_screen = evaluate.evaluate_phase_b(frozen, inputs)
    assert query_screen["candidate"]["query_topology_all_pass"] is False
    assert query_screen["candidate"]["materialization_eligible"] is False


def test_evaluator_never_rebuilds_an_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _tiny_case(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("owner construction was invoked by Phase B")

    monkeypatch.setattr(prepare.core, "build_n_native_owner", forbidden)
    monkeypatch.setattr(prepare.core, "build_frozen_cnbr_family_owner", forbidden)
    evaluate.run(args)
    assert Path(args.output_dir, "screen-manifest.json").is_file()


def test_mid_run_phase_b_source_drift_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _tiny_case(tmp_path)
    copied_dir = tmp_path / "phase-b-source-copies"
    copied_dir.mkdir()
    copied_sources: dict[str, Path] = {}
    for name, original in evaluate._phase_b_source_paths().items():
        copied = copied_dir / f"{name}-{original.name}"
        copied.write_bytes(original.read_bytes())
        copied_sources[name] = copied.resolve()
    monkeypatch.setattr(evaluate, "_phase_b_source_paths", lambda: copied_sources)
    original_evaluate = evaluate.evaluate_phase_b

    def drift_after_evaluation(*call_args: object, **call_kwargs: object):
        result = original_evaluate(*call_args, **call_kwargs)
        target = copied_sources["offline_screen"]
        target.write_bytes(target.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(evaluate, "evaluate_phase_b", drift_after_evaluation)
    with pytest.raises(ValueError, match="source changed during evaluation"):
        evaluate.run(args)
    assert not Path(args.output_dir).exists()


def _selection_case(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, Path]]:
    screens: dict[str, Path] = {}
    for dataset, vector_mode in (
        ("sift1m", "canonical-exact"),
        ("glove-200-angular", "cosine-normalized"),
    ):
        for candidate_set in ("formal", "frozen-grid"):
            key = f"{dataset}-{candidate_set}"
            args = _tiny_case(
                tmp_path / key,
                candidate_set,
                dataset_name=dataset,
                vector_mode=vector_mode,
            )
            evaluate.run(args)
            screens[key] = Path(args.output_dir) / "screen-manifest.json"
    run_args = argparse.Namespace(
        sift_grid=str(screens["sift1m-frozen-grid"]),
        glove_grid=str(screens["glove-200-angular-frozen-grid"]),
        sift_formal=str(screens["sift1m-formal"]),
        glove_formal=str(screens["glove-200-angular-formal"]),
        output_dir=str(tmp_path / "selection"),
    )
    return run_args, screens


def _force_unique_test_winner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tiny fixture focused on provenance, not CNBR quality."""

    original = selection._deep_replay_screen

    def replay_with_frozen_test_outcome(*args: object, **kwargs: object):
        replayed, provenance = original(*args, **kwargs)
        replayed = copy.deepcopy(replayed)
        if replayed["candidate_set"] == "formal":
            candidate = replayed["candidate"]
            candidate["screen_eligible"] = True
            candidate["materialization_eligible"] = True
        else:
            for name, candidate in replayed["candidates"].items():
                selected = name == "CNBR_9_4"
                candidate["screen_eligible"] = selected
                candidate["materialization_eligible"] = selected
        return replayed, provenance

    monkeypatch.setattr(selection, "_deep_replay_screen", replay_with_frozen_test_outcome)


def test_dual_dataset_selection_is_unique_checksum_bound_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_args, _screens = _selection_case(tmp_path)
    _force_unique_test_winner(monkeypatch)
    manifest_path, digest = selection.run(run_args)
    output_dir = Path(run_args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dual_dataset_all_gate_pass"] == ["CNBR_9_4"]
    assert manifest["selected_formal_candidate"] == "C_CNBR"
    assert (output_dir / "selection-manifest.json.sha256").read_text().strip() == digest
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o444
        for path in output_dir.iterdir()
        if path.is_file()
    )


@pytest.mark.parametrize(
    "target",
    [
        "sift1m-formal",
        "sift1m-frozen-grid",
        "glove-200-angular-formal",
        "glove-200-angular-frozen-grid",
    ],
)
def test_selection_deep_replays_each_of_the_four_screens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    run_args, screens = _selection_case(tmp_path)
    screen_path = screens[target]
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    candidate = (
        screen["candidate"]
        if screen["candidate_set"] == "formal"
        else screen["candidates"]["CNBR_9_4"]
    )
    candidate["owner_sha256"] = "0" * 64
    candidate["actual_metrics"]["physical_copy_load_max_over_mean"] += 0.25
    candidate["graph_topology_all_pass"] = not candidate[
        "graph_topology_all_pass"
    ]
    candidate["screen_eligible"] = not candidate["screen_eligible"]
    _rewrite_frozen_json(screen_path, screen, sidecar=True)
    _force_unique_test_winner(monkeypatch)

    with pytest.raises(ValueError, match="deep replay mismatch"):
        selection.run(run_args)
    assert not Path(run_args.output_dir).exists()


def test_selection_replays_source_binding_instead_of_trusting_screen_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_args, screens = _selection_case(tmp_path)
    screen_path = screens["sift1m-frozen-grid"]
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    binding_path = Path(
        screen["post_freeze_evaluation_inputs"]["source_build_manifest"]
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["vector_lineage"]["preprocessing_contract"] = "forged-contract"
    _rewrite_frozen_json(binding_path, binding, sidecar=True)
    screen["post_freeze_evaluation_inputs"]["source_build_manifest_sha256"] = (
        _sha256(binding_path)
    )
    _rewrite_frozen_json(screen_path, screen, sidecar=True)
    _force_unique_test_winner(monkeypatch)

    with pytest.raises(ValueError, match="binding vector lineage drifted"):
        selection.run(run_args)
    assert not Path(run_args.output_dir).exists()


def test_selection_validates_screen_sidecar_and_replayed_summary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_args, screens = _selection_case(tmp_path)
    screen_path = screens["sift1m-frozen-grid"]
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    screen["identity_all_pass"] = False
    _rewrite_frozen_json(screen_path, screen, sidecar=False)
    _force_unique_test_winner(monkeypatch)
    with pytest.raises(ValueError, match="checksum sidecar mismatch"):
        selection.run(run_args)

    run_args, screens = _selection_case(tmp_path / "summary")
    summary_path = screens["sift1m-frozen-grid"].parent / "summary.json"
    os.chmod(summary_path, 0o644)
    summary_path.write_bytes(summary_path.read_bytes() + b"\n")
    os.chmod(summary_path, 0o444)
    _force_unique_test_winner(monkeypatch)
    with pytest.raises(ValueError, match="summary JSON does not replay"):
        selection.run(run_args)
