from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/orion_native_runtime_profile.py"
    spec = importlib.util.spec_from_file_location("orion_native_runtime_profile", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def faithful_parameters() -> dict[str, object]:
    return {
        "generation": 7,
        "initial_num_shards": 31,
        "vector_distance": "cosine",
        "vector_name": "",
        "sample_denominator": 32,
        "upper_sample_seed": 100,
        "upper_m": 32,
        "upper_ef_construction": 100,
        "attachment_search_ef": 100,
        "upper_search_ef": 2,
        "upper_k": 2,
        "allow_decoupled_runtime_upper_search": False,
        "k_overlap": 10,
        "upper_build_batch_size": 10_000,
        "dynamic_ef_base": 20,
        "dynamic_ef_factor": 4,
        "kmeans_iters": 10,
        "kmeans_seed": 1,
        "topology_iters": 50,
        "use_multi_assign": True,
        "multi_assign_min_max_vote": 2,
        "multi_assign_vote_delta": 0,
        "multi_assign_max_shards": 0,
        "enable_fission": True,
        "upper_graph_seed": 100,
        "cargo_target_dir": "/tmp/source-cargo-target",
    }


def write_source_bundle(module, source_dir: Path) -> dict[str, object]:
    source_dir.mkdir()
    assignments = [[0], [0, 1], [1], [1]]
    assignment_bytes = b"".join(
        module.prepare.experiment.canonical_orion_assignment_line(point_id, shards)
        for point_id, shards in enumerate(assignments)
    )
    assignments_path = source_dir / "orion.assignments.jsonl"
    assignments_path.write_bytes(assignment_bytes)
    assignments_sha256 = module.layout_common.sha256_path(assignments_path)

    vectors = module.prepare.experiment.np.asarray(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [-1.0, 0.0]],
        dtype="<f4",
    )
    vectors_path = source_dir / "orion.f32le"
    vectors_path.write_bytes(vectors.tobytes(order="C"))
    vectors_sha256 = module.layout_common.sha256_path(vectors_path)
    upper_graph = {
        "entry_point": 0,
        "max_level": 0,
        "nodes": [
            {"label": 0, "neighbors_by_level": [[2]]},
            {"label": 2, "neighbors_by_level": [[0]]},
        ],
    }
    graphless = {
        "format_version": 1,
        "generation": 7,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "shard_count": 2,
        "layout_sha256": assignments_sha256,
        "logical_point_count": 4,
        "physical_point_count": 5,
        "upper_k": 2,
        "upper_ef_search": 2,
        "dynamic_ef_base": 20,
        "dynamic_ef_factor": 4,
        "upper_nodes": [
            {"label": 0, "vector": [1.0, 0.0], "shard_membership": [0]},
            {"label": 2, "vector": [0.0, 1.0], "shard_membership": [1]},
        ],
    }
    graphless_path = source_dir / module.layout_common.GRAPHLESS_NAME
    module.layout_common.write_json_new(graphless_path, graphless)
    production = dict(graphless, upper_graph=upper_graph)
    production_path = source_dir / "generation-7.json"
    module.layout_common.write_json_new(production_path, production)
    production_sha256 = module.layout_common.sha256_path(production_path)
    Path(f"{production_path}.sha256").write_text(
        production_sha256 + "\n", encoding="utf-8"
    )

    import_manifest = {
        "format_version": 1,
        "dimension": 2,
        "point_count": 4,
        "shard_count": 2,
        "vector_name": "",
        "orion_generation": 7,
        "orion_artifact_file": production_path.name,
        "orion_artifact_sha256": production_sha256,
        "vectors_file": vectors_path.name,
        "vectors_sha256": vectors_sha256,
        "assignments_file": assignments_path.name,
        "assignments_sha256": assignments_sha256,
        "total_point_copies": 5,
    }
    import_manifest_path = source_dir / "orion.manifest.json"
    module.layout_common.write_json_new(import_manifest_path, import_manifest)

    parameters = faithful_parameters()
    artifact_binding = {
        "generation": 7,
        "layout_sha256": assignments_sha256,
        "logical_point_count": 4,
        "physical_point_count": 5,
        "shard_count": 2,
    }
    routing = {
        "initial_num_shards": 31,
        "effective_num_shards": 2,
        "upper_point_count": 2,
        "logical_point_count": 4,
        "physical_point_count": 5,
        "expansion_ratio": 1.25,
        "topology_iterations": 4,
        "shard_counts": [2, 3],
        "fission_events": [],
    }
    dataset = {
        "path": "/datasets/source.hdf5",
        "size_bytes": 123,
        "sha256": "d" * 64,
        "train_rows_total": 4,
        "train_rows_used": 4,
        "dimension": 2,
    }
    files = module.layout_common.relative_file_records(
        source_dir,
        {module.layout_common.BUILD_MANIFEST_NAME, module.layout_common.CHECKSUMS_NAME},
    )
    build_manifest = {
        "format_version": 1,
        "tool": "tools/orion_native_layout.py",
        "created_at": "2026-07-20T00:00:00Z",
        "mode": "production_bundle",
        "dataset": dataset,
        "parameters": parameters,
        "artifact_binding": artifact_binding,
        "routing": routing,
        "outputs": {
            "graphless_artifact": graphless_path.name,
            "production_artifact": production_path.name,
            "import_manifest": import_manifest_path.name,
            "rust_builder_command": ["cargo", "mock"],
            "files": files,
        },
    }
    module.layout_common.write_json_new(
        source_dir / module.layout_common.BUILD_MANIFEST_NAME,
        build_manifest,
    )
    module.layout_common.write_checksums(source_dir)
    return {
        "graphless": graphless,
        "upper_graph": upper_graph,
        "production_sha256": production_sha256,
        "vectors_path": vectors_path,
        "assignments_path": assignments_path,
        "import_manifest": import_manifest,
        "parameters": parameters,
        "routing": routing,
    }


def runtime_args(module, source_dir: Path, output_dir: Path, *extra: str):
    return module.parse_args(
        [
            "--source-layout-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--generation",
            "8",
            "--upper-k",
            "1",
            "--dynamic-ef-base",
            "48",
            "--dynamic-ef-factor",
            "15",
            *extra,
        ]
    )


def patch_rust_builder(module, monkeypatch, upper_graph):
    calls = []

    def fake_builder(args, graphless_path, production_path):
        calls.append(
            {
                "seed": args.upper_graph_seed,
                "m": args.upper_m,
                "ef": args.upper_ef_construction,
                "cargo": args.cargo,
                "cargo_target_dir": args.cargo_target_dir,
            }
        )
        payload = json.loads(graphless_path.read_text(encoding="utf-8"))
        payload["upper_graph"] = upper_graph
        module.layout_common.write_json_new(production_path, payload)
        Path(f"{production_path}.sha256").write_text(
            module.layout_common.sha256_path(production_path) + "\n",
            encoding="utf-8",
        )
        return ["mock-cargo", "orion_build_artifact"]

    monkeypatch.setattr(module.layout_common, "run_rust_builder", fake_builder)
    return calls


def test_derives_runtime_profile_with_identical_layout_and_hardlinked_import_payloads(
    monkeypatch, tmp_path
):
    module = load_module()
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "derived"
    source = write_source_bundle(module, source_dir)
    source_hashes = {
        path.relative_to(source_dir).as_posix(): module.layout_common.sha256_path(path)
        for path in source_dir.iterdir()
        if path.is_file()
    }
    calls = patch_rust_builder(module, monkeypatch, source["upper_graph"])

    summary = module.build(
        runtime_args(module, source_dir, output_dir, "--payload-mode", "hardlink")
    )

    assert calls == [
        {
            "seed": 100,
            "m": 32,
            "ef": 100,
            "cargo": "cargo",
            "cargo_target_dir": None,
        }
    ]
    assert summary["layout_sha256"] == source["graphless"]["layout_sha256"]
    assert summary["payload_mode"] == "hardlink"
    assert summary["formal_evidence_eligible"] is False
    assert all(proof["same_inode"] for proof in summary["reused_payloads"].values())
    for relative, digest in source_hashes.items():
        assert module.layout_common.sha256_path(source_dir / relative) == digest

    derived_graphless = json.loads(
        (output_dir / module.layout_common.GRAPHLESS_NAME).read_text(encoding="utf-8")
    )
    changed = {
        key
        for key in set(source["graphless"]) | set(derived_graphless)
        if source["graphless"].get(key) != derived_graphless.get(key)
    }
    assert changed == {
        "generation",
        "upper_k",
        "upper_ef_search",
        "dynamic_ef_base",
        "dynamic_ef_factor",
    }
    assert derived_graphless["generation"] == 8
    assert derived_graphless["upper_k"] == derived_graphless["upper_ef_search"] == 1
    assert derived_graphless["dynamic_ef_base"] == 48
    assert derived_graphless["dynamic_ef_factor"] == 15
    assert derived_graphless["upper_nodes"] == source["graphless"]["upper_nodes"]

    derived_import = json.loads(
        (output_dir / "orion.manifest.json").read_text(encoding="utf-8")
    )
    import_changes = {
        key
        for key in set(source["import_manifest"]) | set(derived_import)
        if source["import_manifest"].get(key) != derived_import.get(key)
    }
    assert import_changes == {
        "orion_generation",
        "orion_artifact_file",
        "orion_artifact_sha256",
    }
    assert derived_import["orion_generation"] == 8
    assert derived_import["assignments_sha256"] == source["graphless"]["layout_sha256"]
    assert (output_dir / "orion.f32le").stat().st_ino == source[
        "vectors_path"
    ].stat().st_ino
    assert (output_dir / "orion.assignments.jsonl").stat().st_ino == source[
        "assignments_path"
    ].stat().st_ino

    build_manifest = json.loads(
        (output_dir / module.layout_common.BUILD_MANIFEST_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert build_manifest["tool"] == module.TOOL_NAME
    assert build_manifest["dataset"] == {
        "path": "/datasets/source.hdf5",
        "size_bytes": 123,
        "sha256": "d" * 64,
        "train_rows_total": 4,
        "train_rows_used": 4,
        "dimension": 2,
    }
    assert build_manifest["routing"] == source["routing"]
    assert build_manifest["derivation"]["source"]["parameters"] == source[
        "parameters"
    ]
    assert build_manifest["parameters"]["cargo_target_dir"] == (
        source["parameters"]["cargo_target_dir"]
    )
    assert build_manifest["derivation"]["allowed_parameter_changes"] == list(
        module.RUNTIME_PARAMETER_KEYS
    )

    loaded = module.prepare.load_routed_layout("orion", output_dir)
    assert loaded["generation"] == 8
    assert loaded["artifact"]["layout_sha256"] == source["graphless"][
        "layout_sha256"
    ]


def test_copy_mode_creates_independent_byte_identical_payloads(monkeypatch, tmp_path):
    module = load_module()
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "derived"
    source = write_source_bundle(module, source_dir)
    patch_rust_builder(module, monkeypatch, source["upper_graph"])

    summary = module.build(runtime_args(module, source_dir, output_dir))

    assert summary["formal_evidence_eligible"] is True
    assert all(
        proof["materialization"] == "copy" and proof["same_inode"] is False
        for proof in summary["reused_payloads"].values()
    )
    for name, source_path in (
        ("orion.f32le", source["vectors_path"]),
        ("orion.assignments.jsonl", source["assignments_path"]),
    ):
        destination = output_dir / name
        assert destination.read_bytes() == source_path.read_bytes()
        assert destination.stat().st_ino != source_path.stat().st_ino


def test_validated_runtime_profile_can_be_the_source_of_another_profile(
    monkeypatch, tmp_path
):
    module = load_module()
    source_dir = tmp_path / "source"
    first_dir = tmp_path / "profile-8"
    second_dir = tmp_path / "profile-9"
    source = write_source_bundle(module, source_dir)
    patch_rust_builder(module, monkeypatch, source["upper_graph"])
    module.build(runtime_args(module, source_dir, first_dir))

    second_args = runtime_args(module, first_dir, second_dir)
    second_args.generation = 9
    second_args.dynamic_ef_base = 64
    second_args.dynamic_ef_factor = 20
    summary = module.build(second_args)

    assert summary["source_generation"] == 8
    assert summary["generation"] == 9
    assert summary["layout_sha256"] == source["graphless"]["layout_sha256"]
    loaded = module.prepare.load_routed_layout("orion", second_dir)
    assert loaded["generation"] == 9
    manifest = json.loads(
        (second_dir / module.layout_common.BUILD_MANIFEST_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["derivation"]["source"]["generation"] == 8
    assert manifest["derivation"]["source"]["parameters"]["generation"] == 8


def test_existing_output_is_rejected_before_source_validation(monkeypatch, tmp_path):
    module = load_module()
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    args = runtime_args(module, tmp_path / "missing-source", output_dir)

    def must_not_load(_source_dir):
        raise AssertionError("existing output must be rejected first")

    monkeypatch.setattr(module, "validate_source_bundle", must_not_load)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.build(args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation", 0, "generation"),
        ("upper_k", 0, "upper-k"),
        ("dynamic_ef_base", 0, "dynamic-ef-base"),
        ("dynamic_ef_factor", -1, "dynamic-ef-factor"),
    ],
)
def test_invalid_runtime_parameters_are_rejected(tmp_path, field, value, message):
    module = load_module()
    args = runtime_args(module, tmp_path / "source", tmp_path / "derived")
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        module.validate_args(args)


def test_same_generation_is_valid_in_a_new_bundle_but_upper_k_is_bounded(
    monkeypatch, tmp_path
):
    module = load_module()
    source_dir = tmp_path / "source"
    source = write_source_bundle(module, source_dir)
    patch_rust_builder(module, monkeypatch, source["upper_graph"])

    same_generation = runtime_args(module, source_dir, tmp_path / "same")
    same_generation.generation = 7
    summary = module.build(same_generation)
    assert summary["source_generation"] == summary["generation"] == 7
    assert module.prepare.load_routed_layout("orion", tmp_path / "same")[
        "generation"
    ] == 7

    too_large = runtime_args(module, source_dir, tmp_path / "large")
    too_large.upper_k = 3
    with pytest.raises(ValueError, match="exceeds source upper tier"):
        module.build(too_large)


def test_rejects_source_graphless_and_production_drift(tmp_path):
    module = load_module()
    source_dir = tmp_path / "source"
    write_source_bundle(module, source_dir)
    production_path = source_dir / "generation-7.json"
    production = json.loads(production_path.read_text(encoding="utf-8"))
    production["upper_nodes"][0]["vector"][0] = 0.25
    production_path.write_text(json.dumps(production), encoding="utf-8")
    production_sha256 = module.layout_common.sha256_path(production_path)
    Path(f"{production_path}.sha256").write_text(
        production_sha256 + "\n", encoding="utf-8"
    )
    import_manifest_path = source_dir / "orion.manifest.json"
    import_manifest = json.loads(import_manifest_path.read_text(encoding="utf-8"))
    import_manifest["orion_artifact_sha256"] = production_sha256
    import_manifest_path.write_text(json.dumps(import_manifest), encoding="utf-8")
    build_manifest_path = source_dir / module.layout_common.BUILD_MANIFEST_NAME
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    build_manifest["outputs"]["files"] = module.layout_common.relative_file_records(
        source_dir,
        {module.layout_common.BUILD_MANIFEST_NAME, module.layout_common.CHECKSUMS_NAME},
    )
    build_manifest_path.write_text(json.dumps(build_manifest), encoding="utf-8")
    (source_dir / module.layout_common.CHECKSUMS_NAME).unlink()
    module.layout_common.write_checksums(source_dir)

    with pytest.raises(RuntimeError, match="differs from production artifact"):
        module.validate_source_bundle(source_dir)


def test_graphless_binding_accepts_equivalent_float32_decimal_spellings():
    module = load_module()
    source = {
        "generation": 1,
        "upper_nodes": [
            {
                "label": 7,
                "vector": [0.10000000149011612, -0.20000000298023224],
                "shard_membership": [1, 3],
            }
        ],
    }
    canonical = {
        "generation": 1,
        "upper_nodes": [
            {
                "label": 7,
                "vector": [0.1, -0.2],
                "shard_membership": [1, 3],
            }
        ],
    }

    module.validate_graphless_typed_binding(source, canonical)


def test_prepare_rejects_runtime_profile_source_binding_drift(monkeypatch, tmp_path):
    module = load_module()
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "derived"
    source = write_source_bundle(module, source_dir)
    patch_rust_builder(module, monkeypatch, source["upper_graph"])
    module.build(runtime_args(module, source_dir, output_dir))

    build_manifest_path = output_dir / module.layout_common.BUILD_MANIFEST_NAME
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    build_manifest["derivation"]["source"]["assignments_sha256"] = "e" * 64
    build_manifest_path.write_text(json.dumps(build_manifest), encoding="utf-8")
    (output_dir / module.layout_common.CHECKSUMS_NAME).unlink()
    module.layout_common.write_checksums(output_dir)

    with pytest.raises(RuntimeError, match="source assignments"):
        module.prepare.load_routed_layout("orion", output_dir)
