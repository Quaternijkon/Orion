from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/simple_kmeans_native_runtime_profile.py"
    spec = importlib.util.spec_from_file_location(
        "simple_kmeans_native_runtime_profile", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fixture(module, tmp_path: Path) -> tuple[Path, dict]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    layout_sha = "a" * 64
    artifact = {
        "format_version": 1,
        "generation": 1,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "shard_count": 2,
        "layout_sha256": layout_sha,
        "logical_point_count": 2,
        "physical_point_count": 2,
        "routing_distance": "squared_l2",
        "nprobe": 1,
        "lower_hnsw_ef": 32,
        "centroids": [
            {"shard_id": 0, "vector": [1.0, 0.0]},
            {"shard_id": 1, "vector": [0.0, 1.0]},
        ],
    }
    graphless_path = source_dir / module.simple_layout.GRAPHLESS_NAME
    graphless_path.write_text(json.dumps(artifact), encoding="utf-8")
    production_path = source_dir / "generation-1.json"
    production_path.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    vectors_path = source_dir / "simple.f32le"
    vectors_path.write_bytes(b"\x00\x00\x80?" * 4)
    assignments_path = source_dir / "simple.assignments.jsonl"
    assignments_path.write_text(
        '{"id":0,"shards":[0]}\n{"id":1,"shards":[1]}\n',
        encoding="utf-8",
    )
    layout_sha = sha256(assignments_path)
    artifact["layout_sha256"] = layout_sha
    graphless_path.write_text(json.dumps(artifact), encoding="utf-8")
    production_path.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    import_manifest = {
        "format_version": 2,
        "routing_policy": "simple_kmeans",
        "routing_generation": 1,
        "routing_artifact_file": production_path.name,
        "routing_artifact_sha256": sha256(production_path),
        "dimension": 2,
        "point_count": 2,
        "shard_count": 2,
        "total_point_copies": 2,
        "vector_name": "",
        "vectors_file": vectors_path.name,
        "vectors_sha256": sha256(vectors_path),
        "assignments_file": assignments_path.name,
        "assignments_sha256": sha256(assignments_path),
    }
    import_manifest_path = source_dir / "simple.manifest.json"
    import_manifest_path.write_text(json.dumps(import_manifest), encoding="utf-8")
    parameters = {
        "generation": 1,
        "num_shards": 2,
        "vector_distance": "cosine",
        "vector_name": "",
        "routing_distance": "squared_l2",
        "nprobe": 1,
        "lower_hnsw_ef": 32,
        "kmeans_train_size": 2,
        "kmeans_iters": 3,
        "kmeans_seed": 7,
        "cargo_target_dir": "/source/target",
    }
    artifact_binding = {
        "format_version": 1,
        "generation": 1,
        "shard_count": 2,
        "logical_point_count": 2,
        "physical_point_count": 2,
        "routing_distance": "squared_l2",
        "nprobe": 1,
        "lower_hnsw_ef": 32,
        "layout_sha256": layout_sha,
    }
    routing = {
        "policy": "simple_kmeans",
        "logical_point_count": 2,
        "physical_point_count": 2,
        "expansion_ratio": 1.0,
        "shard_counts": [1, 1],
    }
    dataset = {
        "path": "/data/test.hdf5",
        "size_bytes": 123,
        "sha256": "b" * 64,
        "train_rows_total": 2,
        "train_rows_used": 2,
        "dimension": 2,
    }
    build_manifest_path = source_dir / module.layout_common.BUILD_MANIFEST_NAME
    build_manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "tool": "tools/simple_kmeans_native_layout.py",
                "mode": "production_bundle",
                "dataset": dataset,
                "parameters": parameters,
                "artifact_binding": artifact_binding,
                "routing": routing,
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )
    source = {
        "layout": {
            "generation": 1,
            "shard_count": 2,
            "logical_point_count": 2,
            "physical_point_count": 2,
            "artifact": {"layout_sha256": layout_sha},
            "artifact_sha256": sha256(production_path),
            "import_manifest_sha256": sha256(import_manifest_path),
        },
        "build_manifest": json.loads(build_manifest_path.read_text()),
        "build_manifest_path": build_manifest_path,
        "graphless_path": graphless_path,
        "production": artifact,
        "production_path": production_path,
        "import_manifest": import_manifest,
        "import_manifest_path": import_manifest_path,
        "vectors_path": vectors_path,
        "assignments_path": assignments_path,
        "parameters": parameters,
        "routing": routing,
        "artifact_binding": artifact_binding,
        "dataset": dataset,
    }
    return source_dir, source


def args(module, source_dir: Path, output_dir: Path, *extra: str):
    return module.parse_args(
        [
            "--source-layout-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--generation",
            "2",
            "--nprobe",
            "2",
            "--lower-hnsw-ef",
            "96",
            *extra,
        ]
    )


def patch_source_and_builder(module, monkeypatch, source: dict) -> None:
    monkeypatch.setattr(module, "validate_source_bundle", lambda _path: source)

    def fake_builder(_args, graphless_path, production_path):
        value = json.loads(graphless_path.read_text(encoding="utf-8"))
        production_path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        Path(f"{production_path}.sha256").write_text(
            sha256(production_path) + "\n",
            encoding="utf-8",
        )
        return ["mock-cargo", "simple_kmeans_build_artifact"]

    monkeypatch.setattr(module.simple_layout, "run_rust_builder", fake_builder)


def test_build_derives_only_runtime_fields_and_copies_payloads(monkeypatch, tmp_path):
    module = load_module()
    source_dir, source = source_fixture(module, tmp_path)
    output_dir = tmp_path / "derived"
    patch_source_and_builder(module, monkeypatch, source)

    summary = module.build(args(module, source_dir, output_dir))

    artifact = json.loads((output_dir / "generation-2.json").read_text())
    source_artifact = source["production"]
    assert artifact["generation"] == 2
    assert artifact["nprobe"] == 2
    assert artifact["lower_hnsw_ef"] == 96
    for field in (
        "vector_schema",
        "shard_count",
        "layout_sha256",
        "logical_point_count",
        "physical_point_count",
        "routing_distance",
        "centroids",
    ):
        assert artifact[field] == source_artifact[field]

    import_manifest = json.loads(
        (output_dir / source["import_manifest_path"].name).read_text()
    )
    assert import_manifest["vectors_sha256"] == source["import_manifest"][
        "vectors_sha256"
    ]
    assert import_manifest["assignments_sha256"] == source["import_manifest"][
        "assignments_sha256"
    ]
    assert import_manifest["routing_generation"] == 2
    assert import_manifest["routing_artifact_file"] == "generation-2.json"
    manifest = json.loads(
        (output_dir / module.layout_common.BUILD_MANIFEST_NAME).read_text()
    )
    assert manifest["tool"] == module.TOOL_NAME
    assert manifest["derivation"]["kind"] == module.DERIVATION_KIND
    assert manifest["derivation"]["formal_evidence_eligible"] is True
    assert manifest["parameters"]["kmeans_seed"] == 7
    assert manifest["parameters"]["nprobe"] == 2
    assert manifest["parameters"]["lower_hnsw_ef"] == 96
    assert summary["formal_evidence_eligible"] is True
    for name in ("vectors", "assignments"):
        proof = summary["reused_payloads"][name]
        assert proof["materialization"] == "copy"
        assert proof["same_inode"] is False


def test_hardlink_profile_is_explicitly_diagnostic(monkeypatch, tmp_path):
    module = load_module()
    source_dir, source = source_fixture(module, tmp_path)
    output_dir = tmp_path / "hardlinked"
    patch_source_and_builder(module, monkeypatch, source)

    summary = module.build(
        args(module, source_dir, output_dir, "--payload-mode", "hardlink")
    )

    assert summary["formal_evidence_eligible"] is False
    assert all(
        proof["same_inode"] is True
        for proof in summary["reused_payloads"].values()
    )
    manifest = json.loads(
        (output_dir / module.layout_common.BUILD_MANIFEST_NAME).read_text()
    )
    assert manifest["derivation"]["formal_evidence_eligible"] is False


def test_typed_binding_accepts_decimal_spelling_but_not_changed_f32_bits():
    module = load_module()
    graphless = {
        "generation": 1,
        "centroids": [{"shard_id": 0, "vector": [0.10000000149011612]}],
    }
    canonical = {
        "generation": 1,
        "centroids": [{"shard_id": 0, "vector": [0.1]}],
    }
    module.validate_typed_binding(graphless, canonical)

    changed = {
        "generation": 1,
        "centroids": [{"shard_id": 0, "vector": [0.1001]}],
    }
    with pytest.raises(RuntimeError, match="float32 precision"):
        module.validate_typed_binding(graphless, changed)


def test_runtime_parameters_preserve_all_offline_kmeans_controls(tmp_path):
    module = load_module()
    source = {
        "generation": 1,
        "nprobe": 19,
        "lower_hnsw_ef": 110,
        "num_shards": 46,
        "kmeans_train_size": 10000,
        "kmeans_iters": 10,
        "kmeans_seed": 1,
        "vector_distance": "cosine",
    }
    runtime_args = argparse.Namespace(generation=2, nprobe=26, lower_hnsw_ef=210)

    derived = module.runtime_parameters(source, runtime_args)

    assert derived == {
        **source,
        "generation": 2,
        "nprobe": 26,
        "lower_hnsw_ef": 210,
    }


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--generation", "0"), "generation"),
        (("--nprobe", "0"), "nprobe"),
        (("--lower-hnsw-ef", "0"), "lower-hnsw-ef"),
    ],
)
def test_invalid_runtime_parameters_are_rejected(tmp_path, extra, message):
    module = load_module()
    parsed = args(module, tmp_path / "source", tmp_path / "output", *extra)

    with pytest.raises(ValueError, match=message):
        module.validate_args(parsed)
