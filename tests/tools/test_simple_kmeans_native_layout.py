from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/simple_kmeans_native_layout.py"
    spec = importlib.util.spec_from_file_location("simple_kmeans_native_layout", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_train_hdf5(module, path: Path) -> None:
    with module.experiment.h5py.File(path, "w") as handle:
        handle.create_dataset(
            "train",
            data=module.experiment.np.asarray(
                [
                    [3.0, 4.0],
                    [0.0, 2.0],
                    [1.0, 1.0],
                    [-2.0, 0.0],
                    [0.5, -0.5],
                    [2.0, 1.0],
                ],
                dtype=module.experiment.np.float32,
            ),
        )


def smoke_args(module, hdf5_path: Path, output_dir: Path, *extra: str):
    return module.parse_args(
        [
            "--hdf5-path",
            str(hdf5_path),
            "--output-dir",
            str(output_dir),
            "--train-limit",
            "6",
            "--p",
            "2",
            "--nprobe",
            "1",
            "--lower-hnsw-ef",
            "32",
            "--kmeans-train-size",
            "5",
            "--kmeans-iters",
            "3",
            "--kmeans-seed",
            "7",
            *extra,
        ]
    )


def patch_kmeans(module, monkeypatch):
    captured = {}
    assignments = module.experiment.np.asarray(
        [0, 1, 0, 1, 0, 1],
        dtype=module.experiment.np.int32,
    )
    centroids = module.experiment.np.asarray(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=module.experiment.np.float32,
    )

    def fake_build(train, num_shards, train_size, kmeans_iters, seed):
        captured.update(
            train=train.copy(),
            num_shards=num_shards,
            train_size=train_size,
            kmeans_iters=kmeans_iters,
            seed=seed,
        )
        return assignments, centroids

    monkeypatch.setattr(
        module.experiment,
        "build_cpp_kmeans_baseline_assignments",
        fake_build,
    )
    return captured, assignments, centroids


def test_graphless_only_reuses_cpp_kmeans_and_keeps_single_assignment(monkeypatch, tmp_path):
    module = load_module()
    hdf5_path = tmp_path / "simple.hdf5"
    output_dir = tmp_path / "layout"
    write_train_hdf5(module, hdf5_path)
    captured, assignments, centroids = patch_kmeans(module, monkeypatch)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("graphless-only must not invoke Rust")

    monkeypatch.setattr(module, "run_rust_builder", must_not_run)
    args = smoke_args(
        module,
        hdf5_path,
        output_dir,
        "--generation",
        "4",
        "--graphless-only",
    )

    summary = module.build(args)

    assert captured["num_shards"] == 2
    assert captured["train_size"] == 5
    assert captured["kmeans_iters"] == 3
    assert captured["seed"] == 7
    module.experiment.np.testing.assert_allclose(
        module.experiment.np.linalg.norm(captured["train"], axis=1),
        module.experiment.np.ones(6),
    )
    artifact = json.loads((output_dir / module.GRAPHLESS_NAME).read_text())
    assert artifact["generation"] == 4
    assert artifact["physical_point_count"] == artifact["logical_point_count"] == 6
    assert artifact["routing_distance"] == "squared_l2"
    assert artifact["nprobe"] == 1
    assert artifact["lower_hnsw_ef"] == 32
    assert artifact["centroids"] == [
        {"shard_id": 0, "vector": centroids[0].tolist()},
        {"shard_id": 1, "vector": centroids[1].tolist()},
    ]
    assert artifact["layout_sha256"] == module.experiment.orion_layout_sha256(
        [[int(value)] for value in assignments.tolist()],
        2,
    )
    assert summary["mode"] == "graphless_only"
    assert summary["physical_point_count"] == 6
    assert not (output_dir / "generation-4.json").exists()
    assert (output_dir / module.BUILD_MANIFEST_NAME).is_file()
    assert (output_dir / module.CHECKSUMS_NAME).is_file()


def test_full_mode_mocks_rust_canonicalizer_and_writes_generic_v2_bundle(
    monkeypatch,
    tmp_path,
):
    module = load_module()
    hdf5_path = tmp_path / "simple-full.hdf5"
    output_dir = tmp_path / "full-layout"
    write_train_hdf5(module, hdf5_path)
    _captured, assignments, _centroids = patch_kmeans(module, monkeypatch)

    def fake_rust_builder(_args, graphless_path, production_path):
        production_path.write_bytes(graphless_path.read_bytes())
        Path(f"{production_path}.sha256").write_text(
            module.common.sha256_path(production_path) + "\n",
            encoding="utf-8",
        )
        return ["mock-cargo", "simple_kmeans_build_artifact"]

    monkeypatch.setattr(module, "run_rust_builder", fake_rust_builder)
    args = smoke_args(
        module,
        hdf5_path,
        output_dir,
        "--generation",
        "8",
        "--bundle-prefix",
        "simple-native",
    )

    summary = module.build(args)

    assert summary["mode"] == "production_bundle"
    assert summary["production_artifact_sha256"] == module.common.sha256_path(
        output_dir / "generation-8.json"
    )
    manifest_path = output_dir / "simple-native.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format_version"] == 2
    assert manifest["routing_policy"] == "simple_kmeans"
    assert manifest["routing_generation"] == 8
    assert manifest["routing_artifact_file"] == "generation-8.json"
    assert manifest["routing_artifact_sha256"] == summary["production_artifact_sha256"]
    assert manifest["point_count"] == manifest["total_point_copies"] == 6
    assert manifest["assignments_sha256"] == module.common.sha256_path(
        output_dir / manifest["assignments_file"]
    )
    records = [
        json.loads(line)
        for line in (output_dir / manifest["assignments_file"])
        .read_text()
        .splitlines()
    ]
    assert records == [
        {"id": point_id, "shards": [int(shard_id)]}
        for point_id, shard_id in enumerate(assignments.tolist())
    ]
    build_manifest = json.loads((output_dir / module.BUILD_MANIFEST_NAME).read_text())
    assert build_manifest["outputs"]["rust_builder_command"] == [
        "mock-cargo",
        "simple_kmeans_build_artifact",
    ]
    assert build_manifest["routing"]["expansion_ratio"] == 1.0
    assert "simple-native.f32le" in build_manifest["outputs"]["files"]


def test_rust_builder_command_and_external_target_env(monkeypatch, tmp_path):
    module = load_module()
    target_dir = tmp_path / "cargo-target"
    args = module.parse_args(
        [
            "--hdf5-path",
            str(tmp_path / "input.hdf5"),
            "--output-dir",
            str(tmp_path / "layout"),
            "--cargo",
            "/opt/rust/bin/cargo",
            "--cargo-target-dir",
            str(target_dir),
        ]
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    graphless_path = tmp_path / "graphless.json"
    production_path = tmp_path / "generation-1.json"

    command = module.run_rust_builder(args, graphless_path, production_path)

    assert command == [
        "/opt/rust/bin/cargo",
        "run",
        "--release",
        "-p",
        "collection",
        "--example",
        "simple_kmeans_build_artifact",
        "--",
        str(graphless_path),
        str(production_path),
    ]
    assert captured["command"] == command
    assert captured["cwd"] == module.REPO_ROOT
    assert captured["check"] is True
    assert captured["env"]["CARGO_TARGET_DIR"] == str(target_dir.resolve())


def test_existing_output_is_rejected_before_dataset_work(monkeypatch, tmp_path):
    module = load_module()
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    args = smoke_args(
        module,
        tmp_path / "missing.hdf5",
        output_dir,
        "--graphless-only",
    )

    def must_not_load(*_args, **_kwargs):
        raise AssertionError("dataset must not be read when output already exists")

    monkeypatch.setattr(module.common, "load_train_vectors", must_not_load)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.build(args)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--generation", "0"), "generation"),
        (("--nprobe", "3", "--p", "2"), "nprobe"),
        (("--lower-hnsw-ef", "0"), "lower-hnsw-ef"),
        (("--kmeans-train-size", "0"), "kmeans-train-size"),
    ],
)
def test_invalid_simple_kmeans_parameters_are_rejected(tmp_path, extra, message):
    module = load_module()
    args = smoke_args(
        module,
        tmp_path / "input.hdf5",
        tmp_path / "layout",
        *extra,
    )

    with pytest.raises(ValueError, match=message):
        module.validate_args(args)
