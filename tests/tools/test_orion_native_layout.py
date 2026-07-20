from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/orion_native_layout.py"
    spec = importlib.util.spec_from_file_location("orion_native_layout", path)
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
            "--sample-denominator",
            "2",
            "--upper-k",
            "2",
            "--upper-search-ef",
            "4",
            "--k-overlap",
            "2",
            "--upper-build-batch-size",
            "3",
            *extra,
        ]
    )


def patch_algorithm_pipeline(module, monkeypatch):
    calls = []
    upper_indices = module.experiment.np.asarray([4, 1, 3], dtype=module.experiment.np.int64)
    point_to_l1s = [[4, 1], [1, 4], [3, 1], [3, 4], [4, 3], [1, 3]]
    routing = SimpleNamespace(
        initial_num_shards=2,
        num_shards=3,
        point_to_shards=[[0], [1], [2], [0], [1], [2]],
        total_assigned=6,
        expansion_ratio=1.0,
        topology_iterations=4,
        shard_counts=module.experiment.np.asarray([2, 2, 2], dtype=module.experiment.np.int64),
        fission_events=[{"source_shard": 1, "accepted": True, "split_k": 2}],
    )

    real_prepare = module.experiment.prepare_vectors_for_distance

    def prepare(train, distance):
        calls.append(("prepare_vectors_for_distance", distance, len(train)))
        return real_prepare(train, distance)

    def select(num_points, denominator, seed):
        calls.append(("global_upper_indices", num_points, denominator, seed))
        return upper_indices

    upper_index = object()

    def build_upper(vectors, labels, dim, m, ef_construction, ef_search, space):
        calls.append(
            (
                "build_upper_index",
                vectors.copy(),
                labels.copy(),
                dim,
                m,
                ef_construction,
                ef_search,
                space,
            )
        )
        return upper_index

    def attach(index, train, k_overlap, batch_size):
        calls.append(("compute_point_to_l1s", index, len(train), k_overlap, batch_size))
        return point_to_l1s

    def build_routing(
        train,
        selected_upper,
        attachments,
        initial_num_shards,
        kmeans_iters,
        kmeans_seed,
        topology_iters,
        **kwargs,
    ):
        calls.append(
            (
                "build_original_routing_state",
                len(train),
                selected_upper.copy(),
                attachments,
                initial_num_shards,
                kmeans_iters,
                kmeans_seed,
                topology_iters,
                kwargs,
            )
        )
        return routing

    def write_graphless(
        train,
        selected_upper,
        point_to_shards,
        num_shards,
        output_path,
        **kwargs,
    ):
        calls.append(
            (
                "write_orion_graphless_artifact",
                len(train),
                selected_upper.copy(),
                point_to_shards,
                num_shards,
                kwargs,
            )
        )
        output_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "generation": kwargs["generation"],
                    "layout_sha256": "a" * 64,
                    "logical_point_count": len(train),
                    "physical_point_count": sum(len(shards) for shards in point_to_shards),
                    "shard_count": num_shards,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(module.experiment, "prepare_vectors_for_distance", prepare)
    monkeypatch.setattr(module.experiment, "global_upper_indices", select)
    monkeypatch.setattr(module.experiment, "build_upper_index", build_upper)
    monkeypatch.setattr(module.experiment, "compute_point_to_l1s", attach)
    monkeypatch.setattr(module.experiment, "build_original_routing_state", build_routing)
    monkeypatch.setattr(module.experiment, "write_orion_graphless_artifact", write_graphless)
    return calls, routing


def test_graphless_only_is_a_thin_wrapper_over_existing_orion_pipeline(monkeypatch, tmp_path):
    module = load_module()
    hdf5_path = tmp_path / "smoke.hdf5"
    output_dir = tmp_path / "layout"
    write_train_hdf5(module, hdf5_path)
    calls, routing = patch_algorithm_pipeline(module, monkeypatch)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("graphless-only must not invoke the Rust builder")

    monkeypatch.setattr(module, "run_rust_builder", must_not_run)
    args = smoke_args(
        module,
        hdf5_path,
        output_dir,
        "--generation",
        "7",
        "--dynamic-ef-base",
        "48",
        "--dynamic-ef-factor",
        "15",
        "--multi-assign-min-max-vote",
        "3",
        "--multi-assign-vote-delta",
        "1",
        "--multi-assign-max-shards",
        "2",
        "--graphless-only",
    )

    summary = module.build(args)

    assert [call[0] for call in calls] == [
        "prepare_vectors_for_distance",
        "global_upper_indices",
        "build_upper_index",
        "compute_point_to_l1s",
        "build_original_routing_state",
        "write_orion_graphless_artifact",
    ]
    routing_call = calls[4]
    assert routing_call[4:8] == (2, 10, 1, 50)
    assert routing_call[8] == {
        "use_multi_assign": True,
        "enable_fission": True,
        "multi_assign_min_max_vote": 3,
        "multi_assign_vote_delta": 1,
        "multi_assign_max_shards": 2,
    }
    graphless_call = calls[5]
    assert graphless_call[3] == routing.point_to_shards
    assert graphless_call[4] == 3
    assert graphless_call[5] == {
        "generation": 7,
        "vector_distance": "cosine",
        "upper_k": 2,
        "upper_ef_search": 4,
        "dynamic_ef_base": 48,
        "dynamic_ef_factor": 15,
        "vector_name": "",
    }
    assert summary["mode"] == "graphless_only"
    assert summary["effective_num_shards"] == 3
    assert summary["layout_sha256"] == "a" * 64
    assert not (output_dir / "generation-7.json").exists()

    manifest = json.loads((output_dir / module.BUILD_MANIFEST_NAME).read_text())
    assert manifest["mode"] == "graphless_only"
    assert manifest["dataset"]["train_rows_used"] == 6
    assert manifest["routing"]["physical_point_count"] == 6
    assert manifest["artifact_binding"]["layout_sha256"] == "a" * 64
    assert manifest["routing"]["fission_events"] == routing.fission_events
    checksum_lines = (output_dir / module.CHECKSUMS_NAME).read_text().splitlines()
    assert any(line.endswith(f"  {module.GRAPHLESS_NAME}") for line in checksum_lines)
    assert any(line.endswith(f"  {module.BUILD_MANIFEST_NAME}") for line in checksum_lines)


def test_full_mode_mocks_rust_builder_then_reuses_existing_bundle_writer(monkeypatch, tmp_path):
    module = load_module()
    hdf5_path = tmp_path / "full-smoke.hdf5"
    output_dir = tmp_path / "full-layout"
    write_train_hdf5(module, hdf5_path)
    _calls, routing = patch_algorithm_pipeline(module, monkeypatch)
    captured = {}

    def fake_rust_builder(args, graphless_path, production_path):
        captured["rust"] = (args.upper_graph_seed, graphless_path, production_path)
        production_path.write_text(
            json.dumps({"format_version": 1, "upper_graph": {"entry_point": 0}}),
            encoding="utf-8",
        )
        Path(f"{production_path}.sha256").write_text(
            module.sha256_path(production_path) + "\n",
            encoding="utf-8",
        )
        return ["mock-cargo", "orion_build_artifact"]

    def fake_bundle_writer(
        train,
        point_to_shards,
        num_shards,
        target_dir,
        **kwargs,
    ):
        captured["bundle"] = {
            "train_rows": len(train),
            "point_to_shards": point_to_shards,
            "num_shards": num_shards,
            "target_dir": target_dir,
            **kwargs,
        }
        prefix = kwargs["prefix"]
        (target_dir / f"{prefix}.f32le").write_bytes(b"vectors")
        (target_dir / f"{prefix}.assignments.jsonl").write_text(
            '{"id":0,"shards":[0]}\n', encoding="utf-8"
        )
        manifest_path = target_dir / f"{prefix}.manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        return manifest_path

    monkeypatch.setattr(module, "run_rust_builder", fake_rust_builder)
    monkeypatch.setattr(
        module.experiment,
        "write_orion_numeric_shard_import_bundle",
        fake_bundle_writer,
    )
    args = smoke_args(
        module,
        hdf5_path,
        output_dir,
        "--generation",
        "9",
        "--upper-graph-seed",
        "123",
        "--bundle-prefix",
        "native-smoke",
    )

    summary = module.build(args)

    assert captured["rust"] == (
        123,
        output_dir / module.GRAPHLESS_NAME,
        output_dir / "generation-9.json",
    )
    assert captured["bundle"]["train_rows"] == 6
    assert captured["bundle"]["point_to_shards"] == routing.point_to_shards
    assert captured["bundle"]["num_shards"] == routing.num_shards
    assert captured["bundle"]["orion_artifact_path"] == output_dir / "generation-9.json"
    assert captured["bundle"]["prefix"] == "native-smoke"
    assert summary["mode"] == "production_bundle"
    assert summary["production_artifact"] == str(output_dir / "generation-9.json")
    assert summary["import_manifest"] == str(output_dir / "native-smoke.manifest.json")
    manifest = json.loads((output_dir / module.BUILD_MANIFEST_NAME).read_text())
    assert manifest["outputs"]["rust_builder_command"] == [
        "mock-cargo",
        "orion_build_artifact",
    ]
    assert "native-smoke.f32le" in manifest["outputs"]["files"]


def test_rust_builder_command_targets_collection_production_example(tmp_path):
    module = load_module()
    args = module.parse_args(
        [
            "--hdf5-path",
            str(tmp_path / "input.hdf5"),
            "--output-dir",
            str(tmp_path / "layout"),
            "--cargo",
            "/opt/rust/bin/cargo",
            "--upper-graph-seed",
            "17",
            "--upper-m",
            "24",
            "--upper-ef-construction",
            "88",
        ]
    )

    command = module.rust_builder_command(
        args,
        tmp_path / "graphless.json",
        tmp_path / "generation-1.json",
    )

    assert command == [
        "/opt/rust/bin/cargo",
        "run",
        "--release",
        "-p",
        "collection",
        "--example",
        "orion_build_artifact",
        "--",
        str(tmp_path / "graphless.json"),
        str(tmp_path / "generation-1.json"),
        "--seed",
        "17",
        "--m",
        "24",
        "--ef",
        "88",
    ]


def test_run_rust_builder_passes_external_cargo_target_dir(monkeypatch, tmp_path):
    module = load_module()
    target_dir = tmp_path / "external-cargo-target"
    args = module.parse_args(
        [
            "--hdf5-path",
            str(tmp_path / "input.hdf5"),
            "--output-dir",
            str(tmp_path / "layout"),
            "--cargo-target-dir",
            str(target_dir),
        ]
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    command = module.run_rust_builder(
        args,
        tmp_path / "graphless.json",
        tmp_path / "generation-1.json",
    )

    assert captured["command"] == command
    assert captured["cwd"] == module.REPO_ROOT
    assert captured["check"] is True
    assert captured["env"]["CARGO_TARGET_DIR"] == str(target_dir.resolve())


def test_existing_output_is_rejected_before_dataset_or_algorithm_work(monkeypatch, tmp_path):
    module = load_module()
    hdf5_path = tmp_path / "unused.hdf5"
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    args = smoke_args(module, hdf5_path, output_dir, "--graphless-only")

    def must_not_load(*_args, **_kwargs):
        raise AssertionError("existing output must be rejected before reading the dataset")

    monkeypatch.setattr(module, "load_train_vectors", must_not_load)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.build(args)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--generation", "0"), "generation"),
        (("--dynamic-ef-factor", "-1"), "dynamic-ef-factor"),
        (("--multi-assign-vote-delta", "-1"), "multi-assign-vote-delta"),
        (("--upper-k", "8", "--upper-search-ef", "4"), "upper-search-ef"),
    ],
)
def test_invalid_layout_parameters_are_rejected(tmp_path, extra, message):
    module = load_module()
    args = smoke_args(
        module,
        tmp_path / "input.hdf5",
        tmp_path / "layout",
        *extra,
    )

    with pytest.raises(ValueError, match=message):
        module.validate_args(args)
