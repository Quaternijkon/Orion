from __future__ import annotations

import importlib.util
import json
import struct
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
            "2",
            "--k-overlap",
            "2",
            "--attachment-search-ef",
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
        l1_to_shard=[0, 2, 2, 0, 1, 2],
        point_to_shards=[[0], [1, 2], [2], [0], [1], [2]],
        total_assigned=7,
        expansion_ratio=7 / 6,
        topology_iterations=4,
        shard_counts=module.experiment.np.asarray([2, 2, 3], dtype=module.experiment.np.int64),
        fission_events=[],
        balance_diagnostics=None,
    )

    real_prepare = module.experiment.prepare_vectors_for_distance

    def prepare(train, distance):
        calls.append(("prepare_vectors_for_distance", distance, len(train)))
        return real_prepare(train, distance)

    def select(num_points, denominator, seed):
        calls.append(("global_upper_indices", num_points, denominator, seed))
        return upper_indices

    def write_vectors(train, output_path, **kwargs):
        calls.append(("write_orion_numeric_vector_file", len(train), output_path, kwargs))
        output_path.write_bytes(b"canonical-vectors")
        return module.sha256_path(output_path)

    def write_upper_seed(train, selected_upper, output_path, **kwargs):
        calls.append(
            (
                "write_orion_upper_seed_graphless_artifact",
                len(train),
                selected_upper.copy(),
                output_path,
                kwargs,
            )
        )
        output_path.write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "generation": kwargs["generation"],
                    "layout_sha256": "0" * 64,
                    "logical_point_count": len(train),
                    "physical_point_count": len(train),
                    "shard_count": 1,
                    "upper_nodes": [
                        {
                            "label": int(point_id),
                            "vector": train[int(point_id)].tolist(),
                            "owner_shard": 0,
                        }
                        for point_id in selected_upper.tolist()
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path

    def run_builder(args, graphless_path, production_path):
        calls.append(("run_rust_builder", graphless_path, production_path))
        payload = json.loads(graphless_path.read_text(encoding="utf-8"))
        payload["upper_graph"] = {"entry_point": 4, "max_level": 0, "nodes": []}
        production_path.write_text(json.dumps(payload), encoding="utf-8")
        Path(f"{production_path}.sha256").write_text(
            module.sha256_path(production_path) + "\n", encoding="utf-8"
        )
        return ["mock-cargo", "orion_build_artifact"]

    def run_export(
        args,
        production_path,
        vectors_path,
        row_count,
        dimension,
        hits_path,
        manifest_path,
    ):
        calls.append(
            (
                "run_rust_upper_hits_export",
                production_path,
                vectors_path,
                row_count,
                dimension,
                hits_path,
                manifest_path,
            )
        )
        hits_path.write_bytes(b"hits")
        manifest_path.write_text("{}\n", encoding="utf-8")
        return ["mock-cargo", "orion_export_upper_hits"]

    def verify_export(
        args,
        production_path,
        vectors_path,
        hits_path,
        manifest_path,
        **kwargs,
    ):
        calls.append(("verify_attachment_export", kwargs))
        return {
            "artifact_sha256": module.sha256_path(production_path),
            "vectors_sha256": module.sha256_path(vectors_path),
            "hits_sha256": module.sha256_path(hits_path),
        }

    def load_hits(hits_path, **kwargs):
        calls.append(("load_point_to_l1s_from_upper_hits", hits_path, kwargs))
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
                "build_canonical_routing_state",
                len(train),
                selected_upper.copy(),
                attachments,
                initial_num_shards,
                kmeans_iters,
                kmeans_seed,
                topology_iters,
            )
        )
        return routing

    def write_graphless(
        train,
        selected_upper,
        point_to_shards,
        upper_owner_by_point,
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
                upper_owner_by_point,
                num_shards,
                kwargs,
            )
        )
        output_path.write_text(
            json.dumps(
                {
                    "format_version": 2,
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
    monkeypatch.setattr(
        module.experiment, "write_orion_numeric_vector_file", write_vectors
    )
    monkeypatch.setattr(
        module.experiment,
        "write_orion_upper_seed_graphless_artifact",
        write_upper_seed,
    )
    monkeypatch.setattr(module, "run_rust_builder", run_builder)
    monkeypatch.setattr(module, "run_rust_upper_hits_export", run_export)
    monkeypatch.setattr(module, "verify_attachment_export", verify_export)
    monkeypatch.setattr(module, "load_point_to_l1s_from_upper_hits", load_hits)
    monkeypatch.setattr(module.experiment, "build_canonical_routing_state", build_routing)
    monkeypatch.setattr(module.experiment, "write_orion_graphless_artifact", write_graphless)
    return calls, routing


def test_graphless_only_builds_one_qdrant_upper_graph_before_layout(monkeypatch, tmp_path):
    module = load_module()
    hdf5_path = tmp_path / "smoke.hdf5"
    output_dir = tmp_path / "layout"
    write_train_hdf5(module, hdf5_path)
    calls, routing = patch_algorithm_pipeline(module, monkeypatch)

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
        "--graphless-only",
    )

    summary = module.build(args)

    assert [call[0] for call in calls] == [
        "prepare_vectors_for_distance",
        "global_upper_indices",
        "write_orion_numeric_vector_file",
        "write_orion_upper_seed_graphless_artifact",
        "run_rust_builder",
        "run_rust_upper_hits_export",
        "verify_attachment_export",
        "load_point_to_l1s_from_upper_hits",
        "build_canonical_routing_state",
        "write_orion_graphless_artifact",
    ]
    assert sum(call[0] == "run_rust_builder" for call in calls) == 1
    routing_call = calls[8]
    assert routing_call[4:8] == (2, 10, 1, 50)
    assert len(routing_call) == 8
    graphless_call = calls[9]
    assert graphless_call[3] == routing.point_to_shards
    assert graphless_call[4] == routing.l1_to_shard
    assert graphless_call[5] == 3
    assert graphless_call[6] == {
        "generation": 7,
        "vector_distance": "cosine",
        "upper_k": 2,
        "upper_ef_search": 2,
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
    assert manifest["routing"]["physical_point_count"] == 7
    assert manifest["artifact_binding"]["layout_sha256"] == "a" * 64
    assert manifest["routing"]["fission_events"] == routing.fission_events
    assert manifest["parameters"]["attachment_search_ef"] == 2
    assert manifest["parameters"]["efs"] == 2
    assert manifest["parameters"]["upper_search_ef"] == 2
    assert manifest["parameters"]["attachment_navigator"] == (
        "qdrant_production_upper_graph"
    )
    assert manifest["parameters"]["single_upper_graph_build"] is True
    assert manifest["navigation_binding"]["single_upper_graph_build"] is True
    assert manifest["navigation_binding"]["upper_graph_sha256"]
    assert manifest["navigation_binding"]["attachments_sha256"]
    assert manifest["outputs"]["rust_builder_command"] == [
        "mock-cargo",
        "orion_build_artifact",
    ]
    assert manifest["outputs"]["rust_rebind_command"] is None
    assert manifest["parameters"]["enable_topology_refinement"] is True
    assert manifest["parameters"]["balance_mode"] == "none"
    assert manifest["parameters"]["enable_fission"] is False
    assert manifest["parameters"]["lower_hnsw_retains_non_base_layers"] is True
    checksum_lines = (output_dir / module.CHECKSUMS_NAME).read_text().splitlines()
    assert any(line.endswith(f"  {module.GRAPHLESS_NAME}") for line in checksum_lines)
    assert any(line.endswith(f"  {module.BUILD_MANIFEST_NAME}") for line in checksum_lines)


def test_full_mode_finalizes_memberships_without_a_second_graph_build(monkeypatch, tmp_path):
    module = load_module()
    hdf5_path = tmp_path / "full-smoke.hdf5"
    output_dir = tmp_path / "full-layout"
    write_train_hdf5(module, hdf5_path)
    calls, routing = patch_algorithm_pipeline(module, monkeypatch)
    captured = {}

    def fake_rebind(args, source_path, sidecar_path, production_path, *, mode):
        captured["rebind"] = (source_path, sidecar_path, production_path, mode)
        source = json.loads(source_path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        source.update(
            {
                "generation": sidecar["generation"],
                "layout_sha256": sidecar["layout_sha256"],
                "shard_count": sidecar["shard_count"],
                "physical_point_count": sidecar["physical_point_count"],
            }
        )
        for node, owner_shard in zip(
            source["upper_nodes"], sidecar["upper_owner_shards"], strict=True
        ):
            node["owner_shard"] = owner_shard
        production_path.write_text(json.dumps(source), encoding="utf-8")
        Path(f"{production_path}.sha256").write_text(
            module.sha256_path(production_path) + "\n",
            encoding="utf-8",
        )
        return ["mock-cargo", "orion_rebind_memberships", "--finalize-build"]

    def fake_finalized(source_path, graphless_path, production_path):
        return {
            "upper_graph_sha256": module.production_upper_graph_sha256(source_path),
            "source_artifact_sha256": module.sha256_path(source_path),
            "production_artifact_sha256": module.sha256_path(production_path),
        }

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
        (target_dir / f"{prefix}.assignments.jsonl").write_text(
            '{"id":0,"shards":[0]}\n', encoding="utf-8"
        )
        manifest_path = target_dir / f"{prefix}.manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        return manifest_path

    monkeypatch.setattr(module, "run_rust_rebind", fake_rebind)
    monkeypatch.setattr(module, "verify_finalized_artifact", fake_finalized)
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

    assert sum(call[0] == "run_rust_builder" for call in calls) == 1
    assert captured["rebind"] == (
        output_dir / "upper-source-generation-9.json",
        output_dir / module.FINALIZATION_SIDECAR_NAME,
        output_dir / "generation-9.json",
        "finalize-build",
    )
    assert captured["bundle"]["train_rows"] == 6
    assert captured["bundle"]["point_to_shards"] == routing.point_to_shards
    assert captured["bundle"]["num_shards"] == routing.num_shards
    assert captured["bundle"]["orion_artifact_path"] == output_dir / "generation-9.json"
    assert captured["bundle"]["prefix"] == "native-smoke"
    assert captured["bundle"]["prewritten_vectors_path"] == (
        output_dir / "native-smoke.f32le"
    )
    assert summary["mode"] == "production_bundle"
    assert summary["production_artifact"] == str(output_dir / "generation-9.json")
    assert summary["import_manifest"] == str(output_dir / "native-smoke.manifest.json")
    manifest = json.loads((output_dir / module.BUILD_MANIFEST_NAME).read_text())
    assert manifest["outputs"]["rust_builder_command"] == [
        "mock-cargo",
        "orion_build_artifact",
    ]
    assert manifest["outputs"]["rust_rebind_command"] == [
        "mock-cargo",
        "orion_rebind_memberships",
        "--finalize-build",
    ]
    assert manifest["navigation_binding"]["graph_identity_verified_after_finalization"] is True
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


def test_canonical_attachment_and_finalization_commands_use_rust_examples(tmp_path):
    module = load_module()
    args = module.parse_args(
        [
            "--hdf5-path",
            str(tmp_path / "input.hdf5"),
            "--output-dir",
            str(tmp_path / "layout"),
            "--cargo",
            "/opt/rust/bin/cargo",
            "--k-overlap",
            "10",
            "--attachment-search-ef",
            "10",
        ]
    )
    source = tmp_path / "upper-source.json"
    vectors = tmp_path / "vectors.f32le"
    hits = tmp_path / "hits.u64le"
    attachment_manifest = tmp_path / "hits.json"
    sidecar = tmp_path / "memberships.json"
    production = tmp_path / "generation-1.json"

    assert module.rust_upper_hits_command(
        args,
        source,
        vectors,
        1000,
        200,
        hits,
        attachment_manifest,
    ) == [
        "/opt/rust/bin/cargo",
        "run",
        "--release",
        "-p",
        "collection",
        "--example",
        "orion_export_upper_hits",
        "--",
        str(source),
        str(vectors),
        "1000",
        "200",
        "10",
        "10",
        str(hits),
        str(attachment_manifest),
    ]
    assert module.rust_rebind_command(
        args,
        source,
        sidecar,
        production,
        mode="finalize-build",
    ) == [
        "/opt/rust/bin/cargo",
        "run",
        "--release",
        "-p",
        "collection",
        "--example",
        "orion_rebind_memberships",
        "--",
        str(source),
        str(sidecar),
        str(production),
        "--finalize-build",
    ]


def test_rust_builder_command_can_use_checksum_bound_prebuilt_binary(tmp_path):
    module = load_module()
    builder = tmp_path / "orion_build_artifact"
    builder.write_bytes(b"fixed-builder")
    builder.chmod(0o755)
    args = module.parse_args(
        [
            "--hdf5-path",
            str(tmp_path / "input.hdf5"),
            "--output-dir",
            str(tmp_path / "layout"),
            "--rust-builder-binary",
            str(builder),
            "--upper-graph-seed",
            "17",
            "--upper-m",
            "24",
            "--upper-ef-construction",
            "88",
        ]
    )

    module.validate_args(args)
    command = module.rust_builder_command(
        args,
        tmp_path / "graphless.json",
        tmp_path / "generation-1.json",
    )
    parameters = module.routing_parameters(args)

    assert command == [
        str(builder.resolve()),
        str(tmp_path / "graphless.json"),
        str(tmp_path / "generation-1.json"),
        "--seed",
        "17",
        "--m",
        "24",
        "--ef",
        "88",
    ]
    assert parameters["rust_builder_binary"] == str(builder.resolve())
    assert parameters["rust_builder_binary_sha256"] == module.sha256_path(builder)


def test_no_refinement_ablation_is_rejected_by_canonical_builder(monkeypatch, tmp_path):
    module = load_module()
    hdf5_path = tmp_path / "ablation-smoke.hdf5"
    output_dir = tmp_path / "ablation-layout"
    write_train_hdf5(module, hdf5_path)
    patch_algorithm_pipeline(module, monkeypatch)
    args = smoke_args(
        module,
        hdf5_path,
        output_dir,
        "--disable-topology-refinement",
        "--graphless-only",
    )

    with pytest.raises(ValueError, match="self-vote refinement"):
        module.build(args)


def test_capacity_balance_is_rejected_by_canonical_builder(monkeypatch, tmp_path):
    module = load_module()
    hdf5_path = tmp_path / "balanced-smoke.hdf5"
    output_dir = tmp_path / "balanced-layout"
    write_train_hdf5(module, hdf5_path)
    patch_algorithm_pipeline(module, monkeypatch)
    args = smoke_args(
        module,
        hdf5_path,
        output_dir,
        "--balance-mode",
        "capacity_constrained",
        "--balance-min-load-ratio",
        "0.95",
        "--balance-max-load-ratio",
        "1.05",
        "--balance-max-passes",
        "6",
        "--balance-max-vote-loss",
        "2",
        "--balance-l1-max-vote-loss",
        "1",
        "--balance-l0-max-vote-loss",
        "2",
        "--graphless-only",
    )

    with pytest.raises(ValueError, match="excludes load-balancing"):
        module.build(args)


def test_post_layout_capacity_balance_mode_is_rejected(tmp_path):
    module = load_module()
    args = module.parse_args(
        [
            "--hdf5-path",
            str(tmp_path / "input.hdf5"),
            "--output-dir",
            str(tmp_path / "layout"),
            "--balance-mode",
            "post_layout_capacity_constrained",
        ]
    )

    with pytest.raises(ValueError, match="excludes load-balancing"):
        module.validate_args(args)


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
        (
            ("--attachment-search-ef", "1", "--k-overlap", "2"),
            "attachment-search-ef",
        ),
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


@pytest.mark.parametrize(
    "extra",
    [
        (
            "--upper-k",
            "1",
            "--upper-search-ef",
            "1",
            "--k-overlap",
            "2",
            "--attachment-search-ef",
            "2",
        ),
        (
            "--upper-k",
            "8",
            "--upper-search-ef",
            "8",
            "--k-overlap",
            "2",
            "--attachment-search-ef",
            "2",
        ),
    ],
)
def test_attachment_and_runtime_upper_ef_are_validated_independently(
    tmp_path, extra
):
    module = load_module()
    args = smoke_args(
        module,
        tmp_path / "input.hdf5",
        tmp_path / "layout",
        *extra,
    )

    module.validate_args(args)


def test_runtime_upper_search_decoupling_requires_explicit_diagnostic_flag(tmp_path):
    module = load_module()
    args = smoke_args(
        module,
        tmp_path / "input.hdf5",
        tmp_path / "layout",
        "--upper-search-ef",
        "4",
    )

    with pytest.raises(ValueError, match="upper-search-ef == --upper-k"):
        module.validate_args(args)

    args.allow_decoupled_runtime_upper_search = True
    module.validate_args(args)


def test_counted_upper_hits_accepts_rows_shorter_than_requested_k(tmp_path):
    module = load_module()
    hits_path = tmp_path / "upper-attachments.counted.bin"
    hits_path.write_bytes(
        struct.pack("<I", 1)
        + struct.pack("<Q", 10)
        + struct.pack("<I", 3)
        + struct.pack("<QQQ", 20, 30, 40)
    )

    assert module.load_point_to_l1s_from_upper_hits(
        hits_path,
        row_count=2,
        top_k=4,
        upper_labels={10, 20, 30, 40},
    ) == [[10], [20, 30, 40]]


def test_counted_upper_hits_rejects_count_above_requested_k(tmp_path):
    module = load_module()
    hits_path = tmp_path / "upper-attachments.counted.bin"
    hits_path.write_bytes(struct.pack("<I", 3) + struct.pack("<QQQ", 10, 20, 30))

    with pytest.raises(RuntimeError, match="exceeding 2"):
        module.load_point_to_l1s_from_upper_hits(
            hits_path,
            row_count=1,
            top_k=2,
            upper_labels={10, 20, 30},
        )
