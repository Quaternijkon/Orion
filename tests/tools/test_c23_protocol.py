from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "experiments/c23/scripts/c23_protocol.py"
    spec = importlib.util.spec_from_file_location("c23_protocol", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_e1_module():
    path = REPO_ROOT / "experiments/c23/scripts/c23_e1.py"
    spec = importlib.util.spec_from_file_location("c23_e1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_e2_module():
    path = REPO_ROOT / "experiments/c23/scripts/c23_e2.py"
    spec = importlib.util.spec_from_file_location("c23_e2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_local_analysis_module():
    path = REPO_ROOT / "experiments/c23/scripts/c23_local_analysis.py"
    spec = importlib.util.spec_from_file_location("c23_local_analysis", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_bootstrap_module():
    path = REPO_ROOT / "experiments/c23/scripts/c23_bootstrap.py"
    spec = importlib.util.spec_from_file_location("c23_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_tiny_dataset(path: Path) -> str:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=np.arange(24, dtype=np.float32).reshape(12, 2))
        handle.create_dataset("test", data=np.arange(8, dtype=np.float32).reshape(4, 2))
        handle.create_dataset(
            "neighbors",
            data=np.tile(np.arange(10, dtype=np.int64), (4, 1)),
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dataset_audit_enforces_checksum_shapes_and_disjoint_split(tmp_path):
    module = load_module()
    path = tmp_path / "tiny.hdf5"
    checksum = write_tiny_dataset(path)
    spec = module.DatasetSpec(
        name="tiny",
        filename=path.name,
        metric="euclid",
        hnsw_space="l2",
        dimension=2,
        expected_train_rows=12,
        expected_query_rows=4,
        sha256=checksum,
    )

    audit = module.dataset_audit(path, spec)

    assert audit["train_shape"] == (12, 2)
    assert audit["test_shape"] == (4, 2)
    assert audit["neighbors_shape"] == (4, 10)
    assert audit["tuning_query_range"] == [0, 1000]
    assert audit["measurement_query_count"] == 0


def test_partition_roundtrip_rejects_empty_shards(tmp_path):
    module = load_module()
    artifact = tmp_path / "orion-m2.npz"
    metadata = {
        "method": "orion",
        "logical_shards": 2,
        "point_count": 6,
        "shard_counts": [3, 3],
    }
    record = module.save_partition(
        artifact,
        np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32),
        metadata,
    )

    assignments, loaded = module.load_partition(artifact, "orion")

    assert assignments.tolist() == [0, 1, 0, 1, 0, 1]
    assert loaded["shard_counts"] == [3, 3]
    assert record["artifact_sha256"] == module.sha256_path(artifact)

    with np.testing.assert_raises_regex(ValueError, "empty shards"):
        module.save_partition(
            tmp_path / "invalid.npz",
            np.zeros(6, dtype=np.int32),
            metadata,
        )


def test_reference_input_export_preserves_raw_rows(tmp_path):
    module = load_module()
    source = tmp_path / "tiny.hdf5"
    checksum = write_tiny_dataset(source)
    spec = module.DatasetSpec(
        name="tiny",
        filename=source.name,
        metric="euclid",
        hnsw_space="l2",
        dimension=2,
        expected_train_rows=12,
        expected_query_rows=4,
        sha256=checksum,
    )
    output = tmp_path / "raw"

    manifest = module.export_reference_input(source, spec, output, chunk_size=3)

    vectors = np.fromfile(output / "vectors.f32le", dtype="<f4").reshape(12, 2)
    queries = np.fromfile(output / "queries.f32le", dtype="<f4").reshape(4, 2)
    truth = np.fromfile(output / "ground_truth.u32le", dtype="<u4").reshape(4, 10)
    assert vectors.tolist() == np.arange(24, dtype=np.float32).reshape(12, 2).tolist()
    assert queries.tolist() == np.arange(8, dtype=np.float32).reshape(4, 2).tolist()
    assert truth[0].tolist() == list(range(10))
    assert manifest["files"]["vectors"]["size_bytes"] == 12 * 2 * 4


def test_original_orion_builder_can_skip_topology_refinement(monkeypatch):
    module = load_module()
    experiment = module.orion
    train = np.zeros((4, 2), dtype=np.float32)
    upper_indices = np.asarray([0, 2], dtype=np.int64)
    point_to_l1s = [[0], [0], [2], [2]]

    monkeypatch.setattr(
        experiment,
        "initial_l1_shards_by_balanced_kmeans",
        lambda *_args, **_kwargs: [0, -1, 1, -1],
    )

    def must_not_refine(*_args, **_kwargs):
        raise AssertionError("P2-A must skip converge_l1_topology")

    monkeypatch.setattr(experiment, "converge_l1_topology", must_not_refine)
    monkeypatch.setattr(
        experiment,
        "recalibrate_l1_weights_by_voting",
        lambda *_args, **_kwargs: np.asarray([2, 2], dtype=np.int64),
    )
    monkeypatch.setattr(
        experiment,
        "assign_points_by_l1_vote",
        lambda *_args, **_kwargs: (
            np.asarray([0, 0, 1, 1], dtype=np.int32),
            [[0], [0], [1], [1]],
        ),
    )

    state = experiment.build_original_routing_state(
        train,
        upper_indices,
        point_to_l1s,
        2,
        2,
        1,
        5,
        use_multi_assign=False,
        enable_fission=False,
        enable_topology_refinement=False,
    )

    assert state.topology_iterations == 0
    assert state.primary_shards.tolist() == [0, 0, 1, 1]
    assert state.expansion_ratio == 1.0


def test_reference_entry_point_calculation_matches_production_configuration():
    module = load_module()

    assert module.expected_hnsw_entry_point_parameters(1_000_000, 128, 10) == {
        "average_vector_bytes": 512,
        "full_scan_threshold_points": 20,
        "entry_points_num": 500_000,
    }
    assert module.expected_hnsw_entry_point_parameters(2_000, 16, 10) == {
        "average_vector_bytes": 64,
        "full_scan_threshold_points": 160,
        "entry_points_num": 120,
    }


def test_e1_metrics_reproduce_edge_cut_retention_and_connectivity():
    module = load_e1_module()
    sources = np.asarray([0, 1, 1, 2, 2, 3], dtype=np.uint32)
    destinations = np.asarray([1, 0, 2, 1, 3, 2], dtype=np.uint32)
    assignments = np.asarray([0, 0, 1, 1], dtype=np.int32)
    degree = np.bincount(sources.astype(np.int64), minlength=4)

    metrics, connectivity = module.compute_e1_for_partition(
        assignments,
        sources,
        destinations,
        degree,
        2,
        edge_chunk_size=2,
    )

    assert metrics["cut_edge_count"] == 2
    assert metrics["edge_cut_ratio"] == 2 / 6
    assert metrics["mean_intra_shard_degree"] == 1.0
    assert metrics["mean_retained_degree_ratio"] == 0.75
    assert metrics["fraction_losing_gt_25pct_neighbors"] == 0.5
    assert metrics["fraction_losing_gt_50pct_neighbors"] == 0.0
    assert [row["connected_components"] for row in connectivity] == [1, 1]
    assert [row["largest_component_fraction"] for row in connectivity] == [1.0, 1.0]
    assert [row["isolated_node_fraction"] for row in connectivity] == [0.0, 0.0]


def test_e2_path_and_query_unique_edge_metrics():
    module = load_e2_module()
    assignments = np.asarray([0, 0, 1, 1], dtype=np.int32)
    visited = np.asarray([0, 1, 2, 3], dtype=np.int64)

    path_shards, transitions, concentration = module.compute_path_metrics(
        assignments,
        visited,
    )
    keys = module.trace_edge_keys(
        [
            {"source_node": 0, "destination_node": 1},
            {"source_node": 1, "destination_node": 2},
            {"source_node": 1, "destination_node": 2},
        ],
        4,
    )

    assert path_shards == 2
    assert transitions == 1
    assert concentration == 0.5
    assert len(keys) == 2
    assert module.cut_counts(assignments[None, :], keys).tolist() == [1]


def test_e4_minimum_shard_fanout_uses_largest_contributions():
    module = load_local_analysis_module()

    assert module.minimum_shards_for_nine(np.asarray([5, 4, 1])) == 2
    assert module.minimum_shards_for_nine(np.asarray([3, 3, 3, 1])) == 3
    assert module.minimum_shards_for_nine(np.asarray([2, 2, 2])) == 4


def test_paired_bootstrap_is_deterministic_and_uses_orion_minus_kmeans():
    module = load_bootstrap_module()
    orion = np.asarray([1.0, 2.0, 3.0])
    kmeans = np.asarray([2.0, 3.0, 4.0])

    first = module.paired_bootstrap(orion, kmeans, seed=17, replicates=1000)
    second = module.paired_bootstrap(orion, kmeans, seed=17, replicates=1000)

    assert first == second
    assert first[0] == -1.0
    assert first[1] == -1.0
    assert first[2] == -1.0


def test_summary_schema_contains_protocol_required_fields():
    header = (
        REPO_ROOT / "experiments/c23/runs/summary.csv"
    ).read_text(encoding="utf-8").strip().split(",")
    required = {
        "experiment_id",
        "dataset_checksum",
        "logical_to_physical_mapping",
        "traversal_weighted_edge_cut",
        "mean_local_target_recall",
        "P_exact_mean",
        "P_HNSW_mean",
        "Delta_P_mean",
        "W90_mean",
    }
    assert required.issubset(header)
