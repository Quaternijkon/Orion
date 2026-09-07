from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - repository test env provides NumPy
    np = None


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(relative: str, name: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


awp = load_module(
    "experiments/l1_balance/attraction_weighted_partitioner.py",
    "attraction_weighted_partitioner_test",
)
prepare = load_module(
    "experiments/l1_balance/prepare_attraction_weighted_metis.py",
    "prepare_attraction_weighted_metis_test",
)
matrix = load_module(
    "experiments/multi_assignment/screen_owner_policy_matrix.py",
    "screen_owner_policy_matrix_attraction_test",
)


def clustered_graph() -> list[list[int]]:
    return [
        [1, 2],
        [0, 2],
        [0, 1, 3],
        [2, 4, 5],
        [3, 5],
        [3, 4],
    ]


def calibration_rows() -> list[list[int]]:
    return [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 3],
        [2, 0, 3],
        [3, 4, 5],
        [3, 5, 4],
        [4, 3, 5],
        [4, 5, 3],
        [5, 4, 3],
        [5, 3, 4],
    ]


def test_sample_attraction_is_smoothed_deterministic_and_exact_total() -> None:
    first = awp.estimate_attraction_weights(
        [6, 3, 1, 0],
        calibration_count=10,
        logical_point_count=100,
    )
    second = awp.estimate_attraction_weights(
        [6, 3, 1, 0],
        calibration_count=10,
        logical_point_count=100,
    )

    assert first == second
    assert first.estimator == awp.SAMPLE_ESTIMATOR
    assert sum(first.values) == 100
    assert min(first.values) >= 1
    assert first.values[0] > first.values[1] > first.values[2] > first.values[3]
    assert len(first.semantic_sha256) == 64


def test_exact_attraction_requires_one_row_per_logical_point() -> None:
    exact = awp.estimate_attraction_weights(
        [2, 1, 1],
        calibration_count=4,
        logical_point_count=4,
        estimator=awp.EXACT_ESTIMATOR,
    )
    assert exact.values == (2, 1, 1)

    with pytest.raises(ValueError, match="calibration_count"):
        awp.estimate_attraction_weights(
            [2, 1, 1],
            calibration_count=4,
            logical_point_count=5,
            estimator=awp.EXACT_ESTIMATOR,
        )


def test_search_weights_only_strengthen_existing_upper_edges() -> None:
    measurements = awp.measure_calibration_rows(
        clustered_graph(), calibration_rows()
    )
    weighted = awp.build_search_weighted_graph(clustered_graph(), measurements)
    original = {
        (left, right)
        for left, row in enumerate(awp.normalise_undirected_graph(clustered_graph()))
        for right in row
        if left < right
    }
    observed = {
        (left, right)
        for left, row in enumerate(weighted.adjacency)
        for right, _weight in row
        if left < right
    }

    assert observed == original
    assert weighted.edge_count == len(original)
    assert weighted.search_overlay_edge_count == 0
    assert weighted.total_edge_weight > weighted.edge_count
    bridge_weight = dict(weighted.adjacency[2])[3]
    dense_weight = dict(weighted.adjacency[0])[1]
    assert dense_weight > bridge_weight


def test_top1_star_adds_only_explicit_search_induced_overlay_edges() -> None:
    rows = [[0, 3, 4], [0, 3, 5], [5, 0, 1]]
    measurements = awp.measure_calibration_rows(
        clustered_graph(), rows, edge_mode=awp.TOP1_STAR_EDGE_MODE
    )
    weighted = awp.build_search_weighted_graph(clustered_graph(), measurements)

    assert weighted.edge_mode == awp.TOP1_STAR_EDGE_MODE
    assert weighted.search_overlay_edge_count > 0
    assert dict(weighted.adjacency[0])[3] == 3
    assert dict(weighted.adjacency[0])[4] == 2


def test_metis_encoding_contains_vertex_and_edge_weights() -> None:
    measurements = awp.measure_calibration_rows(
        clustered_graph(), calibration_rows()
    )
    graph = awp.build_search_weighted_graph(clustered_graph(), measurements)
    weights = awp.estimate_attraction_weights(
        measurements.first_hit_counts,
        calibration_count=measurements.row_count,
        logical_point_count=60,
    )
    text = awp.metis_graph_bytes(graph, weights).decode("ascii").splitlines()

    assert text[0] == f"6 {graph.edge_count} 011 1"
    assert len(text) == 7
    assert int(text[1].split()[0]) == weights.values[0]
    assert len(text[1].split()) == 1 + 2 * len(graph.adjacency[0])


def test_metis_backend_uses_tighter_internal_ufactor() -> None:
    assert prepare.metis_ufactor(0.02) == 19
    assert prepare.metis_ufactor(0.01) == 9


def test_partition_validation_checks_balance_heavy_atoms_and_weighted_cut() -> None:
    measurements = awp.measure_calibration_rows(
        clustered_graph(), calibration_rows()
    )
    graph = awp.build_search_weighted_graph(clustered_graph(), measurements)
    weights = awp.estimate_attraction_weights(
        [2, 2, 2, 2, 2, 2],
        calibration_count=12,
        logical_point_count=60,
    )
    result = awp.validate_partition(
        [0, 0, 0, 1, 1, 1],
        weights,
        graph,
        2,
        imbalance_tolerance=0.02,
        heavy_atom_fraction=0.40,
    )

    assert result.balance_pass is True
    assert result.heavy_atom_pass is True
    assert result.partition_loads == (30, 30)
    assert result.unweighted_cut_edges == 1
    assert result.weighted_cut == dict(graph.adjacency[2])[3]


def test_attraction_refinement_is_deterministic_and_cut_bounded() -> None:
    measurements = awp.measure_calibration_rows(
        clustered_graph(), calibration_rows()
    )
    weights = awp.estimate_attraction_weights(
        measurements.first_hit_counts,
        calibration_count=measurements.row_count,
        logical_point_count=60,
    )
    initial = [0, 0, 0, 0, 1, 1]

    first = awp.refine_owner_by_attraction(
        clustered_graph(), initial, weights, 2
    )
    second = awp.refine_owner_by_attraction(
        clustered_graph(), initial, weights, 2
    )

    assert first == second
    assert first.initial_partition_loads == (40, 20)
    assert first.final_partition_loads == (30, 30)
    assert first.moved_node_count == 1
    assert first.final_cut_edges <= first.cut_limit
    assert first.owner == (0, 0, 0, 1, 1, 1)


def test_navigation_refinement_obeys_vote_loss_and_eight_percent_budget() -> None:
    weights = awp.AttractionWeights(
        estimator=awp.EXACT_ESTIMATOR,
        values=(200, 100, 100, 40, 80, 80),
        calibration_count=600,
        logical_point_count=600,
        total_weight=600,
        max_weight=200,
        mean_weight=100.0,
        semantic_sha256="fixture",
    )
    navigation_rows = [
        [0, 1, 2],
        [1, 0, 2],
        [2, 1, 3],
        [3, 4, 5],
        [4, 3, 5],
        [5, 3, 4],
    ]

    result = awp.refine_owner_by_attraction_navigation(
        clustered_graph(),
        navigation_rows,
        [0, 0, 0, 0, 1, 1],
        weights,
        2,
    )

    assert result.owner == (0, 0, 0, 1, 1, 1)
    assert result.moved_attraction_weight == 40
    assert result.moved_attraction_weight <= 600 * 2 // 25
    assert result.rounds[0]["maximum_accepted_navigation_vote_loss"] <= 1


def test_navigation_refinement_ignores_zero_attraction_nodes() -> None:
    weights = awp.AttractionWeights(
        estimator=awp.EXACT_ESTIMATOR,
        values=(200, 100, 100, 0, 100, 100),
        calibration_count=600,
        logical_point_count=600,
        total_weight=600,
        max_weight=200,
        mean_weight=100.0,
        semantic_sha256="fixture-zero",
    )
    rows = [[0, 1], [1, 0], [2, 3], [3, 4], [4, 3], [5, 4]]

    result = awp.refine_owner_by_attraction_navigation(
        clustered_graph(), rows, [0, 0, 0, 0, 1, 1], weights, 2
    )

    assert result.owner[3] == 0
    assert result.moved_attraction_weight == 0


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_prepare_cli_imports_and_validates_a_precomputed_partition(tmp_path: Path) -> None:
    labels = [100, 101, 102, 103]
    graph_rows = [[101, 103], [100, 102], [101, 103], [102, 100]]
    artifact_path = tmp_path / "artifact.json"
    artifact = {
        "generation": 7,
        "logical_point_count": 40,
        "shard_count": 2,
        "upper_nodes": [
            {"label": label, "vector": [float(index), 0.0], "shard_membership": []}
            for index, label in enumerate(labels)
        ],
        "upper_graph": {
            "entry_point": 100,
            "nodes": [
                {"label": label, "neighbors_by_level": [neighbours]}
                for label, neighbours in zip(labels, graph_rows)
            ],
        },
    }
    write_json(artifact_path, artifact)
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    hit_rows = [
        (100, 101),
        (101, 100),
        (102, 103),
        (103, 102),
    ] * 4
    hits_path = tmp_path / "hits.u64le"
    with hits_path.open("wb") as handle:
        for row in hit_rows:
            handle.write(struct.pack("<QQ", *row))
    hits_sha = hashlib.sha256(hits_path.read_bytes()).hexdigest()
    calibration_manifest = tmp_path / "calibration.json"
    vectors_sha256 = "fixture-calibration-vectors"
    write_json(
        calibration_manifest,
        {
            "format_version": 1,
            "artifact_path": str(artifact_path.resolve()),
            "artifact_sha256": artifact_sha,
            "generation": 7,
            "upper_graph_present": True,
            "source_upper_k": 2,
            "source_upper_ef_search": 10,
            "vectors_path": str((tmp_path / "vectors.f32le").resolve()),
            "vectors_sha256": vectors_sha256,
            "row_count": len(hit_rows),
            "dimension": 2,
            "top_k": 2,
            "search_ef": 10,
            "hits_path": str(hits_path.resolve()),
            "hits_sha256": hits_sha,
            "hits_size_bytes": hits_path.stat().st_size,
            "elapsed_seconds": 0.1,
        },
    )
    selection_manifest = tmp_path / "selection.json"
    write_json(
        selection_manifest,
        {
            "format_version": 1,
            "selection_method": "stable_hash_point_id_v1",
            "seed": 42,
            "row_count": len(hit_rows),
            "vectors_sha256": vectors_sha256,
        },
    )
    partition_path = tmp_path / "owner.part"
    partition_path.write_text("0\n0\n1\n1\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    manifest_path = prepare.run(
        prepare.parse_args(
            [
                "--artifact",
                str(artifact_path),
                "--calibration-hits",
                str(hits_path),
                "--calibration-manifest",
                str(calibration_manifest),
                "--calibration-selection-manifest",
                str(selection_manifest),
                "--placement-search-ef",
                "10",
                "--num-partitions",
                "2",
                "--heavy-atom-fraction",
                "0.6",
                "--partition-file",
                str(partition_path),
                "--output-dir",
                str(output_dir),
            ]
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "PASS"
    assert manifest["contract"]["upper_hnsw_mutated"] is False
    assert manifest["parameters"]["num_partitions"] == 2
    assert manifest["owner"]["validation"]["balance_pass"] is True
    assert manifest["owner"]["validation"]["heavy_atom_pass"] is True
    assert (output_dir / prepare.METIS_GRAPH_NAME).is_file()
    assert (output_dir / prepare.COUNTS_NAME).stat().st_size == 4 * 8
    assert (output_dir / prepare.WEIGHTS_NAME).stat().st_size == 4 * 8
    assert (output_dir / prepare.OWNER_NAME).stat().st_size == 4 * 4
    assert (output_dir / f"{prepare.METIS_GRAPH_NAME}.part.2").is_file()
    assert (output_dir / "checksums.sha256").is_file()


@pytest.mark.skipif(np is None, reason="NumPy is required by the owner screen")
def test_owner_policy_screen_accepts_frozen_owner_binary(tmp_path: Path) -> None:
    owner_path = tmp_path / prepare.OWNER_NAME
    np.asarray([0, 0, 1, 1], dtype="<i4").tofile(owner_path)
    manifest_path = tmp_path / prepare.MANIFEST_NAME
    write_json(
        manifest_path,
        {
            "record_type": "orion_attraction_weighted_search_graph_owner",
            "parameters": {"num_partitions": 2},
            "source": {
                "upper_node_count": 4,
                "artifact_sha256": "artifact-sha",
                "upper_graph_sha256": "graph-sha",
            },
            "contract": {"upper_hnsw_mutated": False},
            "owner": {
                "sha256": hashlib.sha256(owner_path.read_bytes()).hexdigest()
            },
        },
    )
    reference_labels = np.asarray([100, 101, 102, 103], dtype="<u8")

    owner, source = matrix.binary_owner(
        owner_path,
        reference_labels=reference_labels,
        num_partitions=2,
        reference_artifact_sha256="artifact-sha",
        reference_upper_graph_sha256="graph-sha",
    )

    assert owner.tolist() == [0, 0, 1, 1]
    assert source["source_kind"] == "frozen_i32le_owner_binary"
    assert source["owner_generator_contract"]["upper_hnsw_mutated"] is False


def test_owner_inputs_reject_duplicate_names_across_artifact_and_binary(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"x")
    second.write_bytes(b"x")
    with pytest.raises(ValueError, match="duplicate"):
        matrix.parse_named_paths(
            [f"CANDIDATE={first}"],
            [f"CANDIDATE={second}"],
        )


@pytest.mark.skipif(np is None, reason="NumPy is required by the owner screen")
def test_owner_binary_without_identity_manifest_fails_closed(
    tmp_path: Path,
) -> None:
    owner_path = tmp_path / "owner.i32le"
    np.asarray([0, 0, 1, 1], dtype="<i4").tofile(owner_path)

    with pytest.raises(ValueError, match="identity binding"):
        matrix.binary_owner(
            owner_path,
            reference_labels=np.asarray([100, 101, 102, 103], dtype="<u8"),
            num_partitions=2,
            reference_artifact_sha256="artifact-sha",
            reference_upper_graph_sha256="graph-sha",
        )


def test_cross_owner_gates_reject_route_and_topology_regression() -> None:
    reference = {
        "owner_upper_edge_cut_ratio": 0.60,
        "owner_retained_degree_mean": 0.40,
        "owner_retained_degree_p10": 0.15,
        "owner_upper_isolated_fraction": 0.01,
        "owner_largest_component_fraction_mean": 0.99,
        "owner_largest_component_fraction_min": 0.90,
        "gt_routing_coverage_mean": 0.97,
        "gt_queries_full_coverage_fraction": 0.80,
        "gt_queries_zero_coverage": 0,
        "expansion_ratio": 1.10,
        "physical_copy_load_max": 100,
        "query_owner_transitions_mean": 20.0,
        "routed_shards_mean": 8.0,
        "route_entry_points_mean": 50.0,
        "route_ef_sum_mean": 1000.0,
    }
    observed = {
        **reference,
        "owner_upper_edge_cut_ratio": 0.70,
        "gt_routing_coverage_mean": 0.965,
        "routed_shards_mean": 9.0,
        "physical_copy_load_max": 70,
    }

    gates = matrix.canonical_cross_owner_gates(observed, reference, 1000)

    assert gates["physical_copy_load_max"]["pass"] is True
    assert gates["owner_upper_edge_cut_ratio"]["pass"] is False
    assert gates["gt_routing_coverage_mean"]["pass"] is False
    assert gates["routed_shards_mean"]["pass"] is False
