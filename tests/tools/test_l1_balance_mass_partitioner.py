from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "experiments/l1_balance/l1_mass_partitioner.py"
    spec = importlib.util.spec_from_file_location(
        "l1_balance_mass_partitioner_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


partitioner = load_module()


def clustered_graph() -> list[list[int]]:
    adjacency = [set() for _ in range(24)]

    def connect(left: int, right: int) -> None:
        adjacency[left].add(right)
        adjacency[right].add(left)

    for start in (0, 6, 12, 18):
        for node in range(start, start + 6):
            for other in range(node + 1, start + 6):
                connect(node, other)
    connect(5, 6)
    connect(11, 12)
    connect(17, 18)
    return [sorted(row) for row in adjacency]


def upper_vectors() -> list[list[float]]:
    return [
        [float(cluster * 100 + offset), float(offset % 2)]
        for cluster in range(4)
        for offset in range(6)
    ]


def skewed_upper_navigation_rows() -> list[list[int]]:
    """Twenty-four L1 self-queries with nine fixed popular neighbours."""

    rows: list[list[int]] = []
    for query in range(24):
        row = [query]
        for node in range(9):
            if node not in row:
                row.append(node)
        fallback = 9
        while len(row) < partitioner.UPPER_NAVIGATION_TOP_K:
            if fallback not in row:
                row.append(fallback)
            fallback += 1
        rows.append(row)
    return rows


def navigation_mass():
    return partitioner.estimate_upper_navigation_mass(
        skewed_upper_navigation_rows(), 24
    )


def test_mass_is_fixed_top10_hit_frequency_from_all_upper_queries() -> None:
    mass = navigation_mass()

    assert mass.source == partitioner.MASS_SOURCE
    assert mass.transform == "max(1, raw_hit_count - 1)"
    assert mass.query_count == 24
    assert mass.top_k == 10
    assert mass.total_mass == 230
    assert mass.values[:9] == (23,) * 9
    assert mass.values[9] == 9
    assert mass.values[10:] == (1,) * 14
    assert len(mass.sha256) == 64


def test_raw_mass_is_an_explicit_ablation_with_a_different_checksum() -> None:
    rows = skewed_upper_navigation_rows()
    main = partitioner.estimate_upper_navigation_mass(rows, 24)
    raw = partitioner.estimate_raw_upper_navigation_mass(rows, 24)

    assert raw.source == partitioner.RAW_MASS_SOURCE
    assert raw.transform == "raw_hit_count"
    assert raw.total_mass == 240
    assert raw.values[:9] == (24,) * 9
    assert raw.values[9] == 10
    assert raw.values[10:] == (1,) * 14
    assert raw.sha256 != main.sha256


def test_v2_regularized_mass_has_raw_values_but_a_distinct_semantic_identity() -> None:
    rows = skewed_upper_navigation_rows()
    raw = partitioner.estimate_raw_upper_navigation_mass(rows, 24)
    regularized = partitioner.estimate_regularized_upper_navigation_mass(rows, 24)

    assert regularized.values == raw.values
    assert regularized.total_mass == raw.total_mass == 240
    assert regularized.source == partitioner.REGULARIZED_MASS_SOURCE
    assert regularized.transform == "raw_count_with_unit_l1_prior"
    assert regularized.estimator_version == 2
    assert regularized.sha256 != raw.sha256


def test_navigation_mass_cannot_be_directly_constructed_or_replaced_by_loads() -> None:
    with pytest.raises(TypeError, match="derived state"):
        partitioner.UpperNavigationMass()
    with pytest.raises(TypeError, match="estimate_upper_navigation_mass"):
        partitioner.partition_l1_by_navigation_mass(
            clustered_graph(),
            4,
            algorithm="mass-ldg",
            navigation_mass=[60, 60, 60, 60],
        )


@pytest.mark.parametrize("algorithm", partitioner.MASS_SUPPORTED_ALGORITHMS)
def test_every_mass_candidate_is_deterministic_nonempty_and_bounded(
    algorithm: str,
) -> None:
    kwargs = {"vectors": upper_vectors()} if algorithm.endswith("kmeans") else {}
    first = partitioner.partition_l1_by_navigation_mass(
        clustered_graph(),
        4,
        algorithm=algorithm,
        navigation_mass=navigation_mass(),
        entry_point=2,
        **kwargs,
    )
    second = partitioner.partition_l1_by_navigation_mass(
        clustered_graph(),
        4,
        algorithm=algorithm,
        navigation_mass=navigation_mass(),
        entry_point=2,
        **kwargs,
    )

    assert first.owner == second.owner
    assert first.estimated_partition_masses == second.estimated_partition_masses
    assert first.edge_visits == second.edge_visits
    assert sum(first.partition_sizes) == 24
    assert min(first.partition_sizes) > 0
    assert max(first.estimated_partition_masses) <= first.mass_limit
    assert sum(first.estimated_partition_masses) == 230
    assert first.mass_sha256 == navigation_mass().sha256
    assert first.elapsed_seconds >= 0.0


def test_weighted_capacity_allows_unequal_l1_node_counts() -> None:
    result = partitioner.partition_l1_by_navigation_mass(
        clustered_graph(),
        4,
        algorithm="mass-balanced-kmeans",
        navigation_mass=navigation_mass(),
        vectors=upper_vectors(),
    )

    assert len(set(result.partition_sizes)) > 1
    assert result.partition_sizes != (6, 6, 6, 6)


def test_public_apis_have_no_l0_or_observed_balance_inputs() -> None:
    estimate_signature = inspect.signature(
        partitioner.estimate_upper_navigation_mass
    )
    partition_signature = inspect.signature(
        partitioner.partition_l1_by_navigation_mass
    )
    assert set(estimate_signature.parameters) == {"navigation_rows", "node_count"}
    raw_estimate_signature = inspect.signature(
        partitioner.estimate_raw_upper_navigation_mass
    )
    assert set(raw_estimate_signature.parameters) == {
        "navigation_rows",
        "node_count",
    }
    regularized_estimate_signature = inspect.signature(
        partitioner.estimate_regularized_upper_navigation_mass
    )
    assert set(regularized_estimate_signature.parameters) == {
        "navigation_rows",
        "node_count",
    }
    assert set(partition_signature.parameters) == {
        "adjacency",
        "num_partitions",
        "algorithm",
        "navigation_mass",
        "vectors",
        "entry_point",
    }
    assert not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in partition_signature.parameters.values()
    )
    forbidden = {
        "attachments",
        "l0_attachments",
        "actual_load",
        "shard_load",
        "shard_loads",
        "multi_assignment",
        "memberships",
        "copy_count",
    }
    assert forbidden.isdisjoint(estimate_signature.parameters)
    assert forbidden.isdisjoint(partition_signature.parameters)

    for name, value in (
        ("l0_attachments", [[0]]),
        ("actual_load", [1, 2, 3, 4]),
        ("multi_assignment", {0: [0, 1]}),
    ):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            partitioner.partition_l1_by_navigation_mass(
                clustered_graph(),
                4,
                algorithm="mass-ldg",
                navigation_mass=navigation_mass(),
                **{name: value},
            )


@pytest.mark.parametrize("algorithm", ["mass-ldg", "mass-region-grow"])
def test_mass_graph_algorithms_do_not_read_upper_vectors(algorithm: str) -> None:
    baseline = partitioner.partition_l1_by_navigation_mass(
        clustered_graph(),
        4,
        algorithm=algorithm,
        navigation_mass=navigation_mass(),
        vectors=None,
        entry_point=0,
    )
    changed = partitioner.partition_l1_by_navigation_mass(
        clustered_graph(),
        4,
        algorithm=algorithm,
        navigation_mass=navigation_mass(),
        vectors=[[999.0]],
        entry_point=0,
    )

    assert changed.owner == baseline.owner
    assert changed.edge_visits == baseline.edge_visits
    assert baseline.edge_visits > 0


def test_mass_kmeans_is_geometry_and_mass_only() -> None:
    empty = [[] for _ in range(24)]
    baseline = partitioner.partition_l1_by_navigation_mass(
        empty,
        4,
        algorithm="mass-balanced-kmeans",
        navigation_mass=navigation_mass(),
        vectors=upper_vectors(),
    )
    with_graph = partitioner.partition_l1_by_navigation_mass(
        clustered_graph(),
        4,
        algorithm="mass-balanced-kmeans",
        navigation_mass=navigation_mass(),
        vectors=upper_vectors(),
    )

    assert with_graph.owner == baseline.owner
    assert with_graph.edge_visits == baseline.edge_visits == 0


@pytest.mark.parametrize(
    ("rows", "node_count", "message"),
    [
        ([[0] * 10] * 24, 24, "repeats"),
        ([list(range(9))] * 24, 24, "fixed top-k"),
        ([list(range(10))] * 23, 24, "query count"),
        ([list(range(9)) + [24]] * 24, 24, "out-of-range"),
        ([list(range(10))] * 24, 24, "self hit"),
    ],
)
def test_invalid_upper_navigation_rows_fail_closed(rows, node_count, message) -> None:
    with pytest.raises(ValueError, match=message):
        partitioner.estimate_upper_navigation_mass(rows, node_count)


def test_mass_parameters_are_fixed_not_public_tuning_knobs() -> None:
    assert partitioner.UPPER_NAVIGATION_TOP_K == 10
    assert partitioner.MASS_DETERMINISTIC_SEED == 0
    assert partitioner.MASS_KMEANS_ITERATIONS == 8
    parameters = inspect.signature(
        partitioner.partition_l1_by_navigation_mass
    ).parameters
    assert "seed" not in parameters
    assert "mass_limit" not in parameters
    assert "kmeans_iterations" not in parameters
    assert "top_k" not in parameters
