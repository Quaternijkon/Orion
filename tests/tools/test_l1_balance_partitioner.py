from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "experiments/l1_balance/l1_partitioner.py"
    spec = importlib.util.spec_from_file_location("l1_balance_partitioner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


partitioner = load_module()


def clustered_graph():
    adjacency = [set() for _ in range(12)]

    def connect(left: int, right: int) -> None:
        adjacency[left].add(right)
        adjacency[right].add(left)

    for start in (0, 4, 8):
        for node in range(start, start + 4):
            for other in range(node + 1, start + 4):
                connect(node, other)
    connect(3, 4)
    connect(7, 8)
    return [sorted(row) for row in adjacency]


def upper_vectors():
    return [
        [float(cluster * 100 + offset), float(offset % 2)]
        for cluster in range(3)
        for offset in range(4)
    ]


@pytest.mark.parametrize("algorithm", partitioner.SUPPORTED_ALGORITHMS)
def test_every_candidate_is_deterministic_and_exact_quota(algorithm: str) -> None:
    kwargs = {"vectors": upper_vectors()} if algorithm == "balanced-kmeans" else {}
    first = partitioner.partition_l1(
        clustered_graph(),
        3,
        algorithm=algorithm,
        entry_point=2,
        **kwargs,
    )
    second = partitioner.partition_l1(
        clustered_graph(),
        3,
        algorithm=algorithm,
        entry_point=2,
        **kwargs,
    )

    assert first.owner == second.owner
    assert first.edge_visits == second.edge_visits
    assert first.capacities == (4, 4, 4)
    assert first.partition_sizes == first.capacities
    assert set(first.owner) == {0, 1, 2}
    assert first.elapsed_seconds >= 0.0


def test_uneven_node_count_uses_floor_or_ceiling_exact_quota() -> None:
    result = partitioner.partition_l1(
        [[] for _ in range(10)],
        3,
        algorithm="random",
    )
    assert result.capacities == (4, 3, 3)
    assert result.partition_sizes == (4, 3, 3)


def test_public_api_has_no_l0_or_observed_balance_inputs() -> None:
    signature = inspect.signature(partitioner.partition_l1)
    assert not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    forbidden_parameters = {
        "attachments",
        "l0_attachments",
        "shard_load",
        "shard_loads",
        "actual_load",
        "multi_assignment",
        "memberships",
    }
    assert forbidden_parameters.isdisjoint(signature.parameters)
    assert set(signature.parameters) == {
        "adjacency",
        "num_partitions",
        "algorithm",
        "vectors",
        "entry_point",
    }

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        partitioner.partition_l1(
            clustered_graph(),
            3,
            algorithm="ldg0",
            l0_attachments=[[0]],
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        partitioner.partition_l1(
            clustered_graph(),
            3,
            algorithm="ldg0",
            actual_load=[4, 4, 4],
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        partitioner.partition_l1(
            clustered_graph(),
            3,
            algorithm="ldg0",
            multi_assignment={0: [0, 1]},
        )


@pytest.mark.parametrize(
    "algorithm",
    ["ldg0", "ldg-swap1", "fennel", "balanced-region-grow"],
)
def test_graph_only_algorithms_do_not_read_upper_vectors(algorithm: str) -> None:
    graph = clustered_graph()
    baseline = partitioner.partition_l1(
        graph,
        3,
        algorithm=algorithm,
        vectors=None,
        entry_point=0,
    )
    deliberately_wrong_shape = [[999.0]]
    changed = partitioner.partition_l1(
        graph,
        3,
        algorithm=algorithm,
        vectors=deliberately_wrong_shape,
        entry_point=0,
    )
    assert changed.owner == baseline.owner
    assert changed.edge_visits == baseline.edge_visits
    assert baseline.edge_visits > 0


def test_random_is_a_graph_agnostic_negative_control() -> None:
    empty = [[] for _ in range(12)]
    result_empty = partitioner.partition_l1(
        empty, 3, algorithm="random"
    )
    result_graph = partitioner.partition_l1(
        clustered_graph(), 3, algorithm="random"
    )
    assert result_empty.owner == result_graph.owner
    assert result_graph.edge_visits == 0


def test_balanced_kmeans_is_a_geometry_only_control() -> None:
    result_empty = partitioner.partition_l1(
        [[] for _ in range(12)],
        3,
        algorithm="balanced-kmeans",
        vectors=upper_vectors(),
    )
    result_graph = partitioner.partition_l1(
        clustered_graph(),
        3,
        algorithm="balanced-kmeans",
        vectors=upper_vectors(),
    )
    assert result_empty.owner == result_graph.owner
    assert result_graph.edge_visits == 0
    assert len({result_graph.owner[node] for node in range(4)}) == 1
    assert len({result_graph.owner[node] for node in range(4, 8)}) == 1
    assert len({result_graph.owner[node] for node in range(8, 12)}) == 1


def edge_cut(adjacency, owner) -> int:
    return sum(
        owner[left] != owner[right]
        for left, row in enumerate(adjacency)
        for right in row
        if left < right
    )


def test_ldg_swap1_preserves_quota_and_never_worsens_edge_cut() -> None:
    graph = clustered_graph()
    ldg0 = partitioner.partition_l1(graph, 3, algorithm="ldg0", entry_point=2)
    refined = partitioner.partition_l1(
        graph, 3, algorithm="ldg-swap1", entry_point=2
    )
    assert refined.partition_sizes == ldg0.partition_sizes == (4, 4, 4)
    assert edge_cut(graph, refined.owner) <= edge_cut(graph, ldg0.owner)
    assert refined.edge_visits >= ldg0.edge_visits


def test_input_graph_is_not_mutated_and_is_treated_as_undirected() -> None:
    directed = [[1], [], [1], []]
    before = [row.copy() for row in directed]
    result = partitioner.partition_l1(
        directed,
        2,
        algorithm="balanced-region-grow",
        entry_point=0,
    )
    assert directed == before
    assert result.partition_sizes == (2, 2)
    assert result.edge_visits > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_partitions": 0, "algorithm": "ldg0"}, "num_partitions"),
        ({"num_partitions": 5, "algorithm": "ldg0"}, "num_partitions"),
        ({"num_partitions": 2, "algorithm": "missing"}, "unknown algorithm"),
        (
            {"num_partitions": 2, "algorithm": "ldg0", "entry_point": 4},
            "entry_point",
        ),
    ],
)
def test_invalid_contract_inputs_fail_closed(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        partitioner.partition_l1([[], [], [], []], **kwargs)


def test_kmeans_requires_valid_upper_vectors() -> None:
    with pytest.raises(ValueError, match="requires upper vectors"):
        partitioner.partition_l1(
            [[], [], [], []], 2, algorithm="balanced-kmeans"
        )
    with pytest.raises(ValueError, match="shape|rows"):
        partitioner.partition_l1(
            [[], [], [], []],
            2,
            algorithm="balanced-kmeans",
            vectors=[[0.0], [1.0]],
        )


def test_fennel_gamma_is_fixed_not_a_public_tuning_parameter() -> None:
    assert partitioner.FENNEL_GAMMA == 1.5
    assert "gamma" not in inspect.signature(partitioner.partition_l1).parameters


def test_random_and_kmeans_tuning_is_fixed_not_public() -> None:
    assert partitioner.DETERMINISTIC_SEED == 0
    assert partitioner.KMEANS_ITERATIONS == 8
    parameters = inspect.signature(partitioner.partition_l1).parameters
    assert "seed" not in parameters
    assert "kmeans_iterations" not in parameters
