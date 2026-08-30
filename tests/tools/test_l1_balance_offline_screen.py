from __future__ import annotations

import numpy as np

from experiments.l1_balance.run_offline_screen import (
    physical_membership,
    query_metrics,
    topology_metrics,
)


def test_physical_membership_keeps_original_multi_assignment_rule() -> None:
    owner = np.asarray([0, 1, 2, 3], dtype=np.int32)
    attachment_local = np.asarray(
        [
            [0, 0, 1, 1],  # 2-2 tie: two physical copies
            [0, 0, 0, 1],  # 3-1: one copy in shard 0
            [0, 1, 0, 1],  # 2-2 tie despite interleaving
            [0, 1, 2, 3],  # all votes unique: first-hit fallback
        ],
        dtype=np.int32,
    )

    membership, copy_count, loads = physical_membership(
        owner, attachment_local, num_partitions=4
    )

    assert membership.tolist() == [
        [True, True, False, False],
        [True, False, False, False],
        [True, True, False, False],
        [True, False, False, False],
    ]
    assert copy_count.tolist() == [2, 1, 2, 1]
    assert loads.tolist() == [4, 2, 0, 0]


def test_topology_metrics_use_frozen_upper_edges() -> None:
    owner = np.asarray([0, 0, 1, 1], dtype=np.int32)
    edge_left = np.asarray([0, 1, 2], dtype=np.int32)
    edge_right = np.asarray([1, 2, 3], dtype=np.int32)

    metrics = topology_metrics(
        owner,
        edge_left,
        edge_right,
        node_count=4,
        num_partitions=2,
    )

    assert metrics["upper_edge_count"] == 3
    assert np.isclose(metrics["upper_edge_cut_ratio"], 1 / 3)
    assert metrics["largest_component_fraction_min"] == 1.0
    assert metrics["upper_isolated_fraction"] == 0.0


def test_query_metrics_use_final_upper_memberships_not_l1_owner() -> None:
    owner = np.asarray([0, 0, 1, 1], dtype=np.int32)
    query_local = np.asarray([[0, 2], [1, 3]], dtype=np.int32)
    upper_membership = np.asarray(
        [[True, False], [True, True], [False, True], [False, True]],
        dtype=bool,
    )
    full_membership = upper_membership.copy()

    metrics = query_metrics(
        owner,
        query_local,
        upper_membership,
        full_membership,
        ground_truth=None,
        dynamic_ef_base=10,
        dynamic_ef_factor=2,
    )

    assert metrics["query_owner_shards_mean"] == 2.0
    assert metrics["routed_shards_mean"] == 2.0
    assert metrics["route_entry_points_mean"] == 2.5
    assert metrics["route_ef_sum_mean"] == 25.0
