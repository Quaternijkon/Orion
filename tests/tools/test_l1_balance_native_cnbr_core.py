from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = REPO_ROOT / "experiments/l1_balance/native_cnbr_core.py"
SPEC = importlib.util.spec_from_file_location("native_cnbr_core_test", CORE_PATH)
assert SPEC is not None and SPEC.loader is not None
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def _reference_native(vectors: np.ndarray, partitions: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(vectors, dtype=np.float32)
    rng = np.random.default_rng(1)
    centroids = np.zeros((partitions, matrix.shape[1]), dtype=np.float32)
    centroids[0] = matrix[int(rng.integers(0, len(matrix)))]
    minimum = np.full(len(matrix), np.finfo(np.float32).max, dtype=np.float32)
    for partition in range(1, partitions):
        distance = np.sum((matrix - centroids[partition - 1]) ** 2, axis=1)
        minimum = np.minimum(minimum, distance)
        total = float(np.sum(minimum))
        if total <= 0.0:
            position = 0
        else:
            threshold = float(rng.random()) * total
            position = int(np.searchsorted(np.cumsum(minimum), threshold, side="left"))
            position = min(position, len(matrix) - 1)
        centroids[partition] = matrix[position]
    for _ in range(10):
        distances = core._squared_l2_distances(matrix, centroids)
        assignments = np.argmin(distances, axis=1)
        updated = centroids.copy()
        for partition in range(partitions):
            mask = assignments == partition
            if np.any(mask):
                updated[partition] = matrix[mask].mean(axis=0)
        centroids = updated
    owner = np.argmin(core._squared_l2_distances(matrix, centroids), axis=1)
    return centroids, owner


def _cnbr_fixture() -> tuple[list[list[int]], list[list[int]], list[int]]:
    node_count = 40
    owner = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    rows: list[list[int]] = []
    for node in range(node_count):
        if node == 0:
            # Partition 1 is visible only through self-navigation for node 0.
            row = [0, *range(1, 9), 10]
        elif node < 10:
            row = [node] + [value for value in range(10) if value != node]
        else:
            row = [node, *range(9)]
        assert len(row) == 10 and len(set(row)) == 10 and node in row
        rows.append(row)

    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    # Node 0 is the only isolated upper node.  Every other N_native partition
    # is internally connected, so the absolute topology gates remain valid.
    groups = [
        list(range(1, 10)),
        list(range(10, 20)),
        list(range(20, 30)),
        list(range(30, 40)),
    ]
    for group in groups:
        for left, right in zip(group, group[1:] + group[:1], strict=True):
            adjacency[left].append(right)
            adjacency[right].append(left)
    return adjacency, rows, owner


def test_public_core_api_is_upper_only() -> None:
    public = {
        "estimate": core.estimate_upper_self_navigation_hit_frequency_v1,
        "native": core.build_n_native_owner,
        "cnbr": core.build_c_cnbr_owner,
        "family": core.build_frozen_cnbr_family_owner,
    }
    forbidden = {
        "attachment",
        "point_to_l1s",
        "multi_assignment",
        "query",
        "ground_truth",
        "observed_load",
        "fission",
    }
    for function in public.values():
        parameters = set(inspect.signature(function).parameters)
        assert not any(
            any(token in parameter for token in forbidden) for parameter in parameters
        )
    assert "estimate_raw_regularized_v2" not in vars(core)


def test_proxy_mass_replays_exact_rows_and_rejects_mixed_binding() -> None:
    adjacency, rows, owner = _cnbr_fixture()
    mass = core.estimate_upper_self_navigation_hit_frequency_v1(rows, len(rows))
    assert mass.total_mass == 400
    assert sum(mass.values) == 400
    assert mass.values[0] == 40
    assert mass.source == "frozen_production_upper_self_navigation_top10"
    assert mass.transform == "raw_occurrence_count"
    assert mass.estimator_version == 1

    with pytest.raises(ValueError, match="exact replay"):
        core.build_c_cnbr_owner(
            adjacency,
            rows,
            replace(mass, values=(mass.values[0] + 1, *mass.values[1:])),
            owner,
            4,
        )

    other_rows = [list(row) for row in rows]
    other_rows[0][-1] = 11
    other_mass = core.estimate_upper_self_navigation_hit_frequency_v1(
        other_rows, len(other_rows)
    )
    with pytest.raises(ValueError, match="exact replay"):
        core.build_c_cnbr_owner(adjacency, rows, other_mass, owner, 4)


def test_research_external_mass_is_explicit_and_keeps_production_binding_strict() -> None:
    adjacency, rows, owner = _cnbr_fixture()
    production_mass = core.estimate_upper_self_navigation_hit_frequency_v1(
        rows, len(rows)
    )
    research = core.build_cnbr_owner_with_external_mass_for_research(
        adjacency,
        rows,
        np.asarray(production_mass.values, dtype=np.uint64),
        owner,
        4,
    )
    production = core.build_c_cnbr_owner(
        adjacency, rows, production_mass, owner, 4
    )
    assert research == production

    with pytest.raises(ValueError, match="positive total mass"):
        core.build_cnbr_owner_with_external_mass_for_research(
            adjacency,
            rows,
            np.zeros(len(rows), dtype=np.uint64),
            owner,
            4,
        )
    with pytest.raises(TypeError, match="contain integers"):
        core.build_cnbr_owner_with_external_mass_for_research(
            adjacency,
            rows,
            np.ones(len(rows), dtype=np.float32),
            owner,
            4,
        )


def test_n_native_matches_seed1_ten_update_reference_and_is_deterministic() -> None:
    vectors = np.asarray(
        [[0.0, 1.0]] * 4 + [[10.0, 1.0]] * 4 + [[20.0, 1.0]] * 4,
        dtype=np.float32,
    )
    expected_centroids, expected_owner = _reference_native(vectors, 3)
    first = core.build_n_native_owner(vectors, 3)
    second = core.build_n_native_owner(vectors, 3)
    np.testing.assert_array_equal(first.centroids, expected_centroids)
    assert first.owner == tuple(int(value) for value in expected_owner.tolist())
    assert first.owner == second.owner
    np.testing.assert_array_equal(first.centroids, second.centroids)
    assert sorted(first.partition_sizes) == [4, 4, 4]


def test_n_native_rejects_final_empty_partition() -> None:
    with pytest.raises(ValueError, match="empty partitions"):
        core.build_n_native_owner(np.zeros((10, 2), dtype=np.float32), 2)


def test_cnbr_accepts_navigation_only_target_and_records_integer_proofs() -> None:
    adjacency, rows, owner = _cnbr_fixture()
    mass = core.estimate_upper_self_navigation_hit_frequency_v1(rows, len(rows))
    result = core.build_c_cnbr_owner(adjacency, rows, mass, owner, 4)

    assert result.trigger_ratio_numerator == 9
    assert result.trigger_ratio_denominator == 4
    assert result.trigger_mass_numerator == 9 * 400
    assert result.trigger_mass_denominator == 4 * 4
    assert result.moved_nodes == (0,)
    assert result.owner[0] == 1
    assert all(result.owner[node] == owner[node] for node in range(1, len(owner)))
    assert len(set(result.moved_nodes)) == len(result.moved_nodes)
    assert all(record["pass"] for record in result.topology_gates.values())

    first_round = result.rounds[0]
    proposal = first_round["proposals"][0]
    accepted = first_round["accepted_moves"][0]
    assert proposal["node"] == 0
    assert proposal["target"] == 1
    assert not any(owner[neighbor] == 1 for neighbor in adjacency[0])
    assert all(
        evidence["pass"] for evidence in accepted["commit_load_predicates"].values()
    )
    assert set(accepted["commit_load_predicates"]) == {
        "source_above_trigger",
        "target_below_trigger",
        "source_target_gap_gt_node_mass",
        "post_move_target_at_or_below_trigger",
    }
    cut = accepted["cumulative_edge_cut_predicate"]
    assert cut == {"lhs": 0, "operator": "<=", "rhs": 0, "pass": True}
    assert first_round["pre_owner_sha256"] != first_round[
        "post_owner_sha256_before_round_gate"
    ]


def test_cnbr_sequential_commit_uses_snapshot_proposals_and_integer_cut_cap() -> None:
    adjacency, rows, owner = _cnbr_fixture()
    mass = core.estimate_upper_self_navigation_hit_frequency_v1(rows, len(rows))
    result = core.build_c_cnbr_owner(adjacency, rows, mass, owner, 4)
    second_round = result.rounds[1]
    assert second_round["proposal_count"] == 9
    assert second_round["accepted_move_count"] == 0
    assert all(
        decision["skip_reason"] == "cumulative_edge_cut_cap"
        for decision in second_round["commit_decisions"]
    )
    for decision in second_round["commit_decisions"]:
        proof = decision["cumulative_edge_cut_predicate"]
        assert proof["lhs"] > proof["rhs"]
        assert proof["pass"] is False
    assert second_round["pre_owner_sha256"] == second_round[
        "post_owner_sha256_before_round_gate"
    ]


def test_cnbr_rolls_back_entire_round_on_any_graph_gate_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    adjacency, rows, owner = _cnbr_fixture()
    mass = core.estimate_upper_self_navigation_hit_frequency_v1(rows, len(rows))
    original = core.graph_gate_results
    calls = 0

    def fail_first_changed_round(
        observed: dict[str, float | int],
        reference: dict[str, float | int],
    ) -> dict[str, dict[str, object]]:
        nonlocal calls
        calls += 1
        gates = original(observed, reference)
        if calls == 2:
            gates["upper_edge_cut_ratio"] = {
                **gates["upper_edge_cut_ratio"],
                "pass": False,
            }
        return gates

    monkeypatch.setattr(core, "graph_gate_results", fail_first_changed_round)
    result = core.build_c_cnbr_owner(adjacency, rows, mass, owner, 4)
    assert result.owner == tuple(owner)
    assert result.moved_nodes == ()
    assert len(result.rounds) == 1
    record = result.rounds[0]
    assert record["accepted_move_count"] == 1
    assert record["committed_move_count"] == 0
    assert record["rolled_back_move_count"] == 1
    assert record["rolled_back"] is True
    assert record["post_owner_sha256_after_rollback"] == record["pre_owner_sha256"]


def test_frozen_trigger_family_has_no_open_numeric_tuning() -> None:
    assert core.CNBR_FROZEN_TRIGGER_FAMILY == {
        "9/4": (9, 4),
        "2/1": (2, 1),
        "7/4": (7, 4),
        "8/5": (8, 5),
        "3/2": (3, 2),
        "7/5": (7, 5),
    }
    adjacency, rows, owner = _cnbr_fixture()
    mass = core.estimate_upper_self_navigation_hit_frequency_v1(rows, len(rows))
    with pytest.raises(ValueError, match="frozen family"):
        core.build_frozen_cnbr_family_owner(
            adjacency,
            rows,
            mass,
            owner,
            4,
            trigger_id="1.23",
        )


def test_prepared_navigation_matches_legacy_mass_and_is_immutable() -> None:
    _adjacency, rows, _owner = _cnbr_fixture()
    source = np.asarray(rows, dtype=np.int32)
    expected_source = source.copy()
    legacy = core.estimate_upper_self_navigation_hit_frequency_v1(rows, len(rows))
    prepared = core.prepare_upper_navigation(source, len(rows))

    assert prepared.mass == legacy
    assert prepared.mass.semantic_sha256 == (
        "e3d1505645d16a1842340ead33aacd953f06256b25aa20c2270a025abe972c68"
    )
    np.testing.assert_array_equal(prepared.rows, expected_source)
    np.testing.assert_array_equal(
        prepared.self_ranks,
        np.asarray([row.index(node) for node, row in enumerate(rows)], dtype=np.int8),
    )
    assert prepared.rows.flags.writeable is False
    assert prepared.self_ranks.flags.writeable is False
    with pytest.raises(ValueError):
        prepared.rows.setflags(write=True)
    with pytest.raises(ValueError):
        prepared.self_ranks.setflags(write=True)

    source[0, 0] = 1
    np.testing.assert_array_equal(prepared.rows, expected_source)


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (lambda rows: rows[:, :9], ValueError, "exactly 10"),
        (lambda rows: rows.astype(np.float32), TypeError, "non-integer"),
        (lambda rows: rows.astype(bool), TypeError, "non-integer"),
        (
            lambda rows: np.where(
                np.indices(rows.shape)[1] == 0,
                -1,
                rows,
            ).astype(np.int32),
            ValueError,
            "out-of-range",
        ),
        (
            lambda rows: np.where(
                np.indices(rows.shape)[1] == 0,
                len(rows),
                rows,
            ).astype(np.int32),
            ValueError,
            "out-of-range",
        ),
        (
            lambda rows: np.column_stack((rows[:, 1], rows[:, 1:])),
            ValueError,
            "repeats upper node",
        ),
        (
            lambda rows: np.column_stack(
                (np.full(len(rows), 39, dtype=np.int32), rows[:, 1:])
            ),
            ValueError,
            "own upper node",
        ),
    ],
)
def test_prepared_navigation_fails_closed_on_invalid_rows(
    mutate: object,
    error: type[Exception],
    message: str,
) -> None:
    _adjacency, rows, _owner = _cnbr_fixture()
    source = np.asarray(rows, dtype=np.int32)
    invalid = mutate(source.copy())  # type: ignore[operator]
    with pytest.raises(error, match=message):
        core.prepare_upper_navigation(invalid, len(rows))


def test_prepared_and_legacy_results_are_exact_for_every_frozen_trigger() -> None:
    adjacency, rows, owner = _cnbr_fixture()
    mass = core.estimate_upper_self_navigation_hit_frequency_v1(rows, len(rows))
    prepared = core.prepare_upper_navigation(np.asarray(rows, dtype=np.int32), len(rows))

    for trigger_id in core.CNBR_FROZEN_TRIGGER_FAMILY:
        legacy = core.build_frozen_cnbr_family_owner(
            adjacency,
            rows,
            mass,
            owner,
            4,
            trigger_id=trigger_id,
        )
        optimized = core.build_frozen_cnbr_family_owner_prepared(
            adjacency,
            prepared,
            owner,
            4,
            trigger_id=trigger_id,
        )
        assert optimized == legacy

    selected = core.build_c_cnbr_owner_prepared(adjacency, prepared, owner, 4)
    owner_sha = hashlib.sha256(
        np.asarray(selected.owner, dtype="<i4").tobytes(order="C")
    ).hexdigest()
    trace_sha = hashlib.sha256(
        json.dumps(
            selected.rounds,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert owner_sha == "0902cbc782204b8e4af3ab5649f849ae1f8c10ed6ee47af918cf430f5f83606f"
    assert trace_sha == "4d2c942a4b1537607cc780bb7c5eee725739197b729ced9dd8a566786fa7a82b"


def test_prepared_builder_rejects_unsealed_instances_and_preserves_trigger_priority() -> None:
    adjacency, _rows, owner = _cnbr_fixture()
    forged = core.PreparedUpperNavigation()
    with pytest.raises(ValueError, match="private validation seal"):
        core.build_c_cnbr_owner_prepared(adjacency, forged, owner, 4)
    with pytest.raises(ValueError, match="frozen family"):
        core.build_frozen_cnbr_family_owner_prepared(
            adjacency,
            forged,
            owner,
            4,
            trigger_id="not-frozen",
        )
