import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "multi_assignment"
    / "budgeted_policy.py"
)
SPEC = importlib.util.spec_from_file_location("budgeted_policy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_budget_is_global_score_ranked_and_point_id_tied():
    owner = np.asarray([0, 1, 2, 3], dtype=np.int32)
    proxy = np.asarray([2, 20, 10, 1], dtype=np.uint64)
    # Twenty points make the 10% budget exactly two secondary copies.
    rows = []
    for index in range(20):
        if index == 0:
            rows.append([0, 1, 0, 1])  # secondary shard 1, score 20
        elif index == 1:
            rows.append([0, 2, 0, 2])  # secondary shard 2, score 10
        elif index == 2:
            rows.append([0, 3, 0, 3])  # secondary shard 3, score 1
        else:
            rows.append([0, 1, 2, 3])  # max vote < 2, no replica candidate
    result = module.build_budgeted_assignment(
        owner=owner,
        attachment_local=np.asarray(rows, dtype=np.int32),
        proxy_mass=proxy,
        num_partitions=4,
    )
    assert result.eligible_secondary_count == 3
    assert result.kept_secondary_count == 2
    assert np.flatnonzero(result.membership[0]).tolist() == [0, 1]
    assert np.flatnonzero(result.membership[1]).tolist() == [0, 2]
    assert np.flatnonzero(result.membership[2]).tolist() == [0]
    assert np.all(result.membership.sum(axis=1) <= 2)


def test_policy_is_deterministic_and_falls_back_to_first_attachment_owner():
    owner = np.asarray([3, 1, 2, 0], dtype=np.int32)
    proxy = np.asarray([5, 5, 5, 5], dtype=np.uint64)
    attachments = np.tile(np.asarray([[0, 1, 2, 3]], dtype=np.int32), (20, 1))
    first = module.build_budgeted_assignment(
        owner=owner,
        attachment_local=attachments,
        proxy_mass=proxy,
        num_partitions=4,
    )
    second = module.build_budgeted_assignment(
        owner=owner,
        attachment_local=attachments,
        proxy_mass=proxy,
        num_partitions=4,
    )
    assert first.membership_semantic_sha256 == second.membership_semantic_sha256
    assert np.array_equal(first.membership, second.membership)
    assert np.all(np.argmax(first.membership, axis=1) == 3)
    assert first.kept_secondary_count == 0
