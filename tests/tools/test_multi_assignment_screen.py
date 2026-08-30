import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "multi_assignment"
    / "screen_fixed_owner.py"
)
SPEC = importlib.util.spec_from_file_location("screen_fixed_owner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

POLICIES = module.POLICIES
build_vote_evidence = module.build_vote_evidence
membership_for_policy = module.membership_for_policy


def apply(name: str, hit_owner: np.ndarray) -> np.ndarray:
    votes, maximum, first_rank, rank_top, id_top = build_vote_evidence(
        hit_owner, 9
    )
    policy = next(policy for policy in POLICIES if policy.name == name)
    return membership_for_policy(
        hit_owner=hit_owner,
        votes=votes,
        maximum=maximum,
        first_rank=first_rank,
        rank_top=rank_top,
        id_top=id_top,
        policy=policy,
    )


def selected(row: np.ndarray) -> list[int]:
    return np.flatnonzero(row).tolist()


def test_current_rule_and_rank_caps_are_exact():
    hits = np.asarray(
        [
            [4, 2, 3, 4, 2, 3, 4, 2, 3, 8],
            [7, 1, 2, 3, 4, 1, 5, 6, 7, 8],
            [5, 4, 3, 2, 1, 0, 8, 7, 6, 5],
        ],
        dtype=np.int32,
    )
    assert selected(apply("current_all_max", hits)[0]) == [2, 3, 4]
    assert selected(apply("cap2_rank", hits)[0]) == [2, 4]
    assert selected(apply("cap3_rank", hits)[0]) == [2, 3, 4]
    assert selected(apply("single_rank", hits)[0]) == [4]
    assert selected(apply("cap2_shard_id", hits)[0]) == [2, 3]
    assert selected(apply("current_all_max", hits)[1]) == [1, 7]
    assert selected(apply("current_all_max", hits)[2]) == [5]


def test_prefix_gate_uses_second_shards_first_navigation_rank():
    hits = np.asarray(
        [[7, 7, 2, 3, 4, 1, 1, 5, 6, 8]], dtype=np.int32
    )
    assert selected(apply("cap2_rank_prefix4", hits)[0]) == [7]
    assert selected(apply("cap2_rank_prefix6", hits)[0]) == [1, 7]
    assert selected(apply("cap2_rank_prefix8", hits)[0]) == [1, 7]


def test_every_policy_is_deterministic_and_keeps_one_supported_shard():
    hits = np.asarray(
        [
            [4, 2, 3, 4, 2, 3, 4, 2, 3, 8],
            [7, 1, 2, 3, 4, 1, 5, 6, 7, 8],
            [5, 4, 3, 2, 1, 0, 8, 7, 6, 5],
        ],
        dtype=np.int32,
    )
    for policy in POLICIES:
        first = apply(policy.name, hits)
        second = apply(policy.name, hits)
        assert np.array_equal(first, second)
        assert np.all(first.sum(axis=1) >= 1)
        for row_hits, row_membership in zip(hits, first, strict=True):
            assert set(np.flatnonzero(row_membership)).issubset(set(row_hits))
