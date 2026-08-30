"""Frozen BMR_10 multi-assignment policy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


CANDIDATE_ID = "BMR_10"
POLICY_VERSION = 1
EXTRA_COPY_BUDGET_NUMERATOR = 1
EXTRA_COPY_BUDGET_DENOMINATOR = 10
SCORE = "secondary_proxy_mass_max"


@dataclass(frozen=True)
class BudgetedAssignment:
    membership: np.ndarray
    primary: np.ndarray
    secondary: np.ndarray
    secondary_score: np.ndarray
    eligible_secondary_count: int
    kept_secondary_count: int
    membership_semantic_sha256: str


def _ranked_top_two(
    votes: np.ndarray, maximum: np.ndarray, first_rank: np.ndarray
) -> np.ndarray:
    rows, partitions = votes.shape
    row_ids = np.arange(rows, dtype=np.int64)
    eligible = (votes == maximum[:, None]) & (maximum[:, None] >= 2)
    shard_ids = np.arange(partitions, dtype=np.int16)[None, :]
    infinity = np.iinfo(np.int16).max
    score = np.where(
        eligible,
        first_rank.astype(np.int16) * partitions + shard_ids,
        infinity,
    )
    result = np.full((rows, 2), -1, dtype=np.int16)
    for column in range(2):
        chosen = np.argmin(score, axis=1)
        valid = score[row_ids, chosen] != infinity
        result[valid, column] = chosen[valid].astype(np.int16)
        score[row_ids[valid], chosen[valid]] = infinity
    return result


def build_budgeted_assignment(
    *,
    owner: np.ndarray,
    attachment_local: np.ndarray,
    proxy_mass: np.ndarray,
    num_partitions: int,
) -> BudgetedAssignment:
    owner = np.asarray(owner, dtype=np.int32)
    attachment_local = np.asarray(attachment_local, dtype=np.int32)
    proxy_mass = np.asarray(proxy_mass, dtype=np.uint64)
    if attachment_local.ndim != 2 or attachment_local.shape[1] <= 0:
        raise ValueError("attachment_local must be a non-empty matrix")
    if len(owner) != len(proxy_mass):
        raise ValueError("owner and proxy_mass length differ")
    if np.any(attachment_local < 0) or np.any(attachment_local >= len(owner)):
        raise ValueError("attachment_local contains an invalid upper index")
    if np.any(owner < 0) or np.any(owner >= num_partitions):
        raise ValueError("owner contains an invalid shard")

    hit_owner = owner[attachment_local]
    rows, width = hit_owner.shape
    row_ids = np.arange(rows, dtype=np.int64)
    votes = np.zeros((rows, num_partitions), dtype=np.uint8)
    first_rank = np.full((rows, num_partitions), width, dtype=np.uint8)
    for rank in range(width):
        shard = hit_owner[:, rank]
        np.add.at(votes, (row_ids, shard), 1)
        first_rank[row_ids, shard] = np.minimum(first_rank[row_ids, shard], rank)
    maximum = votes.max(axis=1)
    ranked = _ranked_top_two(votes, maximum, first_rank)

    fallback = maximum < 2
    primary = ranked[:, 0].astype(np.int32, copy=True)
    primary[fallback] = hit_owner[fallback, 0]
    secondary = ranked[:, 1].astype(np.int32, copy=False)
    if np.any(primary < 0):
        raise AssertionError("BMR_10 failed to choose a primary shard")

    safe_secondary = np.maximum(secondary, 0)
    secondary_mask = hit_owner == safe_secondary[:, None]
    hit_mass = proxy_mass[attachment_local]
    secondary_score = np.max(
        np.where(secondary_mask, hit_mass, np.uint64(0)), axis=1
    )
    secondary_score = np.asarray(secondary_score, dtype=np.uint64)

    membership = np.zeros((rows, num_partitions), dtype=bool)
    membership[row_ids, primary] = True
    eligible = np.flatnonzero(secondary >= 0)
    budget = (rows * EXTRA_COPY_BUDGET_NUMERATOR) // EXTRA_COPY_BUDGET_DENOMINATOR
    keep_count = min(int(budget), len(eligible))
    if keep_count:
        order = np.lexsort((eligible, -secondary_score[eligible].astype(np.int64)))
        kept = eligible[order[:keep_count]]
        membership[kept, secondary[kept]] = True

    packed = np.packbits(membership, axis=1, bitorder="little")
    semantic_sha256 = hashlib.sha256(
        np.ascontiguousarray(packed).tobytes()
    ).hexdigest()
    return BudgetedAssignment(
        membership=membership,
        primary=primary,
        secondary=secondary,
        secondary_score=secondary_score,
        eligible_secondary_count=len(eligible),
        kept_secondary_count=keep_count,
        membership_semantic_sha256=semantic_sha256,
    )
