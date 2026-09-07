"""Capacity-bounded placement of a frozen C_CNBR BMR_10 copy budget.

The L1 owner and online routing topology are immutable inputs.  This treatment
changes only the L0 shard location of already-requested BMR_10 copies.  Every
copy remains on a shard represented in the point's frozen upper attachments,
and every point retains exactly the same copy count as ordinary BMR_10.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from experiments.multi_assignment import budgeted_policy


CANDIDATE_ID = "BMR_10_CAP35"
POLICY_VERSION = 1
MIN_LOAD_RATIO = 0.35
MAX_LOAD_RATIO = 3.20
MAX_VOTE_LOSS = 2
MAX_PASSES = 8


@dataclass(frozen=True)
class CapacityBoundedAssignment:
    membership: np.ndarray
    primary: np.ndarray
    base_membership_semantic_sha256: str
    membership_semantic_sha256: str
    diagnostics: dict[str, Any]


def membership_semantic_sha256(membership: np.ndarray) -> str:
    packed = np.packbits(np.asarray(membership, dtype=bool), axis=1, bitorder="little")
    return hashlib.sha256(np.ascontiguousarray(packed).tobytes()).hexdigest()


def build_capacity_bounded_assignment(
    *,
    owner: np.ndarray,
    attachment_local: np.ndarray,
    proxy_mass: np.ndarray,
    num_partitions: int,
) -> CapacityBoundedAssignment:
    """Preserve BMR_10 copy counts while enforcing a physical-copy floor."""

    owner = np.asarray(owner, dtype=np.int32)
    attachment_local = np.asarray(attachment_local, dtype=np.int32)
    proxy_mass = np.asarray(proxy_mass, dtype=np.uint64)
    base = budgeted_policy.build_budgeted_assignment(
        owner=owner,
        attachment_local=attachment_local,
        proxy_mass=proxy_mass,
        num_partitions=int(num_partitions),
    )
    required_copies = base.membership.sum(axis=1, dtype=np.int16)
    initial = [
        np.flatnonzero(row).astype(int).tolist() for row in base.membership
    ]

    from tools.qdrant_two_level_routing_experiment import (
        assign_points_by_l1_vote_capacity_constrained,
    )

    primary, assignments, diagnostics = assign_points_by_l1_vote_capacity_constrained(
        attachment_local,
        owner.tolist(),
        int(num_partitions),
        True,
        min_load_ratio=MIN_LOAD_RATIO,
        max_load_ratio=MAX_LOAD_RATIO,
        max_passes=MAX_PASSES,
        max_vote_loss=MAX_VOTE_LOSS,
        initial_point_to_shards=initial,
        required_copies_override=required_copies,
    )
    membership = np.zeros_like(base.membership, dtype=bool)
    for point_id, shards in enumerate(assignments):
        membership[point_id, shards] = True

    observed_copies = membership.sum(axis=1, dtype=np.int16)
    if not np.array_equal(observed_copies, required_copies):
        raise RuntimeError("capacity-bounded BMR_10 changed a point's copy count")
    if int(membership.sum()) != int(base.membership.sum()):
        raise RuntimeError("capacity-bounded BMR_10 changed total physical copies")
    if diagnostics.get("bounds_satisfied") is not True:
        raise RuntimeError(
            "capacity-bounded BMR_10 evidence graph is infeasible: "
            f"under={diagnostics.get('under_lower_shards')}, "
            f"over={diagnostics.get('over_upper_shards')}"
        )
    if diagnostics.get("copy_count_preserved") is not True:
        raise RuntimeError("capacity-bounded BMR_10 copy proof failed")
    if diagnostics.get("all_assignments_have_navigation_evidence") is not True:
        raise RuntimeError("capacity-bounded BMR_10 used a non-evidence shard")
    if diagnostics.get("non_evidence_assignment_count") != 0:
        raise RuntimeError("capacity-bounded BMR_10 emitted non-evidence copies")
    if diagnostics.get("required_copy_count_source") != "explicit_override":
        raise RuntimeError("capacity-bounded BMR_10 did not freeze copy counts")
    if int(diagnostics.get("maximum_assignment_target_vote_loss", -1)) > MAX_VOTE_LOSS:
        raise RuntimeError("capacity-bounded BMR_10 exceeded its vote-loss bound")

    return CapacityBoundedAssignment(
        membership=membership,
        primary=np.asarray(primary, dtype=np.int32),
        base_membership_semantic_sha256=base.membership_semantic_sha256,
        membership_semantic_sha256=membership_semantic_sha256(membership),
        diagnostics=diagnostics,
    )
