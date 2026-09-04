"""Upper-only N_native and CNBR owner construction.

The public API accepts only the immutable upper graph, ordered upper vectors,
and ordered upper self-navigation rows.  It deliberately has no downstream
point-placement, observed-load, query, or replication inputs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from numbers import Integral
import struct
from typing import Any

import numpy as np


UPPER_NAVIGATION_TOP_K = 10
PROXY_MASS_CONTRACT_ID = "cnbr-upper-self-navigation-raw-occurrence-v1"
MASS_MODE = PROXY_MASS_CONTRACT_ID
MASS_SOURCE = "frozen_production_upper_self_navigation_top10"
MASS_TRANSFORM = "raw_occurrence_count"
MASS_ESTIMATOR_VERSION = 1

NATIVE_KMEANS_SEED = 1
NATIVE_KMEANS_ITERATIONS = 10

CNBR_TRIGGER_NUMERATOR = 9
CNBR_TRIGGER_DENOMINATOR = 4
CNBR_SELECTED_TRIGGER_ID = "9/4"
CNBR_FROZEN_TRIGGER_FAMILY = {
    "9/4": (9, 4),
    "2/1": (2, 1),
    "7/4": (7, 4),
    "8/5": (8, 5),
    "3/2": (3, 2),
    "7/5": (7, 5),
}
CNBR_MAX_ROUNDS = 8
CNBR_NODE_MOVE_LIMIT = 1
CNBR_REQUIRES_CROSS_OWNER_DIRECT_GRAPH_NEIGHBOR = False
CNBR_SELF_NAVIGATION_ONLY_TARGET_ALLOWED = True

EDGE_CUT_MAX_RATIO = 1.03
RETAINED_DEGREE_MIN_RATIO = 0.95
RETAINED_DEGREE_P10_MAX_DROP = 0.025
ISOLATED_FRACTION_MAX_DELTA = 0.01
ISOLATED_FRACTION_ABSOLUTE_MAX = 0.03
LARGEST_COMPONENT_MEAN_MAX_DROP = 0.02
LARGEST_COMPONENT_MIN_FLOOR = 0.25


@dataclass(frozen=True)
class UpperNavigationMass:
    values: tuple[int, ...]
    query_count: int
    top_k: int
    total_mass: int
    source: str
    transform: str
    estimator_version: int
    semantic_sha256: str


_PREPARED_UPPER_NAVIGATION_SEAL = object()


@dataclass(frozen=True, eq=False, init=False)
class PreparedUpperNavigation:
    """One-time validated upper navigation with immutable canonical arrays."""

    rows: np.ndarray
    self_ranks: np.ndarray
    mass: UpperNavigationMass
    node_count: int
    top_k: int
    _seal: object


@dataclass(frozen=True)
class NativeKMeansResult:
    owner: tuple[int, ...]
    centroids: np.ndarray
    partition_sizes: tuple[int, ...]


@dataclass(frozen=True)
class CnbrResult:
    owner: tuple[int, ...]
    partition_masses: tuple[int, ...]
    moved_nodes: tuple[int, ...]
    rounds: tuple[dict[str, Any], ...]
    topology_metrics: dict[str, float | int]
    topology_gates: dict[str, dict[str, Any]]
    trigger_ratio_numerator: int
    trigger_ratio_denominator: int
    trigger_mass_numerator: int
    trigger_mass_denominator: int
    edges_examined: int
    navigation_hits_examined: int
    topology_evaluations: int


def estimate_upper_self_navigation_hit_frequency_v1(
    navigation_rows: Sequence[Sequence[int]],
    node_count: int,
) -> UpperNavigationMass:
    """Count occurrences in exactly ``U x top10`` frozen self-navigation rows."""

    rows = _validated_navigation_rows(navigation_rows, node_count)
    return _mass_from_validated_rows(rows, node_count)


def prepare_upper_navigation(
    navigation_rows: np.ndarray,
    node_count: int,
) -> PreparedUpperNavigation:
    """Vectorize the frozen row validation and raw occurrence replay once."""

    if isinstance(node_count, bool) or not isinstance(node_count, Integral):
        raise TypeError("node_count must be an integer")
    count = int(node_count)
    if count < UPPER_NAVIGATION_TOP_K:
        raise ValueError(
            f"node_count must be at least fixed top-k {UPPER_NAVIGATION_TOP_K}"
        )
    if not isinstance(navigation_rows, np.ndarray):
        raise TypeError("navigation_rows must be a numpy array")
    if navigation_rows.ndim != 2 or navigation_rows.shape[0] != count:
        raise ValueError("navigation row count must equal upper node count")
    if navigation_rows.shape[1] != UPPER_NAVIGATION_TOP_K:
        raise ValueError(
            "each navigation row must contain exactly "
            f"{UPPER_NAVIGATION_TOP_K} hits"
        )
    if navigation_rows.dtype.kind not in "iu" or navigation_rows.dtype.kind == "b":
        raise TypeError("navigation rows contain a non-integer node")
    if navigation_rows.dtype.kind == "i" and bool(np.any(navigation_rows < 0)):
        row, column = np.argwhere(navigation_rows < 0)[0]
        raise ValueError(
            f"navigation_rows[{int(row)}] contains out-of-range node "
            f"{int(navigation_rows[int(row), int(column)])}"
        )
    if bool(np.any(navigation_rows >= count)):
        row, column = np.argwhere(navigation_rows >= count)[0]
        raise ValueError(
            f"navigation_rows[{int(row)}] contains out-of-range node "
            f"{int(navigation_rows[int(row), int(column)])}"
        )

    normalized = np.ascontiguousarray(navigation_rows, dtype="<i4")
    sorted_rows = np.sort(normalized, axis=1)
    repeated = sorted_rows[:, 1:] == sorted_rows[:, :-1]
    if bool(np.any(repeated)):
        row, column = np.argwhere(repeated)[0]
        repeated_node = int(sorted_rows[int(row), int(column)])
        raise ValueError(
            f"navigation_rows[{int(row)}] repeats upper node {repeated_node}"
        )

    self_matches = normalized == np.arange(count, dtype=np.int32)[:, None]
    self_counts = np.count_nonzero(self_matches, axis=1)
    if bool(np.any(self_counts != 1)):
        row = int(np.flatnonzero(self_counts != 1)[0])
        raise ValueError(
            f"navigation_rows[{row}] does not contain its own upper node exactly once"
        )
    self_ranks_array = np.argmax(self_matches, axis=1).astype(np.int8, copy=False)
    mass_values = np.bincount(normalized.reshape(-1), minlength=count).astype(
        np.int64,
        copy=False,
    )
    mass = _mass_from_values(mass_values, count)

    rows_bytes = normalized.tobytes(order="C")
    rows = np.frombuffer(rows_bytes, dtype="<i4").reshape(
        count,
        UPPER_NAVIGATION_TOP_K,
    )
    self_ranks_bytes = self_ranks_array.tobytes(order="C")
    self_ranks = np.frombuffer(self_ranks_bytes, dtype=np.int8)
    prepared = object.__new__(PreparedUpperNavigation)
    object.__setattr__(prepared, "rows", rows)
    object.__setattr__(prepared, "self_ranks", self_ranks)
    object.__setattr__(prepared, "mass", mass)
    object.__setattr__(prepared, "node_count", count)
    object.__setattr__(prepared, "top_k", UPPER_NAVIGATION_TOP_K)
    object.__setattr__(prepared, "_seal", _PREPARED_UPPER_NAVIGATION_SEAL)
    return prepared


def _mass_from_validated_rows(
    rows: tuple[tuple[int, ...], ...],
    node_count: int,
) -> UpperNavigationMass:
    """Replay raw occurrences after the structural row contract is proven."""

    mass = [0] * node_count
    for row in rows:
        for node in row:
            mass[node] += 1
    values = tuple(mass)
    total_mass = sum(values)
    digest = hashlib.sha256()
    digest.update(MASS_SOURCE.encode("ascii"))
    digest.update(b"\0")
    digest.update(MASS_TRANSFORM.encode("ascii"))
    digest.update(struct.pack("<QQ", node_count, UPPER_NAVIGATION_TOP_K))
    for value in values:
        digest.update(struct.pack("<Q", value))
    return UpperNavigationMass(
        values=values,
        query_count=node_count,
        top_k=UPPER_NAVIGATION_TOP_K,
        total_mass=total_mass,
        source=MASS_SOURCE,
        transform=MASS_TRANSFORM,
        estimator_version=MASS_ESTIMATOR_VERSION,
        semantic_sha256=digest.hexdigest(),
    )


def _mass_from_values(
    values_array: np.ndarray,
    node_count: int,
) -> UpperNavigationMass:
    if (
        not isinstance(values_array, np.ndarray)
        or values_array.ndim != 1
        or len(values_array) != node_count
        or values_array.dtype.kind not in "iu"
    ):
        raise ValueError("navigation mass must contain one integer per upper node")
    if values_array.dtype.kind == "i" and bool(np.any(values_array < 0)):
        raise ValueError("navigation mass contains a negative occurrence count")
    canonical = np.ascontiguousarray(values_array, dtype="<u8")
    values = tuple(int(value) for value in canonical.tolist())
    digest = hashlib.sha256()
    digest.update(MASS_SOURCE.encode("ascii"))
    digest.update(b"\0")
    digest.update(MASS_TRANSFORM.encode("ascii"))
    digest.update(struct.pack("<QQ", node_count, UPPER_NAVIGATION_TOP_K))
    digest.update(canonical.tobytes(order="C"))
    return UpperNavigationMass(
        values=values,
        query_count=node_count,
        top_k=UPPER_NAVIGATION_TOP_K,
        total_mass=int(np.sum(canonical, dtype=np.uint64)),
        source=MASS_SOURCE,
        transform=MASS_TRANSFORM,
        estimator_version=MASS_ESTIMATOR_VERSION,
        semantic_sha256=digest.hexdigest(),
    )


def build_n_native_owner(
    upper_vectors: Sequence[Sequence[float]] | np.ndarray,
    num_partitions: int,
) -> NativeKMeansResult:
    """Run the frozen native owner: seed-1 KMeans, ten updates, plain argmin."""

    partitions = _validated_partition_count(num_partitions)
    matrix = _validated_vectors(upper_vectors)
    node_count = len(matrix)
    if partitions > node_count:
        raise ValueError("num_partitions must not exceed upper node count")

    centroids = _native_kmeans_plus_plus(matrix, partitions)
    for _ in range(NATIVE_KMEANS_ITERATIONS):
        assignments = np.argmin(_squared_l2_distances(matrix, centroids), axis=1)
        updated = centroids.copy()
        for partition in range(partitions):
            mask = assignments == partition
            if bool(np.any(mask)):
                updated[partition] = matrix[mask].mean(axis=0)
        centroids = updated

    owner_array = np.argmin(_squared_l2_distances(matrix, centroids), axis=1).astype(
        np.int32,
        copy=False,
    )
    sizes = np.bincount(owner_array, minlength=partitions)
    empty = np.flatnonzero(sizes == 0)
    if len(empty):
        raise ValueError(
            "N_native final argmin produced empty partitions: "
            + ",".join(str(int(value)) for value in empty.tolist())
        )
    return NativeKMeansResult(
        owner=tuple(int(value) for value in owner_array.tolist()),
        centroids=np.ascontiguousarray(centroids, dtype=np.float32),
        partition_sizes=tuple(int(value) for value in sizes.tolist()),
    )


def topology_metrics(
    adjacency: Sequence[Sequence[int]],
    owner: Sequence[int] | np.ndarray,
    num_partitions: int,
) -> dict[str, float | int]:
    """Compute the six fail-closed upper-graph topology metrics."""

    graph = normalize_undirected_adjacency(adjacency)
    return _topology_metrics_normalized(graph, owner, num_partitions)


def _topology_metrics_normalized(
    graph: tuple[tuple[int, ...], ...],
    owner: Sequence[int] | np.ndarray,
    num_partitions: int,
    *,
    edge_left: np.ndarray | None = None,
    edge_right: np.ndarray | None = None,
    degree: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute topology metrics for an already normalized undirected graph."""

    partitions = _validated_partition_count(num_partitions)
    owner_array = _validated_owner(owner, len(graph), partitions)
    if edge_left is None or edge_right is None:
        edge_left, edge_right = _edge_arrays(graph)
    node_count = len(graph)

    if len(edge_left):
        internal = owner_array[edge_left] == owner_array[edge_right]
        internal_degree = np.bincount(
            np.concatenate((edge_left[internal], edge_right[internal])),
            minlength=node_count,
        )
        if degree is None:
            degree = np.bincount(
                np.concatenate((edge_left, edge_right)),
                minlength=node_count,
            )
        edge_cut_ratio = float(1.0 - np.mean(internal))
    else:
        internal = np.empty(0, dtype=bool)
        internal_degree = np.zeros(node_count, dtype=np.int64)
        if degree is None:
            degree = np.zeros(node_count, dtype=np.int64)
        edge_cut_ratio = 0.0
    retained = np.divide(
        internal_degree,
        degree,
        out=np.ones(node_count, dtype=np.float64),
        where=degree > 0,
    )

    parent = np.arange(node_count, dtype=np.int32)
    component_size = np.ones(node_count, dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[node]) != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    for raw_left, raw_right in zip(edge_left[internal], edge_right[internal], strict=True):
        left = find(int(raw_left))
        right = find(int(raw_right))
        if left == right:
            continue
        if component_size[left] < component_size[right]:
            left, right = right, left
        parent[right] = left
        component_size[left] += component_size[right]

    partition_sizes = np.bincount(owner_array, minlength=partitions)
    largest = np.zeros(partitions, dtype=np.int32)
    for node in range(node_count):
        root = find(node)
        partition = int(owner_array[node])
        largest[partition] = max(largest[partition], component_size[root])
    largest_fraction = np.divide(
        largest,
        partition_sizes,
        out=np.zeros(partitions, dtype=np.float64),
        where=partition_sizes > 0,
    )

    return {
        "upper_edge_count": int(len(edge_left)),
        "upper_edge_cut_count": int(np.count_nonzero(~internal)),
        "upper_edge_cut_ratio": edge_cut_ratio,
        "retained_degree_mean": float(np.mean(retained)),
        "retained_degree_p10": float(np.percentile(retained, 10)),
        "upper_isolated_fraction": float(np.mean(internal_degree == 0)),
        "largest_component_fraction_mean": float(np.mean(largest_fraction)),
        "largest_component_fraction_min": float(np.min(largest_fraction)),
    }


def graph_gate_results(
    observed: dict[str, float | int],
    reference: dict[str, float | int],
) -> dict[str, dict[str, Any]]:
    """Evaluate all six fixed graph gates relative to N_native."""

    thresholds = {
        "upper_edge_cut_ratio": float(reference["upper_edge_cut_ratio"])
        * EDGE_CUT_MAX_RATIO,
        "retained_degree_mean": float(reference["retained_degree_mean"])
        * RETAINED_DEGREE_MIN_RATIO,
        "retained_degree_p10": float(reference["retained_degree_p10"])
        - RETAINED_DEGREE_P10_MAX_DROP,
        "upper_isolated_fraction": min(
            float(reference["upper_isolated_fraction"])
            + ISOLATED_FRACTION_MAX_DELTA,
            ISOLATED_FRACTION_ABSOLUTE_MAX,
        ),
        "largest_component_fraction_mean": float(
            reference["largest_component_fraction_mean"]
        )
        - LARGEST_COMPONENT_MEAN_MAX_DROP,
        "largest_component_fraction_min": LARGEST_COMPONENT_MIN_FLOOR,
    }
    operators = {
        "upper_edge_cut_ratio": "<=",
        "retained_degree_mean": ">=",
        "retained_degree_p10": ">=",
        "upper_isolated_fraction": "<=",
        "largest_component_fraction_mean": ">=",
        "largest_component_fraction_min": ">=",
    }
    gates: dict[str, dict[str, Any]] = {}
    for metric, threshold in thresholds.items():
        value = float(observed[metric])
        operator = operators[metric]
        passed = value <= threshold if operator == "<=" else value >= threshold
        gates[metric] = {
            "reference": float(reference[metric]),
            "observed": value,
            "operator": operator,
            "threshold": float(threshold),
            "pass": bool(passed),
        }
    return gates


def build_c_cnbr_owner(
    adjacency: Sequence[Sequence[int]],
    navigation_rows: Sequence[Sequence[int]],
    navigation_mass: UpperNavigationMass,
    n_owner: Sequence[int] | np.ndarray,
    num_partitions: int,
) -> CnbrResult:
    """Repair N_native by the frozen cap-9/4 local boundary rule."""

    return _build_cnbr_owner_with_trigger(
        adjacency,
        navigation_rows,
        navigation_mass,
        n_owner,
        num_partitions,
        trigger_ratio_numerator=CNBR_TRIGGER_NUMERATOR,
        trigger_ratio_denominator=CNBR_TRIGGER_DENOMINATOR,
    )


def build_c_cnbr_owner_prepared(
    adjacency: Sequence[Sequence[int]],
    prepared_navigation: PreparedUpperNavigation,
    n_owner: Sequence[int] | np.ndarray,
    num_partitions: int,
) -> CnbrResult:
    """Repair N_native using one strictly prepared navigation snapshot."""

    return _build_cnbr_owner_with_trigger(
        adjacency,
        prepared_navigation,
        None,
        n_owner,
        num_partitions,
        trigger_ratio_numerator=CNBR_TRIGGER_NUMERATOR,
        trigger_ratio_denominator=CNBR_TRIGGER_DENOMINATOR,
        prepared_navigation=True,
    )


def build_cnbr_owner_with_external_mass_for_research(
    adjacency: Sequence[Sequence[int]],
    navigation_rows: Sequence[Sequence[int]],
    mass_values: Sequence[int] | np.ndarray,
    n_owner: Sequence[int] | np.ndarray,
    num_partitions: int,
) -> CnbrResult:
    """Run frozen CNBR 9/4 with an explicitly non-production proxy mass.

    This narrow entry point exists for offline estimator experiments such as a
    deterministic sampled-L0 fill.  The production builders above continue to
    require the exact upper-self-navigation replay and cannot receive this
    override.
    """

    return _build_cnbr_owner_with_trigger(
        adjacency,
        navigation_rows,
        None,
        n_owner,
        num_partitions,
        trigger_ratio_numerator=CNBR_TRIGGER_NUMERATOR,
        trigger_ratio_denominator=CNBR_TRIGGER_DENOMINATOR,
        external_mass_values=mass_values,
    )


def build_frozen_cnbr_family_owner(
    adjacency: Sequence[Sequence[int]],
    navigation_rows: Sequence[Sequence[int]],
    navigation_mass: UpperNavigationMass,
    n_owner: Sequence[int] | np.ndarray,
    num_partitions: int,
    *,
    trigger_id: str,
) -> CnbrResult:
    """Run one member of the immutable preregistered rational trigger family."""

    if trigger_id not in CNBR_FROZEN_TRIGGER_FAMILY:
        raise ValueError(
            "trigger_id must be one of the frozen family: "
            + ", ".join(CNBR_FROZEN_TRIGGER_FAMILY)
        )
    numerator, denominator = CNBR_FROZEN_TRIGGER_FAMILY[trigger_id]
    return _build_cnbr_owner_with_trigger(
        adjacency,
        navigation_rows,
        navigation_mass,
        n_owner,
        num_partitions,
        trigger_ratio_numerator=numerator,
        trigger_ratio_denominator=denominator,
    )


def build_frozen_cnbr_family_owner_prepared(
    adjacency: Sequence[Sequence[int]],
    prepared_navigation: PreparedUpperNavigation,
    n_owner: Sequence[int] | np.ndarray,
    num_partitions: int,
    *,
    trigger_id: str,
) -> CnbrResult:
    """Run one immutable trigger using a reusable prepared navigation."""

    if trigger_id not in CNBR_FROZEN_TRIGGER_FAMILY:
        raise ValueError(
            "trigger_id must be one of the frozen family: "
            + ", ".join(CNBR_FROZEN_TRIGGER_FAMILY)
        )
    numerator, denominator = CNBR_FROZEN_TRIGGER_FAMILY[trigger_id]
    return _build_cnbr_owner_with_trigger(
        adjacency,
        prepared_navigation,
        None,
        n_owner,
        num_partitions,
        trigger_ratio_numerator=numerator,
        trigger_ratio_denominator=denominator,
        prepared_navigation=True,
    )


def _build_cnbr_owner_with_trigger(
    adjacency: Sequence[Sequence[int]],
    navigation_rows: Sequence[Sequence[int]] | PreparedUpperNavigation,
    navigation_mass: UpperNavigationMass | None,
    n_owner: Sequence[int] | np.ndarray,
    num_partitions: int,
    *,
    trigger_ratio_numerator: int,
    trigger_ratio_denominator: int,
    prepared_navigation: bool = False,
    external_mass_values: Sequence[int] | np.ndarray | None = None,
) -> CnbrResult:
    """Shared implementation for the closed rational trigger family."""

    graph = normalize_undirected_adjacency(adjacency)
    node_count = len(graph)
    edge_left, edge_right = _edge_arrays(graph)
    graph_degree = np.bincount(
        np.concatenate((edge_left, edge_right)),
        minlength=node_count,
    )
    partitions = _validated_partition_count(num_partitions)
    if external_mass_values is not None:
        if prepared_navigation:
            raise ValueError("external research mass cannot use prepared navigation")
        if navigation_mass is not None:
            raise ValueError("external research mass must not receive navigation_mass")
        rows = _validated_navigation_rows(navigation_rows, node_count)
        mass = _validated_external_mass(external_mass_values, node_count)
    elif prepared_navigation:
        prepared = _validated_prepared_navigation(navigation_rows, node_count)
        if navigation_mass is not None:
            raise ValueError("prepared navigation must not receive a separate mass")
        rows = prepared.rows
        mass = prepared.mass.values
    else:
        rows = _validated_navigation_rows(navigation_rows, node_count)
        if navigation_mass is None:
            raise TypeError("navigation_mass must be an UpperNavigationMass")
        mass = _validated_mass(navigation_mass, rows, node_count)
    reference_owner = _validated_owner(n_owner, node_count, partitions)
    reference_sizes = np.bincount(reference_owner, minlength=partitions)
    if bool(np.any(reference_sizes == 0)):
        raise ValueError("N_native reference owner contains an empty partition")
    owner = reference_owner.copy()
    loads = _partition_masses(owner, mass, partitions)
    reference_metrics = _topology_metrics_normalized(
        graph,
        reference_owner,
        partitions,
        edge_left=edge_left,
        edge_right=edge_right,
        degree=graph_degree,
    )
    edge_count = int(reference_metrics["upper_edge_count"])
    reference_cut_count = int(reference_metrics["upper_edge_cut_count"])
    current_cut_count = int(reference_metrics["upper_edge_cut_count"])
    total_mass = int(sum(mass))
    trigger_numerator = int(trigger_ratio_numerator) * total_mass
    trigger_denominator = int(trigger_ratio_denominator) * partitions
    moved: set[int] = set()
    round_records: list[dict[str, Any]] = []
    current_metrics = reference_metrics
    current_gates = graph_gate_results(current_metrics, reference_metrics)
    edges_examined = 0
    navigation_hits_examined = 0
    topology_evaluations = 1

    for round_index in range(CNBR_MAX_ROUNDS):
        snapshot_owner = owner.copy()
        snapshot_loads = loads.copy()
        snapshot_moved = set(moved)
        snapshot_cut_count = current_cut_count
        proposals: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        for node in range(node_count):
            if node in moved:
                continue
            source = int(snapshot_owner[node])
            node_mass = int(mass[node])
            if node_mass == 0:
                # A sampled research mass can legitimately leave unseen upper
                # nodes at zero.  Moving one cannot improve the estimated load.
                continue
            source_load = int(snapshot_loads[source])
            if not _above_trigger(
                source_load, trigger_numerator, trigger_denominator
            ):
                continue

            edges_examined += len(graph[node])
            navigation_hits_examined += len(rows[node])
            neighbor_votes = Counter(
                int(snapshot_owner[neighbor]) for neighbor in graph[node]
            )
            navigation_votes = Counter(
                int(snapshot_owner[hit]) for hit in rows[node]
            )
            targets = set(neighbor_votes) | set(navigation_votes)
            best: tuple[tuple[int, ...], int, int, int] | None = None
            for target in sorted(targets):
                if target == source:
                    continue
                target_load = int(snapshot_loads[target])
                if not _below_trigger(
                    target_load, trigger_numerator, trigger_denominator
                ):
                    continue
                if not _at_or_below_trigger(
                    target_load + node_mass,
                    trigger_numerator,
                    trigger_denominator,
                ):
                    continue
                if source_load - target_load <= node_mass:
                    continue
                edge_delta = int(neighbor_votes[source] - neighbor_votes[target])
                navigation_loss = int(
                    navigation_votes[source] - navigation_votes[target]
                )
                target_key = (
                    max(edge_delta, 0),
                    max(navigation_loss, 0),
                    edge_delta,
                    navigation_loss,
                    target_load,
                    -node_mass,
                    int(target),
                )
                candidate = (target_key, int(target), edge_delta, navigation_loss)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            if best is None:
                continue
            target_key, target, edge_delta, navigation_loss = best
            global_key = (
                -source_load,
                target_key,
                int(node),
                source,
                target,
            )
            proposals.append(
                (
                    global_key,
                    {
                        "node": int(node),
                        "source": source,
                        "target": target,
                        "node_mass": node_mass,
                        "snapshot_source_load": source_load,
                        "snapshot_target_load": int(snapshot_loads[target]),
                        "snapshot_edge_delta": edge_delta,
                        "snapshot_navigation_loss": navigation_loss,
                        "target_key": [int(value) for value in target_key],
                        "proposal_key": [
                            -source_load,
                            [int(value) for value in target_key],
                            int(node),
                            source,
                            target,
                        ],
                        "snapshot_load_predicates": _load_predicate_evidence(
                            source_load=source_load,
                            target_load=int(snapshot_loads[target]),
                            node_mass=node_mass,
                            trigger_numerator=trigger_numerator,
                            trigger_denominator=trigger_denominator,
                        ),
                    },
                )
            )

        proposals.sort(key=lambda item: item[0])
        accepted: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for _global_key, proposal in proposals:
            node = int(proposal["node"])
            source = int(proposal["source"])
            target = int(proposal["target"])
            node_mass = int(proposal["node_mass"])
            if node in moved:
                decisions.append(
                    {
                        **proposal,
                        "status": "skipped",
                        "skip_reason": "node_already_moved",
                    }
                )
                continue
            if int(owner[node]) != source:
                decisions.append(
                    {
                        **proposal,
                        "status": "skipped",
                        "skip_reason": "owner_changed",
                    }
                )
                continue
            source_load = int(loads[source])
            target_load = int(loads[target])
            load_predicates = _load_predicate_evidence(
                source_load=source_load,
                target_load=target_load,
                node_mass=node_mass,
                trigger_numerator=trigger_numerator,
                trigger_denominator=trigger_denominator,
            )
            failed_load_predicate = next(
                (
                    name
                    for name, evidence in load_predicates.items()
                    if not bool(evidence["pass"])
                ),
                None,
            )
            if failed_load_predicate is not None:
                skip_reason = {
                    "source_above_trigger": "source_not_above_trigger",
                    "target_below_trigger": "target_not_below_trigger",
                    "source_target_gap_gt_node_mass": (
                        "source_target_gap_not_gt_node_mass"
                    ),
                    "post_move_target_at_or_below_trigger": (
                        "post_move_target_above_trigger"
                    ),
                }[failed_load_predicate]
                decisions.append(
                    {
                        **proposal,
                        "status": "skipped",
                        "skip_reason": skip_reason,
                        "commit_source_load_before": source_load,
                        "commit_target_load_before": target_load,
                        "commit_load_predicates": load_predicates,
                    }
                )
                continue

            source_neighbors = 0
            target_neighbors = 0
            edges_examined += len(graph[node])
            for neighbor in graph[node]:
                neighbor_owner = int(owner[neighbor])
                source_neighbors += neighbor_owner == source
                target_neighbors += neighbor_owner == target
            edge_delta = int(source_neighbors - target_neighbors)
            candidate_cut_count = current_cut_count + edge_delta
            candidate_cut_ratio = (
                candidate_cut_count / edge_count if edge_count else 0.0
            )
            cut_predicate = {
                "lhs": int(candidate_cut_count * 100),
                "operator": "<=",
                "rhs": int(reference_cut_count * 103),
                "pass": bool(
                    candidate_cut_count * 100 <= reference_cut_count * 103
                ),
            }
            if not bool(cut_predicate["pass"]):
                decisions.append(
                    {
                        **proposal,
                        "status": "skipped",
                        "skip_reason": "cumulative_edge_cut_cap",
                        "commit_source_load_before": source_load,
                        "commit_target_load_before": target_load,
                        "commit_load_predicates": load_predicates,
                        "commit_edge_delta": edge_delta,
                        "candidate_edge_cut_count": int(candidate_cut_count),
                        "cumulative_edge_cut_predicate": cut_predicate,
                    }
                )
                continue

            owner[node] = target
            loads[source] -= node_mass
            loads[target] += node_mass
            current_cut_count = candidate_cut_count
            moved.add(node)
            accepted_record = {
                **proposal,
                "status": "accepted",
                "skip_reason": None,
                "commit_source_load_before": source_load,
                "commit_target_load_before": target_load,
                "commit_source_load_after": int(loads[source]),
                "commit_target_load_after": int(loads[target]),
                "commit_load_predicates": load_predicates,
                "commit_edge_delta": edge_delta,
                "cumulative_edge_cut_count": int(current_cut_count),
                "cumulative_edge_cut_ratio": float(candidate_cut_ratio),
                "cumulative_edge_cut_predicate": cut_predicate,
            }
            accepted.append(accepted_record)
            decisions.append(accepted_record)

        if accepted:
            observed_metrics = _topology_metrics_normalized(
                graph,
                owner,
                partitions,
                edge_left=edge_left,
                edge_right=edge_right,
                degree=graph_degree,
            )
            gates = graph_gate_results(observed_metrics, reference_metrics)
            topology_evaluations += 1
        else:
            observed_metrics = current_metrics
            gates = current_gates
        gates_pass = all(bool(record["pass"]) for record in gates.values())
        record = {
            "round": round_index + 1,
            "pre_owner_sha256": _owner_sha256(snapshot_owner),
            "proposal_count": len(proposals),
            "proposals": [proposal for _key, proposal in proposals],
            "commit_decisions": decisions,
            "accepted_move_count": len(accepted),
            "accepted_moves": accepted,
            "pre_partition_masses": [int(value) for value in snapshot_loads.tolist()],
            "post_partition_masses": [int(value) for value in loads.tolist()],
            "post_owner_sha256_before_round_gate": _owner_sha256(owner),
            "post_topology_metrics": observed_metrics,
            "graph_gates": gates,
            "graph_gates_all_pass": gates_pass,
            "committed": gates_pass,
            "rolled_back": not gates_pass,
            "committed_move_count": len(accepted) if gates_pass else 0,
            "rolled_back_move_count": len(accepted) if not gates_pass else 0,
        }
        if not gates_pass:
            owner = snapshot_owner
            loads = snapshot_loads
            moved = snapshot_moved
            current_cut_count = snapshot_cut_count
            record["post_partition_masses_after_rollback"] = [
                int(value) for value in loads.tolist()
            ]
            record["post_owner_sha256_after_rollback"] = _owner_sha256(owner)
            round_records.append(record)
            break
        current_metrics = observed_metrics
        current_gates = gates
        round_records.append(record)
        if not accepted:
            break

    final_metrics = current_metrics
    final_gates = current_gates
    if not all(bool(record["pass"]) for record in final_gates.values()):
        raise AssertionError("CNBR returned an owner that fails its N-relative graph gates")
    return CnbrResult(
        owner=tuple(int(value) for value in owner.tolist()),
        partition_masses=tuple(int(value) for value in loads.tolist()),
        moved_nodes=tuple(sorted(moved)),
        rounds=tuple(round_records),
        topology_metrics=final_metrics,
        topology_gates=final_gates,
        trigger_ratio_numerator=int(trigger_ratio_numerator),
        trigger_ratio_denominator=int(trigger_ratio_denominator),
        trigger_mass_numerator=trigger_numerator,
        trigger_mass_denominator=trigger_denominator,
        edges_examined=int(edges_examined),
        navigation_hits_examined=int(navigation_hits_examined),
        topology_evaluations=int(topology_evaluations),
    )


def normalize_undirected_adjacency(
    adjacency: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if isinstance(adjacency, (str, bytes)):
        raise TypeError("adjacency must be a sequence of neighbor sequences")
    try:
        node_count = len(adjacency)
    except TypeError as exc:
        raise TypeError("adjacency must have a stable length") from exc
    if node_count <= 0:
        raise ValueError("adjacency must contain at least one node")
    graph = [set() for _ in range(node_count)]
    for node, row in enumerate(adjacency):
        if isinstance(row, (str, bytes)):
            raise TypeError(f"adjacency[{node}] must be a neighbor sequence")
        for raw_neighbor in row:
            if isinstance(raw_neighbor, bool) or not isinstance(raw_neighbor, Integral):
                raise TypeError(f"adjacency[{node}] contains a non-integer neighbor")
            neighbor = int(raw_neighbor)
            if neighbor < 0 or neighbor >= node_count:
                raise ValueError(f"adjacency[{node}] contains out-of-range node {neighbor}")
            if neighbor == node:
                continue
            graph[node].add(neighbor)
            graph[neighbor].add(node)
    return tuple(tuple(sorted(row)) for row in graph)


def _validated_navigation_rows(
    navigation_rows: Sequence[Sequence[int]],
    node_count: int,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(node_count, bool) or not isinstance(node_count, Integral):
        raise TypeError("node_count must be an integer")
    count = int(node_count)
    if count < UPPER_NAVIGATION_TOP_K:
        raise ValueError(
            f"node_count must be at least fixed top-k {UPPER_NAVIGATION_TOP_K}"
        )
    if isinstance(navigation_rows, (str, bytes)) or len(navigation_rows) != count:
        raise ValueError("navigation row count must equal upper node count")
    normalized: list[tuple[int, ...]] = []
    for row_index, row in enumerate(navigation_rows):
        values = tuple(row)
        if len(values) != UPPER_NAVIGATION_TOP_K:
            raise ValueError(
                f"navigation_rows[{row_index}] must contain exactly "
                f"{UPPER_NAVIGATION_TOP_K} hits"
            )
        converted: list[int] = []
        seen: set[int] = set()
        for raw_node in values:
            if isinstance(raw_node, bool) or not isinstance(raw_node, Integral):
                raise TypeError(
                    f"navigation_rows[{row_index}] contains a non-integer node"
                )
            node = int(raw_node)
            if node < 0 or node >= count:
                raise ValueError(
                    f"navigation_rows[{row_index}] contains out-of-range node {node}"
                )
            if node in seen:
                raise ValueError(
                    f"navigation_rows[{row_index}] repeats upper node {node}"
                )
            seen.add(node)
            converted.append(node)
        if row_index not in seen:
            raise ValueError(
                f"navigation_rows[{row_index}] does not contain its own upper node"
            )
        normalized.append(tuple(converted))
    return tuple(normalized)


def _validated_prepared_navigation(
    prepared_navigation: object,
    node_count: int,
) -> PreparedUpperNavigation:
    if not isinstance(prepared_navigation, PreparedUpperNavigation):
        raise TypeError(
            "prepared_navigation must be produced by prepare_upper_navigation"
        )
    if (
        getattr(prepared_navigation, "_seal", None)
        is not _PREPARED_UPPER_NAVIGATION_SEAL
    ):
        raise ValueError("prepared navigation lacks the private validation seal")
    if prepared_navigation.node_count != node_count:
        raise ValueError("prepared navigation node count differs from upper graph")
    if prepared_navigation.top_k != UPPER_NAVIGATION_TOP_K:
        raise ValueError("prepared navigation top-k differs from the frozen contract")
    rows = prepared_navigation.rows
    self_ranks = prepared_navigation.self_ranks
    if (
        not isinstance(rows, np.ndarray)
        or rows.dtype != np.dtype("<i4")
        or rows.shape != (node_count, UPPER_NAVIGATION_TOP_K)
        or rows.flags.writeable
        or not rows.flags.c_contiguous
    ):
        raise ValueError("prepared navigation rows lost their immutable canonical form")
    if (
        not isinstance(self_ranks, np.ndarray)
        or self_ranks.dtype != np.dtype(np.int8)
        or self_ranks.shape != (node_count,)
        or self_ranks.flags.writeable
        or not self_ranks.flags.c_contiguous
    ):
        raise ValueError("prepared navigation self ranks lost their immutable form")
    if not isinstance(prepared_navigation.mass, UpperNavigationMass):
        raise ValueError("prepared navigation mass lost its frozen contract")
    return prepared_navigation


def _validated_mass(
    navigation_mass: UpperNavigationMass,
    navigation_rows: tuple[tuple[int, ...], ...],
    node_count: int,
) -> tuple[int, ...]:
    if not isinstance(navigation_mass, UpperNavigationMass):
        raise TypeError("navigation_mass must be an UpperNavigationMass")
    expected = _mass_from_validated_rows(navigation_rows, node_count)
    if navigation_mass != expected:
        raise ValueError(
            "navigation mass is not the exact replay of the supplied self-navigation rows"
        )
    return expected.values


def _validated_external_mass(
    mass_values: Sequence[int] | np.ndarray,
    node_count: int,
) -> tuple[int, ...]:
    """Validate a research-only mass without assigning production semantics."""

    values = np.asarray(mass_values)
    if values.ndim != 1 or len(values) != node_count:
        raise ValueError("external research mass must contain one value per upper node")
    if values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise TypeError("external research mass must contain integers")
    if values.dtype.kind == "i" and bool(np.any(values < 0)):
        raise ValueError("external research mass contains a negative value")
    canonical = np.ascontiguousarray(values, dtype=np.uint64)
    if int(np.sum(canonical, dtype=np.uint64)) <= 0:
        raise ValueError("external research mass must contain positive total mass")
    return tuple(int(value) for value in canonical.tolist())


def _validated_vectors(
    vectors: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("upper_vectors must be a rectangular numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError("upper_vectors must be a non-empty two-dimensional matrix")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError("upper_vectors contains a non-finite value")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _validated_partition_count(num_partitions: int) -> int:
    if isinstance(num_partitions, bool) or not isinstance(num_partitions, Integral):
        raise TypeError("num_partitions must be an integer")
    partitions = int(num_partitions)
    if partitions <= 0:
        raise ValueError("num_partitions must be positive")
    return partitions


def _validated_owner(
    owner: Sequence[int] | np.ndarray,
    node_count: int,
    num_partitions: int,
) -> np.ndarray:
    array = np.asarray(owner)
    if array.ndim != 1 or len(array) != node_count:
        raise ValueError("owner must contain one partition per upper node")
    if array.dtype.kind not in "iu" or np.any(array < 0) or np.any(array >= num_partitions):
        raise ValueError("owner contains an invalid partition")
    return np.asarray(array, dtype=np.int32).copy()


def _native_kmeans_plus_plus(matrix: np.ndarray, partitions: int) -> np.ndarray:
    rng = np.random.default_rng(NATIVE_KMEANS_SEED)
    centroids = np.zeros((partitions, matrix.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, len(matrix)))
    centroids[0] = matrix[first]
    minimum = np.full(len(matrix), np.finfo(np.float32).max, dtype=np.float32)
    for partition in range(1, partitions):
        distance = np.sum((matrix - centroids[partition - 1]) ** 2, axis=1)
        minimum = np.minimum(minimum, distance)
        total = float(np.sum(minimum))
        if total <= 0.0:
            next_position = 0
        else:
            threshold = float(rng.random()) * total
            next_position = int(
                np.searchsorted(np.cumsum(minimum), threshold, side="left")
            )
            next_position = min(next_position, len(matrix) - 1)
        centroids[partition] = matrix[next_position]
    return centroids


def _squared_l2_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    point_norms = np.sum(points * points, axis=1, keepdims=True)
    centroid_norms = np.sum(centroids * centroids, axis=1, keepdims=True).T
    distances = point_norms + centroid_norms - 2.0 * (points @ centroids.T)
    return np.maximum(distances, 0.0)


def _partition_masses(
    owner: np.ndarray,
    mass: tuple[int, ...],
    num_partitions: int,
) -> np.ndarray:
    result = np.zeros(num_partitions, dtype=np.int64)
    np.add.at(result, owner, np.asarray(mass, dtype=np.int64))
    return result


def _owner_sha256(owner: np.ndarray) -> str:
    data = np.asarray(owner, dtype="<i4").tobytes(order="C")
    return hashlib.sha256(data).hexdigest()


def _load_predicate_evidence(
    *,
    source_load: int,
    target_load: int,
    node_mass: int,
    trigger_numerator: int,
    trigger_denominator: int,
) -> dict[str, dict[str, int | str | bool]]:
    source_lhs = int(source_load) * int(trigger_denominator)
    target_lhs = int(target_load) * int(trigger_denominator)
    gap_lhs = int(source_load) - int(target_load)
    post_target_lhs = (int(target_load) + int(node_mass)) * int(
        trigger_denominator
    )
    return {
        "source_above_trigger": {
            "lhs": source_lhs,
            "operator": ">",
            "rhs": int(trigger_numerator),
            "pass": bool(source_lhs > trigger_numerator),
        },
        "target_below_trigger": {
            "lhs": target_lhs,
            "operator": "<",
            "rhs": int(trigger_numerator),
            "pass": bool(target_lhs < trigger_numerator),
        },
        "source_target_gap_gt_node_mass": {
            "lhs": gap_lhs,
            "operator": ">",
            "rhs": int(node_mass),
            "pass": bool(gap_lhs > node_mass),
        },
        "post_move_target_at_or_below_trigger": {
            "lhs": post_target_lhs,
            "operator": "<=",
            "rhs": int(trigger_numerator),
            "pass": bool(post_target_lhs <= trigger_numerator),
        },
    }


def _edge_arrays(
    graph: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    edges = [
        (left, right)
        for left, row in enumerate(graph)
        for right in row
        if left < right
    ]
    return (
        np.fromiter((left for left, _right in edges), dtype=np.int32),
        np.fromiter((right for _left, right in edges), dtype=np.int32),
    )


def _above_trigger(load: int, numerator: int, denominator: int) -> bool:
    return int(load) * int(denominator) > int(numerator)


def _below_trigger(load: int, numerator: int, denominator: int) -> bool:
    return int(load) * int(denominator) < int(numerator)


def _at_or_below_trigger(load: int, numerator: int, denominator: int) -> bool:
    return int(load) * int(denominator) <= int(numerator)
