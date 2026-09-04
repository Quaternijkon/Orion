"""Upper-only navigation-mass partitioners for Orion's frozen L1 graph.

The mass estimate is intentionally cheap and narrow: query every one of the
``U`` frozen upper vectors against the *same production upper graph*, retain
exactly the first ten hits, and count how often each upper vertex is hit.  No
full-dataset attachment, observed L0 load, or multi-assignment state is accepted
by this module.

Unlike :mod:`l1_partitioner`, these candidates do not force equal L1 node
counts.  They preserve graph/geometry locality while constraining the estimated
navigation mass of every partition to a deterministic discrete upper bound.
The resulting owner still assigns each L1 vertex to exactly one class; Orion's
unchanged L0 multi-assignment is applied only after the owner is frozen.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import heapq
import math
from numbers import Integral
import random
import struct
from time import perf_counter
from typing import Any


UPPER_NAVIGATION_TOP_K = 10
MASS_DETERMINISTIC_SEED = 0
MASS_KMEANS_ITERATIONS = 8
MASS_SOURCE = "production_upper_navigation_top10_self_debiased_floor1"
RAW_MASS_SOURCE = "production_upper_navigation_top10_raw_hit_frequency"
REGULARIZED_MASS_SOURCE = (
    "production_upper_navigation_top10_hit_frequency_regularized_v2"
)
MASS_TRANSFORM = "max(1, raw_hit_count - 1)"
RAW_MASS_TRANSFORM = "raw_hit_count"
REGULARIZED_MASS_TRANSFORM = "raw_count_with_unit_l1_prior"
MASS_SUPPORTED_ALGORITHMS = (
    "mass-ldg",
    "mass-region-grow",
    "mass-balanced-kmeans",
)


@dataclass(frozen=True, init=False)
class UpperNavigationMass:
    """Validated mass derived from exactly ``U`` production L1 searches.

    Direct construction is disabled so the partition API cannot silently
    accept an arbitrary vector of downstream L0 loads.  Use
    :func:`estimate_upper_navigation_mass` with local upper-navigation rows.
    """

    values: tuple[int, ...]
    query_count: int
    top_k: int
    total_mass: int
    source: str
    transform: str
    estimator_version: int
    sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "UpperNavigationMass is derived state; use "
            "estimate_upper_navigation_mass()"
        )


@dataclass(frozen=True)
class MassPartitionResult:
    """One weighted L1 classification and its fixed capacity contract."""

    algorithm: str
    owner: tuple[int, ...]
    estimated_partition_masses: tuple[int, ...]
    mass_target: float
    mass_limit: int
    mass_sha256: str
    elapsed_seconds: float
    edge_visits: int

    @property
    def partition_sizes(self) -> tuple[int, ...]:
        sizes = [0] * len(self.estimated_partition_masses)
        for partition in self.owner:
            sizes[partition] += 1
        return tuple(sizes)


def estimate_upper_navigation_mass(
    navigation_rows: Sequence[Sequence[int]],
    node_count: int,
) -> UpperNavigationMass:
    """Count non-self top-10 hit frequency from all ``U`` upper queries.

    ``navigation_rows`` must contain exactly ``node_count`` rows and every row
    must contain ten distinct *local L1 node IDs*.  Thus the estimator cost is
    ``U`` production upper-graph searches, where Orion samples ``U = N / 32``
    upper vertices.  Row ``u`` must contain ``u`` itself; that one guaranteed
    self hit is subtracted from every vertex's raw count, then a floor of one is
    applied.  The transform is fixed as ``max(1, raw_hit_count - 1)`` before
    either SIFT or GloVe screening.  Rank weighting is deliberately absent.
    """

    return _estimate_upper_navigation_mass(
        navigation_rows, node_count, mode="self-debiased-floor1"
    )


def estimate_raw_upper_navigation_mass(
    navigation_rows: Sequence[Sequence[int]],
    node_count: int,
) -> UpperNavigationMass:
    """Raw top-10 hit frequency retained only as the fixed diagnostic control."""

    return _estimate_upper_navigation_mass(
        navigation_rows, node_count, mode="raw"
    )


def estimate_regularized_upper_navigation_mass(
    navigation_rows: Sequence[Sequence[int]],
    node_count: int,
) -> UpperNavigationMass:
    """V2 confirmation mass: non-self hit count plus one unit L1 prior."""

    return _estimate_upper_navigation_mass(
        navigation_rows, node_count, mode="raw-regularized-v2"
    )


def _estimate_upper_navigation_mass(
    navigation_rows: Sequence[Sequence[int]],
    node_count: int,
    *,
    mode: str,
) -> UpperNavigationMass:

    count = _validate_node_count(node_count)
    if isinstance(navigation_rows, (str, bytes)):
        raise TypeError("navigation_rows must be upper-navigation hit rows")
    try:
        row_count = len(navigation_rows)
    except TypeError as exc:
        raise TypeError("navigation_rows must have a stable length") from exc
    if row_count != count:
        raise ValueError(
            "upper-navigation query count must equal L1 node count: "
            f"rows={row_count}, nodes={count}"
        )

    mass = [0] * count
    for row_index, row in enumerate(navigation_rows):
        if isinstance(row, (str, bytes)):
            raise TypeError(f"navigation_rows[{row_index}] must be a hit sequence")
        try:
            hits = tuple(row)
        except TypeError as exc:
            raise TypeError(
                f"navigation_rows[{row_index}] must be iterable"
            ) from exc
        if len(hits) != UPPER_NAVIGATION_TOP_K:
            raise ValueError(
                f"navigation_rows[{row_index}] has {len(hits)} hits; "
                f"expected fixed top-k {UPPER_NAVIGATION_TOP_K}"
            )
        seen: set[int] = set()
        for raw_node in hits:
            if isinstance(raw_node, bool) or not isinstance(raw_node, Integral):
                raise TypeError(
                    f"navigation_rows[{row_index}] contains non-integer L1 ID "
                    f"{raw_node!r}"
                )
            node = int(raw_node)
            if node < 0 or node >= count:
                raise ValueError(
                    f"navigation_rows[{row_index}] contains out-of-range L1 ID "
                    f"{node}"
                )
            if node in seen:
                raise ValueError(
                    f"navigation_rows[{row_index}] repeats L1 ID {node}"
                )
            seen.add(node)
            mass[node] += 1

        if row_index not in seen:
            raise ValueError(
                f"navigation_rows[{row_index}] lacks its L1 self hit"
            )
        if mode == "self-debiased-floor1":
            mass[row_index] -= 1

    if mode == "self-debiased-floor1":
        mass = [max(1, value) for value in mass]
    values = tuple(mass)
    total_mass = sum(values)
    digest = hashlib.sha256()
    if mode == "self-debiased-floor1":
        source = MASS_SOURCE
        transform = MASS_TRANSFORM
        estimator_version = 1
    elif mode == "raw":
        source = RAW_MASS_SOURCE
        transform = RAW_MASS_TRANSFORM
        estimator_version = 1
    elif mode == "raw-regularized-v2":
        source = REGULARIZED_MASS_SOURCE
        transform = REGULARIZED_MASS_TRANSFORM
        estimator_version = 2
    else:
        raise AssertionError(f"unsupported internal mass mode {mode!r}")
    digest.update(source.encode("ascii"))
    digest.update(b"\0")
    digest.update(transform.encode("ascii"))
    digest.update(struct.pack("<QQ", count, UPPER_NAVIGATION_TOP_K))
    for value in values:
        digest.update(struct.pack("<Q", value))

    result = object.__new__(UpperNavigationMass)
    object.__setattr__(result, "values", values)
    object.__setattr__(result, "query_count", count)
    object.__setattr__(result, "top_k", UPPER_NAVIGATION_TOP_K)
    object.__setattr__(result, "total_mass", total_mass)
    object.__setattr__(result, "source", source)
    object.__setattr__(result, "transform", transform)
    object.__setattr__(result, "estimator_version", estimator_version)
    object.__setattr__(result, "sha256", digest.hexdigest())
    return result


def partition_l1_by_navigation_mass(
    adjacency: Sequence[Sequence[int]],
    num_partitions: int,
    *,
    algorithm: str,
    navigation_mass: UpperNavigationMass,
    vectors: Sequence[Sequence[float]] | Any | None = None,
    entry_point: int | None = None,
) -> MassPartitionResult:
    """Partition L1 using only its graph, vectors, and derived navigation mass.

    All algorithm parameters are fixed module constants.  ``vectors`` is read
    only by ``mass-balanced-kmeans``.  The graph algorithms ignore it.  There
    is intentionally no ``**kwargs`` escape hatch and no L0-related argument.
    """

    started = perf_counter()
    graph = _normalise_graph(adjacency)
    node_count = len(graph)
    partitions = _validate_partition_count(num_partitions, node_count)
    root = _validate_entry_point(entry_point, node_count)
    mass = _validated_mass(navigation_mass, node_count)
    mass_target = sum(mass) / partitions
    mass_limit = math.ceil(mass_target) + max(mass) - 1

    if algorithm == "mass-ldg":
        owner, edge_visits = _mass_ldg(
            graph, mass, partitions, mass_target, mass_limit, root
        )
    elif algorithm == "mass-region-grow":
        owner, edge_visits = _mass_region_grow(
            graph, mass, partitions, mass_target, mass_limit, root
        )
    elif algorithm == "mass-balanced-kmeans":
        owner, edge_visits = _mass_balanced_kmeans(
            vectors,
            mass,
            partitions,
            mass_target,
            mass_limit,
            MASS_DETERMINISTIC_SEED,
            MASS_KMEANS_ITERATIONS,
        )
    else:
        choices = ", ".join(MASS_SUPPORTED_ALGORITHMS)
        raise ValueError(f"unknown algorithm {algorithm!r}; expected one of: {choices}")

    frozen_owner = tuple(int(value) for value in owner)
    partition_masses = _assert_mass_contract(
        frozen_owner, mass, partitions, mass_limit
    )
    return MassPartitionResult(
        algorithm=algorithm,
        owner=frozen_owner,
        estimated_partition_masses=partition_masses,
        mass_target=mass_target,
        mass_limit=mass_limit,
        mass_sha256=navigation_mass.sha256,
        elapsed_seconds=perf_counter() - started,
        edge_visits=int(edge_visits),
    )


def _validate_node_count(node_count: int) -> int:
    if isinstance(node_count, bool) or not isinstance(node_count, Integral):
        raise TypeError("node_count must be an integer")
    count = int(node_count)
    if count < UPPER_NAVIGATION_TOP_K:
        raise ValueError(
            f"node_count must be at least top-k {UPPER_NAVIGATION_TOP_K}"
        )
    return count


def _validated_mass(
    navigation_mass: UpperNavigationMass, node_count: int
) -> tuple[int, ...]:
    if not isinstance(navigation_mass, UpperNavigationMass):
        raise TypeError(
            "navigation_mass must come from estimate_upper_navigation_mass()"
        )
    if navigation_mass.source not in {
        MASS_SOURCE,
        RAW_MASS_SOURCE,
        REGULARIZED_MASS_SOURCE,
    }:
        raise ValueError("navigation mass has an unsupported source")
    expected_transform = {
        MASS_SOURCE: MASS_TRANSFORM,
        RAW_MASS_SOURCE: RAW_MASS_TRANSFORM,
        REGULARIZED_MASS_SOURCE: REGULARIZED_MASS_TRANSFORM,
    }[navigation_mass.source]
    if navigation_mass.transform != expected_transform:
        raise ValueError("navigation mass has an unsupported transform")
    expected_version = 2 if navigation_mass.source == REGULARIZED_MASS_SOURCE else 1
    if navigation_mass.estimator_version != expected_version:
        raise ValueError("navigation mass has an unsupported estimator version")
    if navigation_mass.query_count != node_count:
        raise ValueError("navigation mass node count differs from adjacency")
    if navigation_mass.top_k != UPPER_NAVIGATION_TOP_K:
        raise ValueError("navigation mass top-k differs from the fixed contract")
    values = navigation_mass.values
    if len(values) != node_count:
        raise ValueError("navigation mass vector length differs from adjacency")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
        for value in values
    ):
        raise ValueError("navigation mass values must be non-negative integers")
    frozen = tuple(int(value) for value in values)
    total_mass = sum(frozen)
    if total_mass != navigation_mass.total_mass:
        raise ValueError("navigation mass total differs from its bound metadata")
    if navigation_mass.source == MASS_SOURCE:
        lower = node_count * (UPPER_NAVIGATION_TOP_K - 1)
        upper = node_count * UPPER_NAVIGATION_TOP_K
        if min(frozen) < 1 or not lower <= total_mass <= upper:
            raise ValueError("self-debiased navigation mass violates its bounds")
    elif total_mass != node_count * UPPER_NAVIGATION_TOP_K:
        raise ValueError("raw navigation mass total differs from U * top-k")
    digest = hashlib.sha256()
    digest.update(navigation_mass.source.encode("ascii"))
    digest.update(b"\0")
    digest.update(navigation_mass.transform.encode("ascii"))
    digest.update(struct.pack("<QQ", node_count, UPPER_NAVIGATION_TOP_K))
    for value in frozen:
        digest.update(struct.pack("<Q", value))
    if digest.hexdigest() != navigation_mass.sha256:
        raise ValueError("navigation mass checksum mismatch")
    return frozen


def _normalise_graph(
    adjacency: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if isinstance(adjacency, (str, bytes)):
        raise TypeError("adjacency must be a sequence of neighbour sequences")
    try:
        node_count = len(adjacency)
    except TypeError as exc:
        raise TypeError("adjacency must have a stable length") from exc
    if node_count < UPPER_NAVIGATION_TOP_K:
        raise ValueError(
            f"adjacency must contain at least {UPPER_NAVIGATION_TOP_K} L1 nodes"
        )

    neighbours: list[set[int]] = [set() for _ in range(node_count)]
    for node, row in enumerate(adjacency):
        if isinstance(row, (str, bytes)):
            raise TypeError(f"adjacency[{node}] must be a neighbour sequence")
        try:
            iterator = iter(row)
        except TypeError as exc:
            raise TypeError(f"adjacency[{node}] must be iterable") from exc
        for raw_neighbour in iterator:
            if isinstance(raw_neighbour, bool) or not isinstance(
                raw_neighbour, Integral
            ):
                raise TypeError(
                    f"adjacency[{node}] contains non-integer neighbour "
                    f"{raw_neighbour!r}"
                )
            neighbour = int(raw_neighbour)
            if neighbour < 0 or neighbour >= node_count:
                raise ValueError(
                    f"adjacency[{node}] contains out-of-range node {neighbour}"
                )
            if neighbour != node:
                neighbours[node].add(neighbour)
                neighbours[neighbour].add(node)
    return tuple(tuple(sorted(row)) for row in neighbours)


def _validate_partition_count(num_partitions: int, node_count: int) -> int:
    if isinstance(num_partitions, bool) or not isinstance(num_partitions, Integral):
        raise TypeError("num_partitions must be an integer")
    partitions = int(num_partitions)
    if partitions < 1 or partitions > node_count:
        raise ValueError(
            f"num_partitions must be in [1, {node_count}], got {partitions}"
        )
    return partitions


def _validate_entry_point(entry_point: int | None, node_count: int) -> int:
    if entry_point is None:
        return 0
    if isinstance(entry_point, bool) or not isinstance(entry_point, Integral):
        raise TypeError("entry_point must be an integer or None")
    root = int(entry_point)
    if root < 0 or root >= node_count:
        raise ValueError(f"entry_point must be in [0, {node_count}), got {root}")
    return root


def _bfs_order(
    graph: tuple[tuple[int, ...], ...], entry_point: int
) -> tuple[list[int], int]:
    visited = bytearray(len(graph))
    order: list[int] = []
    edge_visits = 0
    roots = [entry_point]
    roots.extend(node for node in range(len(graph)) if node != entry_point)
    for root in roots:
        if visited[root]:
            continue
        visited[root] = 1
        queue = deque([root])
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in graph[node]:
                edge_visits += 1
                if not visited[neighbour]:
                    visited[neighbour] = 1
                    queue.append(neighbour)
    return order, edge_visits


def _spread_seeds(order: Sequence[int], partitions: int) -> tuple[int, ...]:
    node_count = len(order)
    seeds = tuple(order[(partition * node_count) // partitions] for partition in range(partitions))
    if len(set(seeds)) != partitions:
        raise AssertionError("deterministic seed selection produced duplicates")
    return seeds


def _fitting_partitions(
    loads: Sequence[int], weight: int, mass_limit: int
) -> list[int]:
    candidates = [
        partition
        for partition, load in enumerate(loads)
        if load + weight <= mass_limit
    ]
    if not candidates:
        raise AssertionError("weighted capacity became infeasible")
    return candidates


def _mass_ldg(
    graph: tuple[tuple[int, ...], ...],
    mass: tuple[int, ...],
    partitions: int,
    mass_target: float,
    mass_limit: int,
    entry_point: int,
) -> tuple[list[int], int]:
    order, edge_visits = _bfs_order(graph, entry_point)
    seeds = _spread_seeds(order, partitions)
    owner = [-1] * len(graph)
    loads = [0] * partitions
    for partition, node in enumerate(seeds):
        owner[node] = partition
        loads[partition] = mass[node]

    for node in order:
        if owner[node] >= 0:
            continue
        neighbour_counts: dict[int, int] = {}
        for neighbour in graph[node]:
            edge_visits += 1
            partition = owner[neighbour]
            if partition >= 0:
                neighbour_counts[partition] = neighbour_counts.get(partition, 0) + 1
        candidates = _fitting_partitions(loads, mass[node], mass_limit)
        selected = max(
            candidates,
            key=lambda partition: (
                neighbour_counts.get(partition, 0)
                * (mass_limit - loads[partition])
                / mass_limit,
                neighbour_counts.get(partition, 0),
                -(loads[partition] + mass[node]) / mass_target,
                -partition,
            ),
        )
        owner[node] = selected
        loads[selected] += mass[node]
    return owner, edge_visits


def _mass_region_grow(
    graph: tuple[tuple[int, ...], ...],
    mass: tuple[int, ...],
    partitions: int,
    mass_target: float,
    mass_limit: int,
    entry_point: int,
) -> tuple[list[int], int]:
    order, edge_visits = _bfs_order(graph, entry_point)
    seeds = _spread_seeds(order, partitions)
    owner = [-1] * len(graph)
    loads = [0] * partitions
    frontiers: list[list[tuple[int, int]]] = [[] for _ in range(partitions)]
    queued: list[set[int]] = [set() for _ in range(partitions)]

    def expose(partition: int, node: int, distance: int) -> None:
        nonlocal edge_visits
        for neighbour in graph[node]:
            edge_visits += 1
            if owner[neighbour] < 0 and neighbour not in queued[partition]:
                queued[partition].add(neighbour)
                heapq.heappush(frontiers[partition], (distance + 1, neighbour))

    for partition, node in enumerate(seeds):
        owner[node] = partition
        loads[partition] = mass[node]
    for partition, node in enumerate(seeds):
        expose(partition, node, 0)

    remaining = len(graph) - partitions
    fallback_offset = 0
    while remaining:
        active: list[tuple[int, float, int, int]] = []
        for partition, frontier in enumerate(frontiers):
            while frontier:
                distance, node = frontier[0]
                if owner[node] >= 0 or loads[partition] + mass[node] > mass_limit:
                    heapq.heappop(frontier)
                    continue
                active.append(
                    (distance, loads[partition] / mass_target, node, partition)
                )
                break

        if active:
            _distance_key, _load_key, node, selected = min(active)
            distance, popped = heapq.heappop(frontiers[selected])
            if popped != node:
                raise AssertionError("region frontier changed unexpectedly")
        else:
            while owner[order[fallback_offset]] >= 0:
                fallback_offset += 1
            node = order[fallback_offset]
            candidates = _fitting_partitions(loads, mass[node], mass_limit)
            neighbour_counts: dict[int, int] = {}
            for neighbour in graph[node]:
                edge_visits += 1
                partition = owner[neighbour]
                if partition >= 0:
                    neighbour_counts[partition] = neighbour_counts.get(partition, 0) + 1
            selected = max(
                candidates,
                key=lambda partition: (
                    neighbour_counts.get(partition, 0),
                    -(loads[partition] + mass[node]) / mass_target,
                    -partition,
                ),
            )
            distance = 0

        if owner[node] >= 0:
            continue
        owner[node] = selected
        loads[selected] += mass[node]
        remaining -= 1
        expose(selected, node, distance)
    return owner, edge_visits


def _mass_balanced_kmeans(
    vectors: Sequence[Sequence[float]] | Any | None,
    mass: tuple[int, ...],
    partitions: int,
    mass_target: float,
    mass_limit: int,
    seed: int,
    iterations: int,
) -> tuple[list[int], int]:
    if vectors is None:
        raise ValueError("mass-balanced-kmeans requires upper vectors")
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("mass-balanced-kmeans requires NumPy") from exc

    try:
        matrix = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("vectors must be a rectangular numeric matrix") from exc
    node_count = len(mass)
    if matrix.ndim != 2 or matrix.shape[0] != node_count or matrix.shape[1] == 0:
        raise ValueError(
            "vectors must have shape "
            f"({node_count}, dimension>0), got {tuple(matrix.shape)}"
        )
    if not bool(np.isfinite(matrix).all()):
        raise ValueError("vectors must contain only finite values")

    first = random.Random(seed).randrange(node_count)
    selected_nodes = [first]
    selected_set = {first}
    centroids = [matrix[first].copy()]
    minimum_distance = ((matrix - matrix[first]) ** 2).sum(axis=1)
    for _partition in range(1, partitions):
        farthest = max(
            (node for node in range(node_count) if node not in selected_set),
            key=lambda node: (float(minimum_distance[node]), -node),
        )
        selected_nodes.append(farthest)
        selected_set.add(farthest)
        centroids.append(matrix[farthest].copy())
        candidate_distance = ((matrix - matrix[farthest]) ** 2).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, candidate_distance)
    centroid_matrix = np.stack(centroids, axis=0)
    mass_array = np.asarray(mass, dtype=np.int64)
    node_ids = np.arange(node_count, dtype=np.int64)
    row_norm = (matrix * matrix).sum(axis=1, keepdims=True)

    previous_owner = None
    owner_array = np.full(node_count, -1, dtype=np.int64)
    for _iteration in range(iterations):
        distances = (
            row_norm
            + (centroid_matrix * centroid_matrix).sum(axis=1)[None, :]
            - 2.0 * matrix.dot(centroid_matrix.T)
        )
        np.maximum(distances, 0.0, out=distances)
        preferences = np.argsort(distances, axis=1, kind="stable")
        best_distance = distances[node_ids, preferences[:, 0]]
        if partitions == 1:
            margin = np.zeros(node_count, dtype=np.float64)
        else:
            margin = distances[node_ids, preferences[:, 1]] - best_distance
        node_order = np.lexsort((node_ids, best_distance, -mass_array, -margin))

        owner_array.fill(-1)
        loads = np.zeros(partitions, dtype=np.int64)
        for partition, node in enumerate(selected_nodes):
            owner_array[node] = partition
            loads[partition] += mass_array[node]
        for raw_node in node_order:
            node = int(raw_node)
            if owner_array[node] >= 0:
                continue
            weight = int(mass_array[node])
            assigned = False
            for raw_partition in preferences[node]:
                partition = int(raw_partition)
                if int(loads[partition]) + weight <= mass_limit:
                    owner_array[node] = partition
                    loads[partition] += weight
                    assigned = True
                    break
            if not assigned:
                raise AssertionError("weighted K-means capacity became infeasible")

        if previous_owner is not None and bool(np.array_equal(owner_array, previous_owner)):
            break
        previous_owner = owner_array.copy()
        for partition in range(partitions):
            mask = owner_array == partition
            partition_mass = mass_array[mask]
            total_mass = int(partition_mass.sum())
            if total_mass > 0:
                centroid_matrix[partition] = np.average(
                    matrix[mask], axis=0, weights=partition_mass
                )
            else:
                centroid_matrix[partition] = matrix[mask].mean(axis=0)

    return [int(value) for value in owner_array.tolist()], 0


def _assert_mass_contract(
    owner: tuple[int, ...],
    mass: tuple[int, ...],
    partitions: int,
    mass_limit: int,
) -> tuple[int, ...]:
    sizes = [0] * partitions
    loads = [0] * partitions
    for node, partition in enumerate(owner):
        if partition < 0 or partition >= partitions:
            raise AssertionError(f"node {node} has invalid partition {partition}")
        sizes[partition] += 1
        loads[partition] += mass[node]
    if any(size == 0 for size in sizes):
        raise AssertionError(f"weighted partitioner produced an empty class: {sizes}")
    if max(loads) > mass_limit:
        raise AssertionError(
            f"weighted partitioner exceeded mass limit {mass_limit}: {loads}"
        )
    if sum(loads) != sum(mass):
        raise AssertionError("weighted partitioner lost navigation mass")
    return tuple(loads)
