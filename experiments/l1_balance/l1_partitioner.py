"""Deterministic, exact-quota partitioners for Orion's frozen L1 graph.

This module deliberately has a narrow input boundary.  A partitioner may inspect
only the immutable upper-graph adjacency, the optional upper vectors, the upper
entry point, and the requested partition count.  In particular, this stage has
no API for L0 attachments, observed shard sizes, or multi-assignment state.

All graph algorithms operate on the deterministic undirected simple graph formed
by symmetrising ``adjacency``.  This is an experimental L1 classification step;
the returned ``owner[u]`` is one L1 class, not an L0 storage-membership list.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
import heapq
import math
from numbers import Integral
import random
from time import perf_counter
from typing import Any


FENNEL_GAMMA = 1.5
DETERMINISTIC_SEED = 0
KMEANS_ITERATIONS = 8
SUPPORTED_ALGORITHMS = (
    "random",
    "balanced-kmeans",
    "ldg0",
    "ldg-swap1",
    "fennel",
    "balanced-region-grow",
)


@dataclass(frozen=True)
class PartitionResult:
    """One exact-quota L1 classification and its construction cost counters."""

    algorithm: str
    owner: tuple[int, ...]
    capacities: tuple[int, ...]
    elapsed_seconds: float
    edge_visits: int

    @property
    def partition_sizes(self) -> tuple[int, ...]:
        sizes = [0] * len(self.capacities)
        for partition in self.owner:
            sizes[partition] += 1
        return tuple(sizes)


def partition_l1(
    adjacency: Sequence[Sequence[int]],
    num_partitions: int,
    *,
    algorithm: str,
    vectors: Sequence[Sequence[float]] | Any | None = None,
    entry_point: int | None = None,
) -> PartitionResult:
    """Partition a frozen L1 graph without consulting any L0-derived state.

    ``algorithm`` is one of :data:`SUPPORTED_ALGORITHMS`.  ``vectors`` is read
    only by ``balanced-kmeans``; changing it cannot affect any graph-only method.
    The random seed and K-means iteration count are fixed module constants.
    Fennel likewise fixes ``gamma`` at 1.5.  Keeping them out of the API prevents
    candidate-specific tuning based on downstream L0 outcomes.
    """

    started = perf_counter()
    graph = _normalise_graph(adjacency)
    node_count = len(graph)
    partitions = _validate_partition_count(num_partitions, node_count)
    capacities = _balanced_capacities(node_count, partitions)
    root = _validate_entry_point(entry_point, node_count)

    if algorithm == "random":
        owner, edge_visits = _random_exact_quota(
            node_count, capacities, DETERMINISTIC_SEED
        )
    elif algorithm == "balanced-kmeans":
        owner, edge_visits = _balanced_kmeans(
            vectors,
            node_count,
            capacities,
            DETERMINISTIC_SEED,
            KMEANS_ITERATIONS,
        )
    elif algorithm == "ldg0":
        owner, edge_visits = _ldg0(graph, capacities, root)
    elif algorithm == "ldg-swap1":
        owner, edge_visits = _ldg_swap1(graph, capacities, root)
    elif algorithm == "fennel":
        owner, edge_visits = _fennel(graph, capacities, root)
    elif algorithm == "balanced-region-grow":
        owner, edge_visits = _balanced_region_grow(graph, capacities, root)
    else:
        choices = ", ".join(SUPPORTED_ALGORITHMS)
        raise ValueError(f"unknown algorithm {algorithm!r}; expected one of: {choices}")

    frozen_owner = tuple(int(value) for value in owner)
    _assert_exact_quota(frozen_owner, capacities)
    return PartitionResult(
        algorithm=algorithm,
        owner=frozen_owner,
        capacities=capacities,
        elapsed_seconds=perf_counter() - started,
        edge_visits=int(edge_visits),
    )


def _normalise_graph(
    adjacency: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if isinstance(adjacency, (str, bytes)):
        raise TypeError("adjacency must be a sequence of neighbour sequences")
    try:
        node_count = len(adjacency)
    except TypeError as exc:
        raise TypeError("adjacency must have a stable length") from exc
    if node_count == 0:
        raise ValueError("adjacency must contain at least one L1 node")

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
            if neighbour == node:
                continue
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


def _balanced_capacities(
    node_count: int, num_partitions: int
) -> tuple[int, ...]:
    quotient, remainder = divmod(node_count, num_partitions)
    return tuple(
        quotient + (1 if partition < remainder else 0)
        for partition in range(num_partitions)
    )


def _assert_exact_quota(
    owner: tuple[int, ...], capacities: tuple[int, ...]
) -> None:
    sizes = [0] * len(capacities)
    for node, partition in enumerate(owner):
        if partition < 0 or partition >= len(capacities):
            raise AssertionError(f"node {node} has invalid partition {partition}")
        sizes[partition] += 1
    if tuple(sizes) != capacities:
        raise AssertionError(
            f"partitioner violated exact quota: sizes={tuple(sizes)}, "
            f"capacities={capacities}"
        )


def _bfs_order(
    graph: tuple[tuple[int, ...], ...], entry_point: int
) -> tuple[list[int], int]:
    node_count = len(graph)
    visited = bytearray(node_count)
    order: list[int] = []
    edge_visits = 0
    roots = [entry_point]
    roots.extend(node for node in range(node_count) if node != entry_point)
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


def _random_exact_quota(
    node_count: int, capacities: tuple[int, ...], seed: int
) -> tuple[list[int], int]:
    nodes = list(range(node_count))
    random.Random(seed).shuffle(nodes)
    owner = [-1] * node_count
    offset = 0
    for partition, capacity in enumerate(capacities):
        for node in nodes[offset : offset + capacity]:
            owner[node] = partition
        offset += capacity
    return owner, 0


def _balanced_kmeans(
    vectors: Sequence[Sequence[float]] | Any | None,
    node_count: int,
    capacities: tuple[int, ...],
    seed: int,
    iterations: int,
) -> tuple[list[int], int]:
    if vectors is None:
        raise ValueError("balanced-kmeans requires upper vectors")
    try:
        import numpy as np
    except ModuleNotFoundError:
        return _balanced_kmeans_python(
            vectors, node_count, capacities, seed, iterations
        )

    try:
        matrix = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("vectors must be a rectangular numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] != node_count or matrix.shape[1] == 0:
        raise ValueError(
            "vectors must have shape "
            f"({node_count}, dimension>0), got {tuple(matrix.shape)}"
        )
    if not bool(np.isfinite(matrix).all()):
        raise ValueError("vectors must contain only finite values")

    partition_count = len(capacities)
    rng = random.Random(seed)
    first = rng.randrange(node_count)
    selected = {first}
    centroids = [matrix[first].copy()]
    minimum_distance = ((matrix - matrix[first]) ** 2).sum(axis=1)
    for _partition in range(1, partition_count):
        unselected = [node for node in range(node_count) if node not in selected]
        farthest = max(
            unselected,
            key=lambda node: (float(minimum_distance[node]), -node),
        )
        selected.add(farthest)
        centroids.append(matrix[farthest].copy())
        candidate_distance = ((matrix - matrix[farthest]) ** 2).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, candidate_distance)
    centroid_matrix = np.stack(centroids, axis=0)

    previous_owner = None
    owner_array = np.empty(node_count, dtype=np.int64)
    capacity_array = np.asarray(capacities, dtype=np.int64)
    row_norm = (matrix * matrix).sum(axis=1, keepdims=True)
    for _iteration in range(iterations):
        distances = (
            row_norm
            + (centroid_matrix * centroid_matrix).sum(axis=1)[None, :]
            - 2.0 * matrix.dot(centroid_matrix.T)
        )
        np.maximum(distances, 0.0, out=distances)
        preferences = np.argsort(distances, axis=1, kind="stable")
        best_distance = distances[np.arange(node_count), preferences[:, 0]]
        if partition_count == 1:
            node_order = np.arange(node_count)
        else:
            second_distance = distances[np.arange(node_count), preferences[:, 1]]
            margin = second_distance - best_distance
            node_order = np.lexsort(
                (np.arange(node_count), best_distance, -margin)
            )

        sizes = np.zeros(partition_count, dtype=np.int64)
        for raw_node in node_order:
            node = int(raw_node)
            for raw_partition in preferences[node]:
                partition = int(raw_partition)
                if sizes[partition] < capacity_array[partition]:
                    owner_array[node] = partition
                    sizes[partition] += 1
                    break

        if previous_owner is not None and bool(np.array_equal(owner_array, previous_owner)):
            break
        previous_owner = owner_array.copy()
        for partition in range(partition_count):
            centroid_matrix[partition] = matrix[owner_array == partition].mean(axis=0)

    return [int(value) for value in owner_array.tolist()], 0


def _balanced_kmeans_python(
    vectors: Sequence[Sequence[float]],
    node_count: int,
    capacities: tuple[int, ...],
    seed: int,
    iterations: int,
) -> tuple[list[int], int]:
    try:
        rows = tuple(tuple(float(value) for value in row) for row in vectors)
    except (TypeError, ValueError) as exc:
        raise ValueError("vectors must be a rectangular numeric matrix") from exc
    if len(rows) != node_count:
        raise ValueError(f"vectors must contain {node_count} rows")
    dimension = len(rows[0]) if rows else 0
    if dimension == 0 or any(len(row) != dimension for row in rows):
        raise ValueError("vectors must be a non-empty rectangular numeric matrix")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("vectors must contain only finite values")

    def squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
        return sum((a - b) * (a - b) for a, b in zip(left, right))

    partition_count = len(capacities)
    first = random.Random(seed).randrange(node_count)
    selected = {first}
    centroids = [list(rows[first])]
    minimum_distance = [squared_distance(row, rows[first]) for row in rows]
    for _partition in range(1, partition_count):
        farthest = max(
            (node for node in range(node_count) if node not in selected),
            key=lambda node: (minimum_distance[node], -node),
        )
        selected.add(farthest)
        centroids.append(list(rows[farthest]))
        for node, row in enumerate(rows):
            minimum_distance[node] = min(
                minimum_distance[node], squared_distance(row, rows[farthest])
            )

    owner = [-1] * node_count
    previous_owner: list[int] | None = None
    for _iteration in range(iterations):
        distances = [
            [squared_distance(row, centroid) for centroid in centroids]
            for row in rows
        ]
        preferences = [
            sorted(range(partition_count), key=lambda part: (row[part], part))
            for row in distances
        ]
        if partition_count == 1:
            node_order = list(range(node_count))
        else:
            node_order = sorted(
                range(node_count),
                key=lambda node: (
                    -(
                        distances[node][preferences[node][1]]
                        - distances[node][preferences[node][0]]
                    ),
                    distances[node][preferences[node][0]],
                    node,
                ),
            )
        sizes = [0] * partition_count
        for node in node_order:
            for partition in preferences[node]:
                if sizes[partition] < capacities[partition]:
                    owner[node] = partition
                    sizes[partition] += 1
                    break
        if previous_owner == owner:
            break
        previous_owner = owner.copy()
        sums = [[0.0] * dimension for _ in capacities]
        for node, partition in enumerate(owner):
            for coordinate, value in enumerate(rows[node]):
                sums[partition][coordinate] += value
        centroids = [
            [value / capacities[partition] for value in sums[partition]]
            for partition in range(partition_count)
        ]
    return owner, 0


def _ldg0(
    graph: tuple[tuple[int, ...], ...],
    capacities: tuple[int, ...],
    entry_point: int,
) -> tuple[list[int], int]:
    order, edge_visits = _bfs_order(graph, entry_point)
    owner = [-1] * len(graph)
    sizes = [0] * len(capacities)
    for node in order:
        neighbour_counts: dict[int, int] = {}
        for neighbour in graph[node]:
            edge_visits += 1
            partition = owner[neighbour]
            if partition >= 0:
                neighbour_counts[partition] = neighbour_counts.get(partition, 0) + 1
        candidates = [
            partition
            for partition, capacity in enumerate(capacities)
            if sizes[partition] < capacity
        ]
        selected = max(
            candidates,
            key=lambda partition: (
                neighbour_counts.get(partition, 0)
                * (capacities[partition] - sizes[partition])
                / capacities[partition],
                neighbour_counts.get(partition, 0),
                (capacities[partition] - sizes[partition]) / capacities[partition],
                -partition,
            ),
        )
        owner[node] = selected
        sizes[selected] += 1
    return owner, edge_visits


def _ldg_swap1(
    graph: tuple[tuple[int, ...], ...],
    capacities: tuple[int, ...],
    entry_point: int,
) -> tuple[list[int], int]:
    owner, edge_visits = _ldg0(graph, capacities, entry_point)
    neighbour_partition_counts: list[dict[int, int]] = []
    cross_edges: list[tuple[int, int]] = []
    for node, row in enumerate(graph):
        counts: dict[int, int] = {}
        for neighbour in row:
            edge_visits += 1
            partition = owner[neighbour]
            counts[partition] = counts.get(partition, 0) + 1
            if node < neighbour and owner[node] != partition:
                cross_edges.append((node, neighbour))
        neighbour_partition_counts.append(counts)

    candidates: list[tuple[int, int, int]] = []
    for left, right in cross_edges:
        left_partition = owner[left]
        right_partition = owner[right]
        gain = (
            neighbour_partition_counts[left].get(right_partition, 0)
            - 1
            + neighbour_partition_counts[right].get(left_partition, 0)
            - 1
            - neighbour_partition_counts[left].get(left_partition, 0)
            - neighbour_partition_counts[right].get(right_partition, 0)
        )
        if gain > 0:
            candidates.append((-gain, left, right))
    candidates.sort()

    blocked = bytearray(len(graph))
    for _negative_gain, left, right in candidates:
        if blocked[left] or blocked[right]:
            continue
        owner[left], owner[right] = owner[right], owner[left]
        blocked[left] = 1
        blocked[right] = 1
        for neighbour in graph[left]:
            edge_visits += 1
            blocked[neighbour] = 1
        for neighbour in graph[right]:
            edge_visits += 1
            blocked[neighbour] = 1
    return owner, edge_visits


def _fennel(
    graph: tuple[tuple[int, ...], ...],
    capacities: tuple[int, ...],
    entry_point: int,
) -> tuple[list[int], int]:
    order, edge_visits = _bfs_order(graph, entry_point)
    node_count = len(graph)
    partition_count = len(capacities)
    edge_count = sum(len(row) for row in graph) / 2.0
    alpha = (
        edge_count * (partition_count ** (FENNEL_GAMMA - 1.0))
        / (node_count**FENNEL_GAMMA)
    )
    owner = [-1] * node_count
    sizes = [0] * partition_count
    for node in order:
        neighbour_counts: dict[int, int] = {}
        for neighbour in graph[node]:
            edge_visits += 1
            partition = owner[neighbour]
            if partition >= 0:
                neighbour_counts[partition] = neighbour_counts.get(partition, 0) + 1
        candidates = [
            partition
            for partition, capacity in enumerate(capacities)
            if sizes[partition] < capacity
        ]
        selected = max(
            candidates,
            key=lambda partition: (
                neighbour_counts.get(partition, 0)
                - alpha
                * (
                    (sizes[partition] + 1) ** FENNEL_GAMMA
                    - sizes[partition] ** FENNEL_GAMMA
                ),
                neighbour_counts.get(partition, 0),
                (capacities[partition] - sizes[partition]) / capacities[partition],
                -partition,
            ),
        )
        owner[node] = selected
        sizes[selected] += 1
    return owner, edge_visits


def _balanced_region_grow(
    graph: tuple[tuple[int, ...], ...],
    capacities: tuple[int, ...],
    entry_point: int,
) -> tuple[list[int], int]:
    bfs_order, edge_visits = _bfs_order(graph, entry_point)
    node_count = len(graph)
    partition_count = len(capacities)
    seed_candidates = [entry_point]
    seed_candidates.extend(
        bfs_order[(partition * node_count) // partition_count]
        for partition in range(1, partition_count)
    )
    seed_candidates.extend(bfs_order)
    seeds: list[int] = []
    seen_seeds: set[int] = set()
    for node in seed_candidates:
        if node not in seen_seeds:
            seen_seeds.add(node)
            seeds.append(node)
            if len(seeds) == partition_count:
                break

    owner = [-1] * node_count
    sizes = [0] * partition_count
    frontiers: list[list[tuple[int, int]]] = [[] for _ in capacities]
    queued: list[set[int]] = [set() for _ in capacities]

    def expose(partition: int, node: int, distance: int) -> None:
        nonlocal edge_visits
        for neighbour in graph[node]:
            edge_visits += 1
            if owner[neighbour] < 0 and neighbour not in queued[partition]:
                queued[partition].add(neighbour)
                heapq.heappush(frontiers[partition], (distance + 1, neighbour))

    for partition, node in enumerate(seeds):
        owner[node] = partition
        sizes[partition] = 1
    for partition, node in enumerate(seeds):
        expose(partition, node, 0)

    unassigned_count = node_count - partition_count
    fallback_offset = 0
    while unassigned_count:
        active: list[int] = []
        for partition, frontier in enumerate(frontiers):
            if sizes[partition] >= capacities[partition]:
                continue
            while frontier and owner[frontier[0][1]] >= 0:
                heapq.heappop(frontier)
            if frontier:
                active.append(partition)

        if active:
            selected_partition = min(
                active,
                key=lambda partition: (
                    sizes[partition] / capacities[partition],
                    partition,
                ),
            )
            distance, node = heapq.heappop(frontiers[selected_partition])
        else:
            while owner[bfs_order[fallback_offset]] >= 0:
                fallback_offset += 1
            node = bfs_order[fallback_offset]
            selected_partition = min(
                (
                    partition
                    for partition, capacity in enumerate(capacities)
                    if sizes[partition] < capacity
                ),
                key=lambda partition: (
                    sizes[partition] / capacities[partition],
                    partition,
                ),
            )
            distance = 0

        if owner[node] >= 0:
            continue
        owner[node] = selected_partition
        sizes[selected_partition] += 1
        unassigned_count -= 1
        expose(selected_partition, node, distance)
    return owner, edge_visits
