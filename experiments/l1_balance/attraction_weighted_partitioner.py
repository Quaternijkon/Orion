"""Attraction-weighted, search-induced graph labeling for frozen Orion L1.

This module never builds or mutates HNSW.  It consumes:

* the immutable production upper graph adjacency;
* calibration hits produced by the production upper navigator; and
* a requested logical-shard count.

The first hit estimates the amount of L0 data attracted by each upper node.
Co-occurrence among returned upper hits weights only edges that already exist
in the frozen level-0 HNSW graph.  The resulting vertex- and edge-weighted graph
can be handed to METIS/KaHIP without changing any online Orion code.

The public core deliberately has no observed shard-load, query, ground-truth,
or lower-HNSW input.  A fixed calibration sample is distribution evidence, not
feedback from a partially populated deployment.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
SAMPLE_ESTIMATOR = "sample-top1-dirichlet-v1"
EXACT_ESTIMATOR = "exact-top1-count-v1"
SUPPORTED_ESTIMATORS = (SAMPLE_ESTIMATOR, EXACT_ESTIMATOR)
UNIT_EDGE_MODE = "unit"
HIT_COOCCURRENCE_EDGE_MODE = "hit-cooccurrence"
TOP1_STAR_EDGE_MODE = "top1-star"
SUPPORTED_EDGE_MODES = (
    UNIT_EDGE_MODE,
    HIT_COOCCURRENCE_EDGE_MODE,
    TOP1_STAR_EDGE_MODE,
)
DIRICHLET_ALPHA = 1
DEFAULT_IMBALANCE_TOLERANCE = 0.02
DEFAULT_HEAVY_ATOM_FRACTION = 0.20
REFINEMENT_MAX_CUT_NUMERATOR = 103
REFINEMENT_MAX_CUT_DENOMINATOR = 100
REFINEMENT_TARGET_CEILING_NUMERATOR = 23
REFINEMENT_TARGET_CEILING_DENOMINATOR = 20
REFINEMENT_MAX_ROUNDS = 12
NAV_REFINEMENT_MAX_VOTE_LOSS = 1
NAV_REFINEMENT_BUDGET_NUMERATOR = 2
NAV_REFINEMENT_BUDGET_DENOMINATOR = 25


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _fraction(value: Any, name: str, *, upper_inclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    upper_ok = parsed <= 1.0 if upper_inclusive else parsed < 1.0
    if not math.isfinite(parsed) or parsed <= 0.0 or not upper_ok:
        boundary = "(0, 1]" if upper_inclusive else "(0, 1)"
        raise ValueError(f"{name} must lie in {boundary}")
    return parsed


def normalise_undirected_graph(
    adjacency: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Return the deterministic undirected union of level-0 neighbour lists."""

    if isinstance(adjacency, (str, bytes)):
        raise TypeError("adjacency must be a sequence of neighbour sequences")
    try:
        node_count = len(adjacency)
    except TypeError as exc:
        raise TypeError("adjacency must have a stable length") from exc
    if node_count == 0:
        raise ValueError("adjacency must contain at least one upper node")

    graph: list[set[int]] = [set() for _ in range(node_count)]
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
            graph[node].add(neighbour)
            graph[neighbour].add(node)
    return tuple(tuple(sorted(row)) for row in graph)


@dataclass(frozen=True)
class CalibrationMeasurements:
    """Top-1 attraction counts plus frozen-graph search co-occurrence counts."""

    row_count: int
    top_k: int
    first_hit_counts: tuple[int, ...]
    edge_mode: str
    edge_cooccurrence: tuple[tuple[int, int, int], ...]
    semantic_sha256: str


def measure_calibration_rows(
    adjacency: Sequence[Sequence[int]],
    rows: Iterable[Sequence[int]],
    *,
    edge_mode: str = HIT_COOCCURRENCE_EDGE_MODE,
) -> CalibrationMeasurements:
    """Measure attraction and search-induced edge support in one streaming pass.

    Hit IDs must already be local upper-node indices.  Co-occurrence never adds
    an edge: it only increments an existing immutable HNSW level-0 edge when
    both endpoints occur in the same calibration result row.
    """

    if edge_mode not in SUPPORTED_EDGE_MODES:
        choices = ", ".join(SUPPORTED_EDGE_MODES)
        raise ValueError(f"unknown edge_mode {edge_mode!r}; expected one of {choices}")
    graph = normalise_undirected_graph(adjacency)
    node_count = len(graph)
    graph_edges = {
        (left, right)
        for left, neighbours in enumerate(graph)
        for right in neighbours
        if left < right
    }
    first_hit_counts = [0] * node_count
    edge_counts: dict[tuple[int, int], int] = {}
    row_count = 0
    top_k: int | None = None

    for row_index, raw_row in enumerate(rows):
        if isinstance(raw_row, (str, bytes)):
            raise TypeError(f"rows[{row_index}] must be a hit sequence")
        try:
            row = tuple(raw_row)
        except TypeError as exc:
            raise TypeError(f"rows[{row_index}] must be iterable") from exc
        if not row:
            raise ValueError(f"rows[{row_index}] must contain at least one hit")
        if top_k is None:
            top_k = len(row)
        elif len(row) != top_k:
            raise ValueError(
                f"rows[{row_index}] has width {len(row)}; expected {top_k}"
            )

        local: list[int] = []
        seen: set[int] = set()
        for raw_node in row:
            if isinstance(raw_node, bool) or not isinstance(raw_node, Integral):
                raise TypeError(
                    f"rows[{row_index}] contains non-integer node {raw_node!r}"
                )
            node = int(raw_node)
            if node < 0 or node >= node_count:
                raise ValueError(
                    f"rows[{row_index}] contains out-of-range node {node}"
                )
            if node in seen:
                raise ValueError(f"rows[{row_index}] repeats upper node {node}")
            seen.add(node)
            local.append(node)

        first_hit_counts[local[0]] += 1
        if edge_mode == HIT_COOCCURRENCE_EDGE_MODE and len(local) > 1:
            for left_index, raw_left in enumerate(local):
                for raw_right in local[left_index + 1 :]:
                    key = (
                        (raw_left, raw_right)
                        if raw_left < raw_right
                        else (raw_right, raw_left)
                    )
                    if key in graph_edges:
                        edge_counts[key] = edge_counts.get(key, 0) + 1
        elif edge_mode == TOP1_STAR_EDGE_MODE and len(local) > 1:
            first = local[0]
            for neighbour in local[1:]:
                key = (
                    (first, neighbour)
                    if first < neighbour
                    else (neighbour, first)
                )
                edge_counts[key] = edge_counts.get(key, 0) + 1
        row_count += 1

    if row_count == 0 or top_k is None:
        raise ValueError("calibration rows must not be empty")
    frozen_edges = tuple(
        (left, right, count)
        for (left, right), count in sorted(edge_counts.items())
    )
    semantic = {
        "format_version": FORMAT_VERSION,
        "row_count": row_count,
        "top_k": top_k,
        "first_hit_counts": first_hit_counts,
        "edge_mode": edge_mode,
        "edge_cooccurrence": frozen_edges,
    }
    return CalibrationMeasurements(
        row_count=row_count,
        top_k=top_k,
        first_hit_counts=tuple(first_hit_counts),
        edge_mode=edge_mode,
        edge_cooccurrence=frozen_edges,
        semantic_sha256=_canonical_sha256(semantic),
    )


@dataclass(frozen=True)
class AttractionWeights:
    """Integer vertex weights whose sum equals the logical point count."""

    estimator: str
    values: tuple[int, ...]
    calibration_count: int
    logical_point_count: int
    total_weight: int
    max_weight: int
    mean_weight: float
    semantic_sha256: str


def estimate_attraction_weights(
    first_hit_counts: Sequence[int],
    *,
    calibration_count: int,
    logical_point_count: int,
    estimator: str = SAMPLE_ESTIMATOR,
) -> AttractionWeights:
    """Estimate full-corpus top-1 attraction with deterministic apportionment.

    ``sample-top1-dirichlet-v1`` reserves one point for every upper node, then
    apportions the remaining corpus using a fixed Dirichlet(1) prior.  This
    prevents zero-weight upper vertices when a small calibration sample is used.

    ``exact-top1-count-v1`` requires one calibration row per logical point and
    uses the observed counts byte-for-byte as the vertex weights.
    """

    if estimator not in SUPPORTED_ESTIMATORS:
        choices = ", ".join(SUPPORTED_ESTIMATORS)
        raise ValueError(f"unknown estimator {estimator!r}; expected one of {choices}")
    calibration = _positive_integer(calibration_count, "calibration_count")
    logical = _positive_integer(logical_point_count, "logical_point_count")
    if isinstance(first_hit_counts, (str, bytes)):
        raise TypeError("first_hit_counts must be a sequence")
    try:
        raw_counts = tuple(first_hit_counts)
    except TypeError as exc:
        raise TypeError("first_hit_counts must be iterable") from exc
    if not raw_counts:
        raise ValueError("first_hit_counts must not be empty")

    counts: list[int] = []
    for node, raw_count in enumerate(raw_counts):
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
            raise TypeError(f"first_hit_counts[{node}] must be an integer")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"first_hit_counts[{node}] must be non-negative")
        counts.append(count)
    if sum(counts) != calibration:
        raise ValueError(
            "first-hit counts do not sum to calibration_count: "
            f"{sum(counts)} != {calibration}"
        )

    node_count = len(counts)
    if logical < node_count:
        raise ValueError(
            "logical_point_count must be at least the upper-node count"
        )
    if estimator == EXACT_ESTIMATOR:
        if calibration != logical:
            raise ValueError(
                "exact estimator requires calibration_count == logical_point_count"
            )
        values = counts
    else:
        residual = logical - node_count
        denominator = calibration + DIRICHLET_ALPHA * node_count
        floors: list[int] = []
        remainders: list[int] = []
        for count in counts:
            numerator = residual * (count + DIRICHLET_ALPHA)
            quotient, remainder = divmod(numerator, denominator)
            floors.append(1 + quotient)
            remainders.append(remainder)
        remaining = logical - sum(floors)
        if remaining < 0 or remaining > node_count:
            raise AssertionError("largest-remainder apportionment became invalid")
        order = sorted(range(node_count), key=lambda node: (-remainders[node], node))
        for node in order[:remaining]:
            floors[node] += 1
        values = floors

    if sum(values) != logical:
        raise AssertionError("attraction weights do not sum to logical point count")
    if estimator == SAMPLE_ESTIMATOR and min(values) <= 0:
        raise AssertionError("smoothed attraction weights must be positive")
    semantic = {
        "format_version": FORMAT_VERSION,
        "estimator": estimator,
        "dirichlet_alpha": DIRICHLET_ALPHA if estimator == SAMPLE_ESTIMATOR else None,
        "calibration_count": calibration,
        "logical_point_count": logical,
        "first_hit_counts": counts,
        "values": values,
    }
    return AttractionWeights(
        estimator=estimator,
        values=tuple(values),
        calibration_count=calibration,
        logical_point_count=logical,
        total_weight=logical,
        max_weight=max(values),
        mean_weight=logical / node_count,
        semantic_sha256=_canonical_sha256(semantic),
    )


@dataclass(frozen=True)
class SearchWeightedGraph:
    """Undirected frozen graph with positive integer edge weights."""

    adjacency: tuple[tuple[tuple[int, int], ...], ...]
    node_count: int
    edge_count: int
    raw_hnsw_edge_count: int
    search_overlay_edge_count: int
    total_edge_weight: int
    edge_mode: str
    semantic_sha256: str


def build_search_weighted_graph(
    adjacency: Sequence[Sequence[int]],
    measurements: CalibrationMeasurements,
) -> SearchWeightedGraph:
    graph = normalise_undirected_graph(adjacency)
    if len(measurements.first_hit_counts) != len(graph):
        raise ValueError("calibration node count differs from upper graph")
    raw_edges = {
        (left, right): 1
        for left, row in enumerate(graph)
        for right in row
        if left < right
    }
    weights = dict(raw_edges)
    for left, right, count in measurements.edge_cooccurrence:
        key = (left, right)
        if (
            key not in weights
            and measurements.edge_mode != TOP1_STAR_EDGE_MODE
        ):
            raise ValueError("calibration co-occurrence references a non-graph edge")
        if count <= 0:
            raise ValueError("calibration edge co-occurrence must be positive")
        weights[key] = weights.get(key, 1) + count

    rows: list[list[tuple[int, int]]] = [[] for _ in graph]
    for (left, right), weight in sorted(weights.items()):
        rows[left].append((right, weight))
        rows[right].append((left, weight))
    frozen = tuple(tuple(sorted(row)) for row in rows)
    semantic = {
        "format_version": FORMAT_VERSION,
        "edge_mode": measurements.edge_mode,
        "adjacency": frozen,
    }
    return SearchWeightedGraph(
        adjacency=frozen,
        node_count=len(graph),
        edge_count=len(weights),
        raw_hnsw_edge_count=len(raw_edges),
        search_overlay_edge_count=len(weights) - len(raw_edges),
        total_edge_weight=sum(weights.values()),
        edge_mode=measurements.edge_mode,
        semantic_sha256=_canonical_sha256(semantic),
    )


def metis_graph_bytes(
    graph: SearchWeightedGraph,
    vertex_weights: AttractionWeights,
) -> bytes:
    """Encode one vertex-weight and edge-weight constraint in METIS format."""

    if len(vertex_weights.values) != graph.node_count:
        raise ValueError("vertex weight count differs from graph node count")
    lines = [f"{graph.node_count} {graph.edge_count} 011 1"]
    for node, row in enumerate(graph.adjacency):
        fields = [str(vertex_weights.values[node])]
        for neighbour, weight in row:
            if weight <= 0:
                raise ValueError("METIS edge weights must be positive")
            fields.extend((str(neighbour + 1), str(weight)))
        lines.append(" ".join(fields))
    return ("\n".join(lines) + "\n").encode("ascii")


def _raw_cut(
    graph: tuple[tuple[int, ...], ...], owner: Sequence[int]
) -> int:
    return sum(
        owner[left] != owner[right]
        for left, row in enumerate(graph)
        for right in row
        if left < right
    )


@dataclass(frozen=True)
class AttractionRefinementResult:
    owner: tuple[int, ...]
    initial_partition_loads: tuple[int, ...]
    final_partition_loads: tuple[int, ...]
    initial_cut_edges: int
    final_cut_edges: int
    cut_limit: int
    moved_node_count: int
    moved_attraction_weight: int
    rounds: tuple[dict[str, Any], ...]
    semantic_sha256: str


def refine_owner_by_attraction(
    adjacency: Sequence[Sequence[int]],
    initial_owner: Sequence[int],
    vertex_weights: AttractionWeights,
    num_partitions: int,
) -> AttractionRefinementResult:
    """Balance attraction by deterministic boundary moves under a cut cap.

    The immutable HNSW graph supplies the only target-owner evidence.  Each
    upper node moves at most once.  A move must reduce the exact two-shard
    squared-load potential, keep the target below 1.15x mean, and keep global
    raw level-0 edge cut within 1.03x of the initial owner.
    """

    graph = normalise_undirected_graph(adjacency)
    partitions = _positive_integer(num_partitions, "num_partitions")
    if len(initial_owner) != len(graph):
        raise ValueError("initial owner length differs from upper graph")
    if len(vertex_weights.values) != len(graph):
        raise ValueError("attraction weight length differs from upper graph")
    owner: list[int] = []
    loads = [0] * partitions
    for node, raw_partition in enumerate(initial_owner):
        if isinstance(raw_partition, bool) or not isinstance(raw_partition, Integral):
            raise TypeError(f"initial_owner[{node}] must be an integer")
        partition = int(raw_partition)
        if partition < 0 or partition >= partitions:
            raise ValueError(
                f"initial_owner[{node}] has invalid partition {partition}"
            )
        owner.append(partition)
        loads[partition] += vertex_weights.values[node]
    if any(load == 0 for load in loads):
        raise ValueError("initial owner contains an empty attraction shard")

    initial_loads = tuple(loads)
    total_weight = vertex_weights.total_weight
    initial_cut = _raw_cut(graph, owner)
    cut_limit = (
        initial_cut * REFINEMENT_MAX_CUT_NUMERATOR
    ) // REFINEMENT_MAX_CUT_DENOMINATOR
    current_cut = initial_cut
    moved = bytearray(len(owner))
    moved_node_count = 0
    moved_weight = 0
    round_records: list[dict[str, Any]] = []

    for round_index in range(REFINEMENT_MAX_ROUNDS):
        proposals: list[tuple[Any, ...]] = []
        for node, source in enumerate(owner):
            if moved[node] or loads[source] * partitions <= total_weight:
                continue
            weight = vertex_weights.values[node]
            if weight <= 0:
                continue
            neighbour_owner_counts: dict[int, int] = {}
            for neighbour in graph[node]:
                target = owner[neighbour]
                neighbour_owner_counts[target] = (
                    neighbour_owner_counts.get(target, 0) + 1
                )
            source_degree = neighbour_owner_counts.get(source, 0)
            for target, target_degree in neighbour_owner_counts.items():
                if target == source:
                    continue
                if (
                    (loads[target] + weight)
                    * REFINEMENT_TARGET_CEILING_DENOMINATOR
                    * partitions
                    > REFINEMENT_TARGET_CEILING_NUMERATOR * total_weight
                ):
                    continue
                cut_delta = source_degree - target_degree
                if current_cut + cut_delta > cut_limit:
                    continue
                load_gap_after_move = loads[source] - loads[target] - weight
                if load_gap_after_move <= 0:
                    continue
                potential_improvement = 2 * weight * load_gap_after_move
                proposals.append(
                    (
                        max(cut_delta, 0) / potential_improvement,
                        cut_delta,
                        -potential_improvement / weight,
                        loads[target],
                        -weight,
                        node,
                        source,
                        target,
                    )
                )
        proposals.sort()

        accepted = 0
        accepted_weight = 0
        for (
            _cut_cost,
            _proposed_delta,
            _load_gain,
            _target_load,
            _negative_weight,
            node,
            proposed_source,
            target,
        ) in proposals:
            if moved[node] or owner[node] != proposed_source:
                continue
            source = owner[node]
            weight = vertex_weights.values[node]
            if weight <= 0:
                continue
            if (
                (loads[target] + weight)
                * REFINEMENT_TARGET_CEILING_DENOMINATOR
                * partitions
                > REFINEMENT_TARGET_CEILING_NUMERATOR * total_weight
            ):
                continue
            neighbour_owner_counts: dict[int, int] = {}
            for neighbour in graph[node]:
                partition = owner[neighbour]
                neighbour_owner_counts[partition] = (
                    neighbour_owner_counts.get(partition, 0) + 1
                )
            cut_delta = neighbour_owner_counts.get(
                source, 0
            ) - neighbour_owner_counts.get(target, 0)
            if current_cut + cut_delta > cut_limit:
                continue
            load_gap_after_move = loads[source] - loads[target] - weight
            if load_gap_after_move <= 0:
                continue
            owner[node] = target
            loads[source] -= weight
            loads[target] += weight
            current_cut += cut_delta
            moved[node] = 1
            moved_node_count += 1
            moved_weight += weight
            accepted += 1
            accepted_weight += weight

        round_records.append(
            {
                "round": round_index + 1,
                "proposal_count": len(proposals),
                "accepted_move_count": accepted,
                "accepted_attraction_weight": accepted_weight,
                "cut_edges": current_cut,
                "partition_load_min": min(loads),
                "partition_load_max": max(loads),
            }
        )
        if accepted == 0:
            break

    semantic = {
        "format_version": FORMAT_VERSION,
        "method": "attraction_boundary_refinement_v1",
        "owner": owner,
        "initial_partition_loads": initial_loads,
        "final_partition_loads": loads,
        "initial_cut_edges": initial_cut,
        "final_cut_edges": current_cut,
        "cut_limit": cut_limit,
        "moved_node_count": moved_node_count,
        "moved_attraction_weight": moved_weight,
        "rounds": round_records,
    }
    return AttractionRefinementResult(
        owner=tuple(owner),
        initial_partition_loads=initial_loads,
        final_partition_loads=tuple(loads),
        initial_cut_edges=initial_cut,
        final_cut_edges=current_cut,
        cut_limit=cut_limit,
        moved_node_count=moved_node_count,
        moved_attraction_weight=moved_weight,
        rounds=tuple(round_records),
        semantic_sha256=_canonical_sha256(semantic),
    )


def refine_owner_by_attraction_navigation(
    adjacency: Sequence[Sequence[int]],
    navigation_rows: Sequence[Sequence[int]],
    initial_owner: Sequence[int],
    vertex_weights: AttractionWeights,
    num_partitions: int,
) -> AttractionRefinementResult:
    """Run the fixed 8% attraction repair with upper self-navigation votes.

    This post-exploratory candidate retains the raw cut and target-capacity
    contracts from :func:`refine_owner_by_attraction`, but it permits at most
    one vote of self-navigation loss and moves at most 8% of total attraction
    weight.  Targets must be visible either in the node's raw HNSW neighbours
    or in its own frozen self-navigation row.
    """

    graph = normalise_undirected_graph(adjacency)
    partitions = _positive_integer(num_partitions, "num_partitions")
    if len(initial_owner) != len(graph):
        raise ValueError("initial owner length differs from upper graph")
    if len(navigation_rows) != len(graph):
        raise ValueError("self-navigation row count differs from upper graph")
    if len(vertex_weights.values) != len(graph):
        raise ValueError("attraction weight length differs from upper graph")

    frozen_rows: list[tuple[int, ...]] = []
    for row_index, raw_row in enumerate(navigation_rows):
        row: list[int] = []
        seen: set[int] = set()
        for raw_node in raw_row:
            if isinstance(raw_node, bool) or not isinstance(raw_node, Integral):
                raise TypeError(
                    f"navigation_rows[{row_index}] contains a non-integer node"
                )
            node = int(raw_node)
            if node < 0 or node >= len(graph):
                raise ValueError(
                    f"navigation_rows[{row_index}] contains out-of-range node {node}"
                )
            if node in seen:
                raise ValueError(
                    f"navigation_rows[{row_index}] repeats upper node {node}"
                )
            seen.add(node)
            row.append(node)
        if not row:
            raise ValueError(f"navigation_rows[{row_index}] is empty")
        frozen_rows.append(tuple(row))

    owner: list[int] = []
    loads = [0] * partitions
    for node, raw_partition in enumerate(initial_owner):
        if isinstance(raw_partition, bool) or not isinstance(raw_partition, Integral):
            raise TypeError(f"initial_owner[{node}] must be an integer")
        partition = int(raw_partition)
        if partition < 0 or partition >= partitions:
            raise ValueError(
                f"initial_owner[{node}] has invalid partition {partition}"
            )
        owner.append(partition)
        loads[partition] += vertex_weights.values[node]
    if any(load == 0 for load in loads):
        raise ValueError("initial owner contains an empty attraction shard")

    initial_loads = tuple(loads)
    total_weight = vertex_weights.total_weight
    move_budget = (
        total_weight * NAV_REFINEMENT_BUDGET_NUMERATOR
    ) // NAV_REFINEMENT_BUDGET_DENOMINATOR
    initial_cut = _raw_cut(graph, owner)
    cut_limit = (
        initial_cut * REFINEMENT_MAX_CUT_NUMERATOR
    ) // REFINEMENT_MAX_CUT_DENOMINATOR
    current_cut = initial_cut
    moved = bytearray(len(owner))
    moved_node_count = 0
    moved_weight = 0
    round_records: list[dict[str, Any]] = []

    for round_index in range(REFINEMENT_MAX_ROUNDS):
        proposals: list[tuple[Any, ...]] = []
        for node, source in enumerate(owner):
            if moved[node] or loads[source] * partitions <= total_weight:
                continue
            weight = vertex_weights.values[node]
            if weight <= 0:
                continue
            if moved_weight + weight > move_budget:
                continue
            vote_counts: dict[int, int] = {}
            for hit in frozen_rows[node]:
                partition = owner[hit]
                vote_counts[partition] = vote_counts.get(partition, 0) + 1
            source_vote = vote_counts.get(source, 0)
            neighbour_owner_counts: dict[int, int] = {}
            for neighbour in graph[node]:
                partition = owner[neighbour]
                neighbour_owner_counts[partition] = (
                    neighbour_owner_counts.get(partition, 0) + 1
                )
            source_degree = neighbour_owner_counts.get(source, 0)
            targets = set(vote_counts) | set(neighbour_owner_counts)
            targets.discard(source)
            for target in targets:
                navigation_loss = source_vote - vote_counts.get(target, 0)
                if navigation_loss > NAV_REFINEMENT_MAX_VOTE_LOSS:
                    continue
                if (
                    (loads[target] + weight)
                    * REFINEMENT_TARGET_CEILING_DENOMINATOR
                    * partitions
                    > REFINEMENT_TARGET_CEILING_NUMERATOR * total_weight
                ):
                    continue
                cut_delta = source_degree - neighbour_owner_counts.get(target, 0)
                if current_cut + cut_delta > cut_limit:
                    continue
                load_gap_after_move = loads[source] - loads[target] - weight
                if load_gap_after_move <= 0:
                    continue
                potential_improvement = 2 * weight * load_gap_after_move
                proposals.append(
                    (
                        max(navigation_loss, 0),
                        max(cut_delta, 0) / potential_improvement,
                        cut_delta,
                        -potential_improvement / weight,
                        loads[target],
                        -weight,
                        node,
                        source,
                        target,
                    )
                )
        proposals.sort()

        accepted = 0
        accepted_weight = 0
        maximum_vote_loss = 0
        for proposal in proposals:
            node, proposed_source, target = proposal[-3:]
            if moved[node] or owner[node] != proposed_source:
                continue
            source = owner[node]
            weight = vertex_weights.values[node]
            if weight <= 0:
                continue
            if moved_weight + weight > move_budget:
                continue
            vote_counts: dict[int, int] = {}
            for hit in frozen_rows[node]:
                partition = owner[hit]
                vote_counts[partition] = vote_counts.get(partition, 0) + 1
            navigation_loss = vote_counts.get(source, 0) - vote_counts.get(target, 0)
            if navigation_loss > NAV_REFINEMENT_MAX_VOTE_LOSS:
                continue
            if (
                (loads[target] + weight)
                * REFINEMENT_TARGET_CEILING_DENOMINATOR
                * partitions
                > REFINEMENT_TARGET_CEILING_NUMERATOR * total_weight
            ):
                continue
            neighbour_owner_counts: dict[int, int] = {}
            for neighbour in graph[node]:
                partition = owner[neighbour]
                neighbour_owner_counts[partition] = (
                    neighbour_owner_counts.get(partition, 0) + 1
                )
            cut_delta = neighbour_owner_counts.get(
                source, 0
            ) - neighbour_owner_counts.get(target, 0)
            if current_cut + cut_delta > cut_limit:
                continue
            load_gap_after_move = loads[source] - loads[target] - weight
            if load_gap_after_move <= 0:
                continue
            owner[node] = target
            loads[source] -= weight
            loads[target] += weight
            current_cut += cut_delta
            moved[node] = 1
            moved_node_count += 1
            moved_weight += weight
            accepted += 1
            accepted_weight += weight
            maximum_vote_loss = max(maximum_vote_loss, navigation_loss)

        round_records.append(
            {
                "round": round_index + 1,
                "proposal_count": len(proposals),
                "accepted_move_count": accepted,
                "accepted_attraction_weight": accepted_weight,
                "cumulative_moved_attraction_weight": moved_weight,
                "maximum_accepted_navigation_vote_loss": maximum_vote_loss,
                "cut_edges": current_cut,
                "partition_load_min": min(loads),
                "partition_load_max": max(loads),
            }
        )
        if accepted == 0 or moved_weight >= move_budget:
            break

    semantic = {
        "format_version": FORMAT_VERSION,
        "method": "attraction_navigation_refinement_v1",
        "owner": owner,
        "initial_partition_loads": initial_loads,
        "final_partition_loads": loads,
        "initial_cut_edges": initial_cut,
        "final_cut_edges": current_cut,
        "cut_limit": cut_limit,
        "move_budget": move_budget,
        "maximum_navigation_vote_loss": NAV_REFINEMENT_MAX_VOTE_LOSS,
        "moved_node_count": moved_node_count,
        "moved_attraction_weight": moved_weight,
        "rounds": round_records,
    }
    return AttractionRefinementResult(
        owner=tuple(owner),
        initial_partition_loads=initial_loads,
        final_partition_loads=tuple(loads),
        initial_cut_edges=initial_cut,
        final_cut_edges=current_cut,
        cut_limit=cut_limit,
        moved_node_count=moved_node_count,
        moved_attraction_weight=moved_weight,
        rounds=tuple(round_records),
        semantic_sha256=_canonical_sha256(semantic),
    )


def write_metis_graph(
    path: Path,
    graph: SearchWeightedGraph,
    vertex_weights: AttractionWeights,
) -> str:
    data = metis_graph_bytes(graph, vertex_weights)
    with path.open("xb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class PartitionValidation:
    partition_sizes: tuple[int, ...]
    partition_loads: tuple[int, ...]
    load_mean: float
    load_min_over_mean: float
    load_max_over_mean: float
    heavy_atom_ratio: float
    unweighted_cut_edges: int
    weighted_cut: int
    balance_pass: bool
    heavy_atom_pass: bool
    semantic_sha256: str


def validate_partition(
    owner: Sequence[int],
    vertex_weights: AttractionWeights,
    graph: SearchWeightedGraph,
    num_partitions: int,
    *,
    imbalance_tolerance: float = DEFAULT_IMBALANCE_TOLERANCE,
    heavy_atom_fraction: float = DEFAULT_HEAVY_ATOM_FRACTION,
) -> PartitionValidation:
    partitions = _positive_integer(num_partitions, "num_partitions")
    tolerance = _fraction(
        imbalance_tolerance, "imbalance_tolerance", upper_inclusive=False
    )
    atom_limit = _fraction(
        heavy_atom_fraction, "heavy_atom_fraction", upper_inclusive=True
    )
    if len(owner) != graph.node_count or len(owner) != len(vertex_weights.values):
        raise ValueError("owner, graph, and vertex weight lengths differ")

    sizes = [0] * partitions
    loads = [0] * partitions
    frozen_owner: list[int] = []
    for node, raw_partition in enumerate(owner):
        if isinstance(raw_partition, bool) or not isinstance(raw_partition, Integral):
            raise TypeError(f"owner[{node}] must be an integer")
        partition = int(raw_partition)
        if partition < 0 or partition >= partitions:
            raise ValueError(f"owner[{node}] has invalid partition {partition}")
        frozen_owner.append(partition)
        sizes[partition] += 1
        loads[partition] += vertex_weights.values[node]
    if any(size == 0 for size in sizes):
        raise ValueError(f"partitioner produced an empty shard: {sizes}")

    mean = vertex_weights.total_weight / partitions
    min_ratio = min(loads) / mean
    max_ratio = max(loads) / mean
    heavy_ratio = vertex_weights.max_weight / mean
    cut_edges = 0
    weighted_cut = 0
    for left, row in enumerate(graph.adjacency):
        for right, weight in row:
            if left < right and frozen_owner[left] != frozen_owner[right]:
                cut_edges += 1
                weighted_cut += weight
    balance_pass = (
        min_ratio + 1e-12 >= 1.0 - tolerance
        and max_ratio <= 1.0 + tolerance + 1e-12
    )
    heavy_pass = heavy_ratio <= atom_limit + 1e-12
    semantic = {
        "format_version": FORMAT_VERSION,
        "owner": frozen_owner,
        "partition_sizes": sizes,
        "partition_loads": loads,
        "imbalance_tolerance": tolerance,
        "heavy_atom_fraction": atom_limit,
        "unweighted_cut_edges": cut_edges,
        "weighted_cut": weighted_cut,
    }
    return PartitionValidation(
        partition_sizes=tuple(sizes),
        partition_loads=tuple(loads),
        load_mean=mean,
        load_min_over_mean=min_ratio,
        load_max_over_mean=max_ratio,
        heavy_atom_ratio=heavy_ratio,
        unweighted_cut_edges=cut_edges,
        weighted_cut=weighted_cut,
        balance_pass=balance_pass,
        heavy_atom_pass=heavy_pass,
        semantic_sha256=_canonical_sha256(semantic),
    )
