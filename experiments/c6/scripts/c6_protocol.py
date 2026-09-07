#!/usr/bin/env python3
"""Core policy, oracle, statistics, and persistence helpers for C6.

The module is deliberately independent of the live cluster.  A capture runner records
one frozen routing trace and a query/shard/efSearch result cube; all deployable policies
and offline oracles are then evaluated against those immutable inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = 1
TOP_K = 10
TARGET_RECALL = 0.90
TUNING_QUERY_COUNT = 1_000
LOGICAL_SHARD_COUNTS = (4, 8, 16, 32)
EF_GRID = (8, 16, 32, 64, 128, 256, 512)
POLICIES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6")


PointId = int | str


@dataclass(frozen=True)
class ShardEvidence:
    shard_id: int
    candidate_count: int
    best_candidate_rank: int
    minimum_candidate_distance: float
    mean_candidate_distance: float
    entry_points: tuple[PointId, ...]

    @property
    def entry_point_count(self) -> int:
        return len(self.entry_points)


@dataclass(frozen=True)
class RouteQuery:
    query_id: int
    navigation_candidate_count: int
    candidate_shard_count: int
    ranked_candidate_shards: tuple[int, ...]
    adaptive_selected_shards: tuple[int, ...]
    evidence: tuple[ShardEvidence, ...]
    logical_shards: int | None = None

    def evidence_by_shard(self) -> dict[int, ShardEvidence]:
        return {row.shard_id: row for row in self.evidence}

    def fixed_policy_shard_order(self) -> tuple[int, ...]:
        shard_count = self.logical_shards
        if shard_count is None:
            shard_count = max(self.ranked_candidate_shards, default=-1) + 1
        represented = set(self.ranked_candidate_shards)
        return self.ranked_candidate_shards + tuple(
            shard_id for shard_id in range(shard_count) if shard_id not in represented
        )


@dataclass(frozen=True)
class LocalSearchTrace:
    query_id: int
    shard_id: int
    ef_search: int
    result_ids: tuple[PointId, ...]
    result_scores: tuple[float, ...]
    distance_computations: int
    nodes_visited: int
    worker_cpu_time_us: int
    worker_cpu_wall_time_us: int
    latency_us: float
    response_bytes: int = 0


@dataclass(frozen=True)
class PolicyConfig:
    policy: str
    fixed_p: int | None = None
    uniform_ef_search: int | None = None
    alpha: int | None = None
    beta: int | None = None
    ef_min: int = min(EF_GRID)
    ef_max: int = max(EF_GRID)

    def validate(self, logical_shards: int) -> None:
        if self.policy not in POLICIES[:4]:
            raise ValueError("PolicyConfig only represents deployable P0-P3 policies")
        if logical_shards <= 0:
            raise ValueError("logical_shards must be positive")
        if self.ef_min <= 0 or self.ef_max < self.ef_min:
            raise ValueError("invalid efSearch clamp")
        if self.policy in {"P0", "P2"}:
            if self.fixed_p is None or not 1 <= self.fixed_p <= logical_shards:
                raise ValueError("P0/P2 require fixed_p within [1, logical_shards]")
        if self.policy in {"P0", "P1"}:
            if self.uniform_ef_search is None or self.uniform_ef_search <= 0:
                raise ValueError("P0/P1 require a positive uniform_ef_search")
        if self.policy in {"P2", "P3"}:
            if self.alpha is None or self.alpha < 0:
                raise ValueError("P2/P3 require non-negative alpha")
            if self.beta is None or self.beta <= 0:
                raise ValueError("P2/P3 require positive beta")


@dataclass(frozen=True)
class OracleResult:
    policy: str
    query_id: int
    reached_target: bool
    selected_shards: tuple[int, ...]
    assigned_efs: tuple[int, ...]
    recall_at_10: float
    aggregate_distance_computations: int
    max_shard_distance_computations: int
    aggregate_nodes_visited: int
    max_shard_nodes_visited: int
    result_ids: tuple[PointId, ...]
    distance_computations_per_shard: tuple[int, ...] = ()
    nodes_visited_per_shard: tuple[int, ...] = ()


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_path(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def required_hits(target_recall: float = TARGET_RECALL, top_k: int = TOP_K) -> int:
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be within (0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return int(math.ceil(target_recall * top_k - 1e-12))


def recall_at_k(
    predicted: Sequence[PointId],
    ground_truth: Sequence[PointId],
    top_k: int = TOP_K,
) -> float:
    if len(ground_truth) < top_k:
        raise ValueError("ground truth does not contain top_k neighbors")
    return len(set(predicted[:top_k]) & set(ground_truth[:top_k])) / top_k


def distance_computations_from_usage(cpu_units: int, dimension: int) -> int:
    if cpu_units < 0 or dimension <= 0:
        raise ValueError("cpu_units must be non-negative and dimension must be positive")
    unit = dimension * np.dtype(np.float32).itemsize
    quotient, remainder = divmod(int(cpu_units), unit)
    if remainder:
        raise ValueError(
            f"hardware CPU units {cpu_units} are not divisible by dense score unit {unit}"
        )
    return quotient


def deterministic_shard_order(evidence: Iterable[ShardEvidence]) -> tuple[int, ...]:
    rows = list(evidence)
    if not rows:
        raise ValueError("routing evidence contains no candidate shard")
    shard_ids = [row.shard_id for row in rows]
    if len(set(shard_ids)) != len(shard_ids):
        raise ValueError("routing evidence contains duplicate shard IDs")
    return tuple(
        row.shard_id
        for row in sorted(
            rows,
            key=lambda row: (
                -row.candidate_count,
                row.best_candidate_rank,
                row.shard_id,
            ),
        )
    )


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def route_queries_from_trace(trace: Mapping[str, Any]) -> list[RouteQuery]:
    """Validate and load the C6-enriched production-router trace.

    The route tracer emits both the existing sorted targets and C6's ranked shard
    evidence.  This loader verifies that they describe the same adaptive shard set.
    """

    if trace.get("format_version") != 1:
        raise ValueError("unsupported route trace format_version")
    artifact = trace.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("route trace artifact must be an object")
    logical_shards = _positive_int(
        artifact.get("shard_count"), "artifact shard_count"
    )
    raw_queries = trace.get("per_query")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError("route trace per_query must be a non-empty array")
    result: list[RouteQuery] = []
    for expected_index, raw_query in enumerate(raw_queries):
        if not isinstance(raw_query, dict):
            raise ValueError(f"per_query[{expected_index}] must be an object")
        query_id = _non_negative_int(raw_query.get("query_index"), "query_index")
        if query_id != expected_index:
            raise ValueError("route trace query_index must be contiguous and ordered")
        navigation_candidate_count = _positive_int(
            raw_query.get("navigation_candidate_count"),
            "navigation_candidate_count",
        )
        raw_candidates = raw_query.get("candidates")
        if (
            not isinstance(raw_candidates, list)
            or len(raw_candidates) != navigation_candidate_count
        ):
            raise ValueError(
                "candidates must contain exactly navigation_candidate_count rows"
            )
        recomputed: dict[int, dict[str, Any]] = {}
        for expected_rank, candidate in enumerate(raw_candidates, start=1):
            if not isinstance(candidate, dict):
                raise ValueError(f"candidate rank {expected_rank} must be an object")
            if _positive_int(candidate.get("rank"), "candidate rank") != expected_rank:
                raise ValueError("candidate ranks must be contiguous and ordered")
            label = candidate.get("label")
            if isinstance(label, bool) or not isinstance(label, (int, str)):
                raise ValueError("candidate label must be a numeric or string point ID")
            distance = _finite_float(candidate.get("distance"), "candidate distance")
            shard_ids = candidate.get("shard_ids")
            if not isinstance(shard_ids, list) or not shard_ids:
                raise ValueError("candidate shard_ids must be a non-empty array")
            normalized_shards = [
                _non_negative_int(shard_id, "candidate shard_id")
                for shard_id in shard_ids
            ]
            if len(set(normalized_shards)) != len(normalized_shards):
                raise ValueError("candidate shard_ids must be unique")
            for shard_id in normalized_shards:
                accumulator = recomputed.setdefault(
                    shard_id,
                    {
                        "ranks": [],
                        "distances": [],
                        "entry_points": [],
                    },
                )
                accumulator["ranks"].append(expected_rank)
                accumulator["distances"].append(distance)
                accumulator["entry_points"].append(label)
        raw_evidence = raw_query.get("shard_evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise ValueError("C6 route trace requires non-empty shard_evidence")
        evidence: list[ShardEvidence] = []
        for evidence_index, raw in enumerate(raw_evidence):
            if not isinstance(raw, dict):
                raise ValueError(f"shard_evidence[{evidence_index}] must be an object")
            entry_points = raw.get("entry_points")
            if not isinstance(entry_points, list) or not entry_points:
                raise ValueError("shard evidence entry_points must be non-empty")
            if len(set((type(value).__name__, value) for value in entry_points)) != len(
                entry_points
            ):
                raise ValueError("shard evidence entry_points must be ordered-unique")
            row = ShardEvidence(
                shard_id=_non_negative_int(raw.get("shard_id"), "shard_id"),
                candidate_count=_positive_int(
                    raw.get("candidate_count"), "candidate_count"
                ),
                best_candidate_rank=_positive_int(
                    raw.get("best_candidate_rank"), "best_candidate_rank"
                ),
                minimum_candidate_distance=_finite_float(
                    raw.get("minimum_candidate_distance"),
                    "minimum_candidate_distance",
                ),
                mean_candidate_distance=_finite_float(
                    raw.get("mean_candidate_distance"), "mean_candidate_distance"
                ),
                entry_points=tuple(entry_points),
            )
            if row.candidate_count != row.entry_point_count:
                raise ValueError("candidate_count must equal unique entry_point_count")
            if row.best_candidate_rank > navigation_candidate_count:
                raise ValueError("best_candidate_rank exceeds navigation candidate count")
            source = recomputed.get(row.shard_id)
            if source is None:
                raise ValueError("shard evidence is not represented by any candidate")
            if row.candidate_count != len(source["ranks"]):
                raise ValueError("candidate_count differs from the ordered candidate list")
            if row.best_candidate_rank != min(source["ranks"]):
                raise ValueError("best_candidate_rank differs from the candidate list")
            if not math.isclose(
                row.minimum_candidate_distance,
                min(source["distances"]),
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError("minimum_candidate_distance differs from candidates")
            if not math.isclose(
                row.mean_candidate_distance,
                sum(source["distances"]) / len(source["distances"]),
                rel_tol=1e-7,
                abs_tol=1e-7,
            ):
                raise ValueError("mean_candidate_distance differs from candidates")
            if row.entry_points != tuple(source["entry_points"]):
                raise ValueError("entry_points differ from the ordered candidate list")
            evidence.append(row)
        if set(recomputed) != {row.shard_id for row in evidence}:
            raise ValueError("candidate and shard-evidence shard sets differ")
        candidate_shard_count = _positive_int(
            raw_query.get("candidate_shard_count"), "candidate_shard_count"
        )
        if candidate_shard_count != len(evidence):
            raise ValueError("candidate_shard_count differs from shard_evidence")
        ranked = deterministic_shard_order(evidence)
        raw_ranked = raw_query.get("ranked_candidate_shards")
        if not isinstance(raw_ranked, list) or tuple(raw_ranked) != ranked:
            raise ValueError("ranked_candidate_shards differs from deterministic ranking")
        targets = raw_query.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError("route trace targets must be non-empty")
        adaptive = tuple(int(target["shard_id"]) for target in targets)
        evidence_ids = {row.shard_id for row in evidence}
        if set(adaptive) != evidence_ids:
            raise ValueError("adaptive target set differs from routing evidence shard set")
        target_by_shard = {int(target["shard_id"]): target for target in targets}
        for row in evidence:
            target_entry_points = target_by_shard[row.shard_id].get("entry_points")
            if tuple(target_entry_points or ()) != row.entry_points:
                raise ValueError("adaptive target entry points differ from shard evidence")
        result.append(
            RouteQuery(
                query_id=query_id,
                navigation_candidate_count=navigation_candidate_count,
                candidate_shard_count=len(evidence),
                ranked_candidate_shards=ranked,
                adaptive_selected_shards=tuple(
                    shard_id for shard_id in ranked if shard_id in set(adaptive)
                ),
                evidence=tuple(evidence),
                logical_shards=logical_shards,
            )
        )
    return result


def local_trace_lookup(
    traces: Iterable[LocalSearchTrace],
) -> dict[tuple[int, int, int], LocalSearchTrace]:
    result: dict[tuple[int, int, int], LocalSearchTrace] = {}
    for trace in traces:
        if trace.query_id < 0 or trace.shard_id < 0 or trace.ef_search <= 0:
            raise ValueError("invalid local-search trace identity")
        if len(trace.result_ids) != len(trace.result_scores):
            raise ValueError("result IDs and scores have different lengths")
        if any(not math.isfinite(score) for score in trace.result_scores):
            raise ValueError("result score must be finite")
        if (
            trace.distance_computations < 0
            or trace.nodes_visited < 0
            or trace.worker_cpu_time_us < 0
            or trace.worker_cpu_wall_time_us < 0
            or trace.latency_us < 0
            or trace.response_bytes < 0
        ):
            raise ValueError("local-search work fields must be non-negative")
        key = (trace.query_id, trace.shard_id, trace.ef_search)
        if key in result:
            raise ValueError(f"duplicate local-search trace {key}")
        result[key] = trace
    return result


def merge_search_results(
    traces: Sequence[LocalSearchTrace],
    *,
    score_higher_is_better: bool,
    top_k: int = TOP_K,
) -> tuple[PointId, ...]:
    best: dict[PointId, float] = {}
    for trace in traces:
        for point_id, score in zip(trace.result_ids, trace.result_scores, strict=True):
            previous = best.get(point_id)
            if previous is None or (
                score_higher_is_better and score > previous
            ) or (not score_higher_is_better and score < previous):
                best[point_id] = score
    ordered = sorted(
        best.items(),
        key=lambda item: (item[1], str(item[0])),
        reverse=score_higher_is_better,
    )
    return tuple(point_id for point_id, _score in ordered[:top_k])


def clamp_ef(value: int, ef_min: int, ef_max: int) -> int:
    if ef_min <= 0 or ef_max < ef_min:
        raise ValueError("invalid efSearch clamp")
    return min(max(int(value), ef_min), ef_max)


def policy_shards(route: RouteQuery, config: PolicyConfig) -> tuple[int, ...]:
    if config.policy in {"P0", "P2"}:
        assert config.fixed_p is not None
        fixed_order = route.fixed_policy_shard_order()
        if config.fixed_p > len(fixed_order):
            raise ValueError(
                f"query {route.query_id} has only {len(fixed_order)} logical shards, "
                f"fewer than fixed_p={config.fixed_p}"
            )
        return fixed_order[: config.fixed_p]
    return route.adaptive_selected_shards


def policy_efs(
    route: RouteQuery,
    selected_shards: Sequence[int],
    config: PolicyConfig,
) -> tuple[int, ...]:
    if config.policy in {"P0", "P1"}:
        assert config.uniform_ef_search is not None
        return (config.uniform_ef_search,) * len(selected_shards)
    evidence = route.evidence_by_shard()
    assert config.alpha is not None and config.beta is not None
    return tuple(
        clamp_ef(
            config.alpha
            * (evidence[shard_id].entry_point_count if shard_id in evidence else 0)
            + config.beta,
            config.ef_min,
            config.ef_max,
        )
        for shard_id in selected_shards
    )


def _traces_for_allocation(
    lookup: Mapping[tuple[int, int, int], LocalSearchTrace],
    query_id: int,
    shards: Sequence[int],
    efs: Sequence[int],
) -> list[LocalSearchTrace]:
    if len(shards) != len(efs):
        raise ValueError("shard and efSearch allocations differ in length")
    result: list[LocalSearchTrace] = []
    for shard_id, ef_search in zip(shards, efs, strict=True):
        key = (query_id, int(shard_id), int(ef_search))
        try:
            result.append(lookup[key])
        except KeyError as exc:
            raise KeyError(f"missing local-search trace query/shard/ef={key}") from exc
    return result


def _query_metrics(
    route: RouteQuery,
    policy: str,
    selected_shards: Sequence[int],
    assigned_efs: Sequence[int],
    traces: Sequence[LocalSearchTrace],
    ground_truth: Sequence[PointId],
    *,
    dataset: str,
    logical_shards: int,
    score_higher_is_better: bool,
    end_to_end_latency_us: float | None = None,
    oracle_prefix_shards: int | None = None,
    oracle_total_work: int | None = None,
) -> dict[str, Any]:
    result_ids = merge_search_results(
        traces, score_higher_is_better=score_higher_is_better
    )
    distances = [trace.distance_computations for trace in traces]
    nodes = [trace.nodes_visited for trace in traces]
    cpu = [trace.worker_cpu_time_us for trace in traces]
    evidence = route.evidence_by_shard()
    return {
        "query_id": route.query_id,
        "dataset": dataset,
        "logical_shards": logical_shards,
        "policy": policy,
        "navigation_candidate_count": route.navigation_candidate_count,
        "candidate_shard_count": route.candidate_shard_count,
        "ranked_candidate_shards": list(route.ranked_candidate_shards),
        "selected_shard_ids": list(selected_shards),
        "selected_shard_count": len(selected_shards),
        "entry_point_count_per_selected_shard": [
            evidence[shard_id].entry_point_count if shard_id in evidence else 0
            for shard_id in selected_shards
        ],
        "assigned_efSearch_per_shard": list(assigned_efs),
        "recall_at_10": recall_at_k(result_ids, ground_truth),
        "distance_computations_per_shard": distances,
        "nodes_visited_per_shard": nodes,
        "worker_cpu_time_per_shard": cpu,
        "aggregate_distance_computations": sum(distances),
        "max_shard_distance_computations": max(distances, default=0),
        "aggregate_nodes_visited": sum(nodes),
        "max_shard_nodes_visited": max(nodes, default=0),
        "aggregate_worker_cpu_time_us": sum(cpu),
        "max_shard_worker_cpu_time_us": max(cpu, default=0),
        "end_to_end_latency_us_if_physical": end_to_end_latency_us,
        "oracle_prefix_shards": oracle_prefix_shards,
        "oracle_total_work": oracle_total_work,
        "result_ids": list(result_ids),
    }


def evaluate_policy_query(
    route: RouteQuery,
    config: PolicyConfig,
    lookup: Mapping[tuple[int, int, int], LocalSearchTrace],
    ground_truth: Sequence[PointId],
    *,
    dataset: str,
    logical_shards: int,
    score_higher_is_better: bool,
) -> dict[str, Any]:
    config.validate(logical_shards)
    shards = policy_shards(route, config)
    efs = policy_efs(route, shards, config)
    traces = _traces_for_allocation(lookup, route.query_id, shards, efs)
    return _query_metrics(
        route,
        config.policy,
        shards,
        efs,
        traces,
        ground_truth,
        dataset=dataset,
        logical_shards=logical_shards,
        score_higher_is_better=score_higher_is_better,
    )


def policy_config_from_dict(value: Mapping[str, Any]) -> PolicyConfig:
    allowed = {
        "policy",
        "fixed_p",
        "uniform_ef_search",
        "alpha",
        "beta",
        "ef_min",
        "ef_max",
    }
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"unexpected policy configuration fields: {unexpected}")
    return PolicyConfig(**dict(value))


def oracle_query_row(
    route: RouteQuery,
    oracle: OracleResult,
    *,
    dataset: str,
    logical_shards: int,
) -> dict[str, Any]:
    evidence = route.evidence_by_shard()
    return {
        "query_id": route.query_id,
        "dataset": dataset,
        "logical_shards": logical_shards,
        "policy": oracle.policy,
        "navigation_candidate_count": route.navigation_candidate_count,
        "candidate_shard_count": route.candidate_shard_count,
        "ranked_candidate_shards": list(route.ranked_candidate_shards),
        "selected_shard_ids": list(oracle.selected_shards),
        "selected_shard_count": len(oracle.selected_shards),
        "entry_point_count_per_selected_shard": [
            evidence[shard_id].entry_point_count if shard_id in evidence else 0
            for shard_id in oracle.selected_shards
        ],
        "assigned_efSearch_per_shard": list(oracle.assigned_efs),
        "recall_at_10": oracle.recall_at_10,
        "distance_computations_per_shard": list(
            oracle.distance_computations_per_shard
        ),
        "nodes_visited_per_shard": list(oracle.nodes_visited_per_shard),
        "worker_cpu_time_per_shard": [],
        "aggregate_distance_computations": oracle.aggregate_distance_computations,
        "max_shard_distance_computations": oracle.max_shard_distance_computations,
        "aggregate_nodes_visited": oracle.aggregate_nodes_visited,
        "max_shard_nodes_visited": oracle.max_shard_nodes_visited,
        "aggregate_worker_cpu_time_us": 0,
        "max_shard_worker_cpu_time_us": 0,
        "end_to_end_latency_us_if_physical": None,
        "oracle_prefix_shards": len(oracle.selected_shards),
        "oracle_total_work": oracle.aggregate_distance_computations,
        "oracle_reached_target": oracle.reached_target,
        "result_ids": list(oracle.result_ids),
    }


def oracle_prefix_query(
    route: RouteQuery,
    lookup: Mapping[tuple[int, int, int], LocalSearchTrace],
    ground_truth: Sequence[PointId],
    *,
    high_ef_search: int,
    score_higher_is_better: bool,
    target_recall: float = TARGET_RECALL,
) -> OracleResult:
    if high_ef_search <= 0:
        raise ValueError("high_ef_search must be positive")
    last: OracleResult | None = None
    fixed_order = route.fixed_policy_shard_order()
    for prefix in range(1, len(fixed_order) + 1):
        shards = fixed_order[:prefix]
        efs = (high_ef_search,) * prefix
        traces = _traces_for_allocation(lookup, route.query_id, shards, efs)
        result_ids = merge_search_results(
            traces, score_higher_is_better=score_higher_is_better
        )
        recall = recall_at_k(result_ids, ground_truth)
        distances = [trace.distance_computations for trace in traces]
        nodes = [trace.nodes_visited for trace in traces]
        last = OracleResult(
            policy="P4",
            query_id=route.query_id,
            reached_target=recall >= target_recall,
            selected_shards=tuple(shards),
            assigned_efs=efs,
            recall_at_10=recall,
            aggregate_distance_computations=sum(distances),
            max_shard_distance_computations=max(distances, default=0),
            aggregate_nodes_visited=sum(nodes),
            max_shard_nodes_visited=max(nodes, default=0),
            result_ids=result_ids,
            distance_computations_per_shard=tuple(distances),
            nodes_visited_per_shard=tuple(nodes),
        )
        if last.reached_target:
            return last
    assert last is not None
    return last


@dataclass(frozen=True)
class _DpState:
    total_work: int
    max_shard_work: int
    total_nodes: int
    max_shard_nodes: int
    efs: tuple[int, ...]
    traces: tuple[LocalSearchTrace, ...]

    def key(self) -> tuple[Any, ...]:
        return (
            self.total_work,
            self.max_shard_work,
            self.total_nodes,
            self.max_shard_nodes,
            self.efs,
        )


def _oracle_shard_options(
    route: RouteQuery,
    shard_id: int,
    lookup: Mapping[tuple[int, int, int], LocalSearchTrace],
    ground_truth: Sequence[PointId],
    grid: Sequence[int],
) -> list[tuple[int, LocalSearchTrace, int]]:
    by_mask: dict[int, tuple[int, LocalSearchTrace, int]] = {}
    for ef_search in grid:
        key = (route.query_id, shard_id, ef_search)
        if key not in lookup:
            raise KeyError(f"missing oracle trace query/shard/ef={key}")
        trace = lookup[key]
        mask = _ground_truth_mask(trace.result_ids, ground_truth, TOP_K)
        candidate = (mask, trace, ef_search)
        current = by_mask.get(mask)
        if current is None or (
            trace.distance_computations,
            trace.nodes_visited,
            ef_search,
        ) < (
            current[1].distance_computations,
            current[1].nodes_visited,
            current[2],
        ):
            by_mask[mask] = candidate
    options = list(by_mask.values())
    nondominated = []
    for candidate_index, candidate in enumerate(options):
        candidate_mask, candidate_trace, _candidate_ef = candidate
        dominated = any(
            other_mask | candidate_mask == other_mask
            and other_trace.distance_computations
            <= candidate_trace.distance_computations
            and (
                other_trace.distance_computations
                < candidate_trace.distance_computations
                or other_trace.nodes_visited <= candidate_trace.nodes_visited
            )
            for other_index, (other_mask, other_trace, _other_ef) in enumerate(options)
            if other_index != candidate_index
        )
        if not dominated:
            nondominated.append(candidate)
    return sorted(
        nondominated,
        key=lambda item: (
            item[1].distance_computations,
            item[1].nodes_visited,
            item[2],
            -item[0].bit_count(),
            item[0],
        ),
    )


def _advance_oracle_states(
    states: Mapping[int, _DpState],
    options: Sequence[tuple[int, LocalSearchTrace, int]],
) -> dict[int, _DpState]:
    next_states: dict[int, _DpState] = {}
    for mask, state in states.items():
        for option_mask, trace, ef_search in options:
            combined = mask | option_mask
            candidate = _DpState(
                total_work=state.total_work + trace.distance_computations,
                max_shard_work=max(state.max_shard_work, trace.distance_computations),
                total_nodes=state.total_nodes + trace.nodes_visited,
                max_shard_nodes=max(state.max_shard_nodes, trace.nodes_visited),
                efs=state.efs + (ef_search,),
                traces=state.traces + (trace,),
            )
            current = next_states.get(combined)
            if current is None or candidate.key() < current.key():
                next_states[combined] = candidate
    return next_states


def _ground_truth_mask(
    result_ids: Sequence[PointId], ground_truth: Sequence[PointId], top_k: int
) -> int:
    positions = {point_id: index for index, point_id in enumerate(ground_truth[:top_k])}
    mask = 0
    for point_id in result_ids:
        index = positions.get(point_id)
        if index is not None:
            mask |= 1 << index
    return mask


def oracle_local_budget_query(
    route: RouteQuery,
    selected_shards: Sequence[int],
    lookup: Mapping[tuple[int, int, int], LocalSearchTrace],
    ground_truth: Sequence[PointId],
    *,
    ef_grid: Sequence[int] = EF_GRID,
    score_higher_is_better: bool,
    target_recall: float = TARGET_RECALL,
    policy: str = "P5",
) -> OracleResult:
    """Exact minimum-distance allocation over a discrete per-shard EF grid.

    Official top-10 ground truth makes recall a ten-bit coverage problem: every true
    top-10 point returned by any shard ranks ahead of non-ground-truth candidates in
    the global merge.  Dynamic programming therefore needs at most 2**10 states per
    shard instead of an exponential Cartesian allocation sweep.
    """

    if policy not in {"P5", "P6"}:
        raise ValueError("local-budget oracle policy must be P5 or P6")
    shards = tuple(int(shard_id) for shard_id in selected_shards)
    if not shards or len(set(shards)) != len(shards):
        raise ValueError("selected_shards must be non-empty and unique")
    grid = tuple(sorted(set(int(value) for value in ef_grid)))
    if not grid or grid[0] <= 0:
        raise ValueError("ef_grid must contain positive values")
    states: dict[int, _DpState] = {
        0: _DpState(0, 0, 0, 0, (), ())
    }
    for shard_id in shards:
        states = _advance_oracle_states(
            states,
            _oracle_shard_options(route, shard_id, lookup, ground_truth, grid),
        )
    needed = required_hits(target_recall)
    feasible = [
        (mask, state) for mask, state in states.items() if mask.bit_count() >= needed
    ]
    if feasible:
        _mask, best = min(feasible, key=lambda item: item[1].key())
        reached = True
    else:
        _mask, best = max(
            states.items(),
            key=lambda item: (
                item[0].bit_count(),
                -item[1].total_work,
                tuple(-value for value in item[1].efs),
            ),
        )
        reached = False
    result_ids = merge_search_results(
        best.traces, score_higher_is_better=score_higher_is_better
    )
    recall = recall_at_k(result_ids, ground_truth)
    if reached and recall < target_recall:
        raise RuntimeError(
            "ground-truth coverage oracle invariant failed after global result merge"
        )
    return OracleResult(
        policy=policy,
        query_id=route.query_id,
        reached_target=reached,
        selected_shards=shards,
        assigned_efs=best.efs,
        recall_at_10=recall,
        aggregate_distance_computations=best.total_work,
        max_shard_distance_computations=best.max_shard_work,
        aggregate_nodes_visited=best.total_nodes,
        max_shard_nodes_visited=best.max_shard_nodes,
        result_ids=result_ids,
        distance_computations_per_shard=tuple(
            trace.distance_computations for trace in best.traces
        ),
        nodes_visited_per_shard=tuple(trace.nodes_visited for trace in best.traces),
    )


def full_oracle_query(
    route: RouteQuery,
    lookup: Mapping[tuple[int, int, int], LocalSearchTrace],
    ground_truth: Sequence[PointId],
    *,
    ef_grid: Sequence[int] = EF_GRID,
    score_higher_is_better: bool,
    target_recall: float = TARGET_RECALL,
) -> OracleResult:
    fixed_order = route.fixed_policy_shard_order()
    grid = tuple(sorted(set(int(value) for value in ef_grid)))
    if not grid or grid[0] <= 0:
        raise ValueError("ef_grid must contain positive values")
    states: dict[int, _DpState] = {0: _DpState(0, 0, 0, 0, (), ())}
    needed = required_hits(target_recall)
    best_feasible: tuple[tuple[Any, ...], int, _DpState] | None = None
    best_fallback: tuple[tuple[Any, ...], int, _DpState] | None = None
    for prefix, shard_id in enumerate(fixed_order, start=1):
        states = _advance_oracle_states(
            states,
            _oracle_shard_options(route, shard_id, lookup, ground_truth, grid),
        )
        for mask, state in states.items():
            fallback_key = (
                -mask.bit_count(),
                state.total_work,
                state.max_shard_work,
                prefix,
                state.efs,
            )
            if best_fallback is None or fallback_key < best_fallback[0]:
                best_fallback = (fallback_key, prefix, state)
            if mask.bit_count() >= needed:
                feasible_key = (
                    state.total_work,
                    state.max_shard_work,
                    prefix,
                    state.total_nodes,
                    state.max_shard_nodes,
                    state.efs,
                )
                if best_feasible is None or feasible_key < best_feasible[0]:
                    best_feasible = (feasible_key, prefix, state)
    selected = best_feasible or best_fallback
    assert selected is not None
    _key, prefix, state = selected
    result_ids = merge_search_results(
        state.traces, score_higher_is_better=score_higher_is_better
    )
    recall = recall_at_k(result_ids, ground_truth)
    reached = best_feasible is not None
    if reached and recall < target_recall:
        raise RuntimeError(
            "ground-truth coverage oracle invariant failed after global result merge"
        )
    return OracleResult(
        policy="P6",
        query_id=route.query_id,
        reached_target=reached,
        selected_shards=fixed_order[:prefix],
        assigned_efs=state.efs,
        recall_at_10=recall,
        aggregate_distance_computations=state.total_work,
        max_shard_distance_computations=state.max_shard_work,
        aggregate_nodes_visited=state.total_nodes,
        max_shard_nodes_visited=state.max_shard_nodes,
        result_ids=result_ids,
        distance_computations_per_shard=tuple(
            trace.distance_computations for trace in state.traces
        ),
        nodes_visited_per_shard=tuple(trace.nodes_visited for trace in state.traces),
    )


def summarize_query_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty query set")
    recall = [float(row["recall_at_10"]) for row in rows]
    fanout = [int(row["selected_shard_count"]) for row in rows]
    distance = [int(row["aggregate_distance_computations"]) for row in rows]
    max_distance = [int(row["max_shard_distance_computations"]) for row in rows]
    nodes = [int(row["aggregate_nodes_visited"]) for row in rows]
    max_nodes = [int(row["max_shard_nodes_visited"]) for row in rows]
    cpu = [int(row["aggregate_worker_cpu_time_us"]) for row in rows]
    latencies = [
        float(row["end_to_end_latency_us_if_physical"])
        for row in rows
        if row.get("end_to_end_latency_us_if_physical") is not None
    ]
    summary: dict[str, Any] = {
        "query_count": len(rows),
        "recall_at_10": statistics.fmean(recall),
        "p5_per_query_recall": float(np.percentile(recall, 5)),
        "mean_shards_per_query": statistics.fmean(fanout),
        "p95_shards_per_query": float(np.percentile(fanout, 95)),
        "aggregate_distance_computations_per_query": statistics.fmean(distance),
        "p95_aggregate_distance_computations": float(np.percentile(distance, 95)),
        "max_shard_distance_computations_mean": statistics.fmean(max_distance),
        "p95_max_shard_distance_computations": float(
            np.percentile(max_distance, 95)
        ),
        "p99_max_shard_distance_computations": float(
            np.percentile(max_distance, 99)
        ),
        "aggregate_nodes_visited_per_query": statistics.fmean(nodes),
        "p95_max_shard_nodes_visited": float(np.percentile(max_nodes, 95)),
        "worker_cpu_time_us_per_query": statistics.fmean(cpu),
    }
    if latencies:
        summary.update(
            {
                "mean_latency_us": statistics.fmean(latencies),
                "p50_latency_us": float(np.percentile(latencies, 50)),
                "p95_latency_us": float(np.percentile(latencies, 95)),
                "p99_latency_us": float(np.percentile(latencies, 99)),
            }
        )
    return summary


def select_tuned_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    target_recall: float = TARGET_RECALL,
) -> Mapping[str, Any]:
    feasible = [
        row for row in candidates if float(row["recall_at_10"]) >= target_recall
    ]
    if not feasible:
        raise RuntimeError("no tuning candidate reaches the recall target")
    best_work = min(
        float(row["aggregate_distance_computations_per_query"])
        for row in feasible
    )
    near_best = [
        row
        for row in feasible
        if float(row["aggregate_distance_computations_per_query"])
        <= best_work * 1.02
    ]
    return min(
        near_best,
        key=lambda row: (
            float(row["p99_max_shard_distance_computations"]),
            float(row["aggregate_distance_computations_per_query"]),
            canonical_json_sha256(row),
        ),
    )


def paired_bootstrap_relative_reduction(
    baseline: Sequence[float],
    adaptive: Sequence[float],
    *,
    repetitions: int = 10_000,
    seed: int = 20260831,
) -> dict[str, float]:
    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(adaptive, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right) or not len(left):
        raise ValueError("paired samples must be non-empty equal-length vectors")
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if np.any(left <= 0) or np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("bootstrap work samples must be finite with positive baselines")
    observed = float(np.mean((left - right) / left))
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(left), size=len(left))
        samples[index] = np.mean((left[selected] - right[selected]) / left[selected])
    return {
        "mean_relative_reduction": observed,
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "bootstrap_repetitions": float(repetitions),
    }


def classify_static_waste(
    static_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[OracleResult],
    *,
    target_recall: float = TARGET_RECALL,
) -> dict[str, Any]:
    if len(static_rows) != len(oracle_rows) or not static_rows:
        raise ValueError("static and oracle rows must be non-empty and paired")
    counts = {"under_searched": 0, "near_sufficient": 0, "over_searched": 0}
    labels: list[str] = []
    for static, oracle in zip(static_rows, oracle_rows, strict=True):
        if int(static["query_id"]) != oracle.query_id:
            raise ValueError("static/oracle query IDs are not aligned")
        if float(static["recall_at_10"]) < target_recall:
            label = "under_searched"
        elif int(static["aggregate_distance_computations"]) <= 1.10 * max(
            oracle.aggregate_distance_computations, 1
        ):
            label = "near_sufficient"
        else:
            label = "over_searched"
        counts[label] += 1
        labels.append(label)
    return {
        "query_count": len(labels),
        "labels": labels,
        **{key: value / len(labels) for key, value in counts.items()},
    }


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


class DurableJsonlBatchWriter:
    """Append JSONL with bounded-loss batching and a final durable sync."""

    def __init__(self, path: str | Path, *, fsync_interval: int = 100) -> None:
        if fsync_interval <= 0:
            raise ValueError("fsync_interval must be positive")
        self.path = Path(path).expanduser().resolve()
        self.fsync_interval = fsync_interval
        self.handle: Any = None
        self.pending = 0

    def __enter__(self) -> "DurableJsonlBatchWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
        return self

    def append(self, payload: Mapping[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("DurableJsonlBatchWriter is not open")
        self.handle.write(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        self.pending += 1
        if self.pending >= self.fsync_interval:
            self.sync()

    def sync(self) -> None:
        if self.handle is None or self.pending == 0:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.pending = 0

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.sync()
        finally:
            if self.handle is not None:
                self.handle.close()
                self.handle = None


def read_local_search_jsonl(path: str | Path) -> list[LocalSearchTrace]:
    source = Path(path).expanduser().resolve()
    result: list[LocalSearchTrace] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                result.append(
                    LocalSearchTrace(
                        query_id=int(row["query_id"]),
                        shard_id=int(row["shard_id"]),
                        ef_search=int(row["ef_search"]),
                        result_ids=tuple(row["result_ids"]),
                        result_scores=tuple(float(value) for value in row["result_scores"]),
                        distance_computations=int(row["distance_computations"]),
                        nodes_visited=int(row["nodes_visited"]),
                        worker_cpu_time_us=int(row["worker_cpu_time_us"]),
                        worker_cpu_wall_time_us=int(row["worker_cpu_wall_time_us"]),
                        latency_us=float(row["latency_us"]),
                        response_bytes=int(row.get("response_bytes", 0)),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid local-search JSONL row at line {line_number}"
                ) from exc
    local_trace_lookup(result)
    return result


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def write_csv_atomic(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    fields = list(rows[0].keys())
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if list(row.keys()) != fields:
                raise ValueError("CSV rows do not share one ordered schema")
            encoded = {
                key: (
                    json.dumps(value, separators=(",", ":"), ensure_ascii=True)
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(encoded)
    os.replace(temporary, destination)
    return destination


def oracle_as_dict(result: OracleResult) -> dict[str, Any]:
    return asdict(result)
