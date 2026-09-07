#!/usr/bin/env python3
"""Prepare an attraction-weighted frozen-upper graph and optional METIS owner.

Calibration hits must be exported by ``orion_export_upper_hits`` from the same
checksum-bound production artifact and with the same search EF that future L0
placement will use.  This tool changes only upper-node shard labels.  It never
builds a lower HNSW, reads a live shard load, or mutates the upper graph.

The default backend invokes ``gpmetis``.  ``--emit-only`` produces the complete
METIS input for an external KaHIP/METIS run, while ``--partition-file`` imports
and validates a precomputed owner without invoking an installed partitioner.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance import attraction_weighted_partitioner as awp  # noqa: E402


TOOL_PATH = "experiments/l1_balance/prepare_attraction_weighted_metis.py"
MANIFEST_NAME = "attraction-weighted-manifest.json"
METIS_GRAPH_NAME = "upper-attraction.graph"
WEIGHTS_NAME = "attraction-weights.u64le"
COUNTS_NAME = "calibration-first-hit-counts.u64le"
OWNER_NAME = "ATTRACTION_WEIGHTED_METIS.owner.i32le"
FIXED_METIS_SEED = 0
PARTITIONER_BACKENDS = (
    "auto",
    "gpmetis",
    "pymetis",
    "attraction-refine",
    "attraction-nav-refine",
)


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def write_json_new(path: Path, value: Any) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(data)


def write_checksum_listing(output_dir: Path) -> Path:
    path = output_dir / "checksums.sha256"
    checksums = [
        f"{sha256_path(candidate)}  {candidate.name}"
        for candidate in sorted(output_dir.iterdir())
        if candidate.is_file() and candidate.name != path.name
    ]
    with path.open("x", encoding="ascii") as handle:
        handle.write("\n".join(checksums) + "\n")
    return path


def write_failure(
    output_dir: Path,
    *,
    stage: str,
    error: Exception,
    evidence: dict[str, Any] | None = None,
) -> None:
    path = output_dir / "execution-failed.json"
    if not path.exists():
        write_json_new(
            path,
            {
                "status": "FAIL",
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "evidence": evidence or {},
            },
        )
    checksum_path = output_dir / "checksums.sha256"
    if not checksum_path.exists():
        write_checksum_listing(output_dir)


def require_absolute_file(raw: str, name: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not reference a file: {path}")
    return path


def load_upper_graph(
    artifact_path: Path,
) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...], tuple[int, ...], str]:
    artifact = load_json_object(artifact_path, "artifact")
    upper_nodes = artifact.get("upper_nodes")
    if not isinstance(upper_nodes, list) or not upper_nodes:
        raise ValueError("artifact has no upper_nodes")
    labels: list[int] = []
    for index, node in enumerate(upper_nodes):
        if not isinstance(node, dict):
            raise ValueError(f"artifact.upper_nodes[{index}] is not an object")
        try:
            label = int(node["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"artifact.upper_nodes[{index}] has invalid label") from exc
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise ValueError("artifact upper labels are not unique")
    label_to_local = {label: index for index, label in enumerate(labels)}

    upper_graph = artifact.get("upper_graph")
    if not isinstance(upper_graph, dict):
        raise ValueError("artifact has no immutable upper_graph")
    graph_nodes = upper_graph.get("nodes")
    if not isinstance(graph_nodes, list) or len(graph_nodes) != len(labels):
        raise ValueError("upper graph node count differs from upper_nodes")
    adjacency: list[tuple[int, ...] | None] = [None] * len(labels)
    seen: set[int] = set()
    for graph_index, graph_node in enumerate(graph_nodes):
        if not isinstance(graph_node, dict):
            raise ValueError(f"upper_graph.nodes[{graph_index}] is not an object")
        try:
            local = label_to_local[int(graph_node["label"])]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"upper_graph.nodes[{graph_index}] has an unknown label"
            ) from exc
        if local in seen:
            raise ValueError("upper graph repeats a node label")
        seen.add(local)
        levels = graph_node.get("neighbors_by_level")
        if not isinstance(levels, list) or not levels or not isinstance(levels[0], list):
            raise ValueError("upper graph node lacks a level-0 neighbour list")
        row: list[int] = []
        row_seen: set[int] = set()
        for raw_neighbour in levels[0]:
            try:
                neighbour = label_to_local[int(raw_neighbour)]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("upper graph has an unknown neighbour label") from exc
            if neighbour == local:
                raise ValueError("upper graph contains a level-0 self edge")
            if neighbour in row_seen:
                raise ValueError("upper graph repeats a level-0 neighbour")
            row_seen.add(neighbour)
            row.append(neighbour)
        adjacency[local] = tuple(row)
    if len(seen) != len(labels) or any(row is None for row in adjacency):
        raise ValueError("upper graph does not cover every upper node")
    normalised = awp.normalise_undirected_graph(
        tuple(row for row in adjacency if row is not None)
    )
    return artifact, normalised, tuple(labels), canonical_sha256(upper_graph)


def validate_calibration_manifest(
    *,
    artifact_path: Path,
    artifact_sha256: str,
    hits_path: Path,
    manifest_path: Path,
    expected_search_ef: int,
) -> dict[str, Any]:
    manifest = load_json_object(manifest_path, "calibration manifest")
    if manifest.get("format_version") != 1:
        raise ValueError("calibration manifest format_version must be 1")
    recorded_artifact = require_absolute_file(
        str(manifest.get("artifact_path", "")), "calibration.artifact_path"
    )
    if recorded_artifact != artifact_path:
        raise ValueError("calibration manifest references a different artifact")
    if manifest.get("artifact_sha256") != artifact_sha256:
        raise ValueError("calibration artifact checksum differs")
    if manifest.get("upper_graph_present") is not True:
        raise ValueError("calibration was not produced from a portable upper graph")
    recorded_hits = require_absolute_file(
        str(manifest.get("hits_path", "")), "calibration.hits_path"
    )
    if recorded_hits != hits_path:
        raise ValueError("calibration manifest references a different hits file")
    if manifest.get("hits_sha256") != sha256_path(hits_path):
        raise ValueError("calibration hits checksum differs")
    row_count = manifest.get("row_count")
    top_k = manifest.get("top_k")
    search_ef = manifest.get("search_ef")
    for name, value in (
        ("row_count", row_count),
        ("top_k", top_k),
        ("search_ef", search_ef),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"calibration.{name} must be a positive integer")
    if search_ef != expected_search_ef:
        raise ValueError(
            "calibration search EF differs from the frozen placement EF: "
            f"{search_ef} != {expected_search_ef}"
        )
    expected_size = row_count * top_k * 8
    if hits_path.stat().st_size != expected_size:
        raise ValueError(
            f"calibration hits have {hits_path.stat().st_size} bytes; "
            f"expected {expected_size}"
        )
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
    }


def validate_calibration_selection(
    *,
    estimator: str,
    selection_manifest_path: Path | None,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    if estimator == awp.EXACT_ESTIMATOR:
        if selection_manifest_path is not None:
            raise ValueError(
                "exact full-corpus attraction must not provide a sample selection manifest"
            )
        return {
            "mode": "full_corpus_exact_top1",
            "row_count": int(calibration["row_count"]),
            "vectors_sha256": calibration["vectors_sha256"],
        }
    if selection_manifest_path is None:
        raise ValueError(
            "sample estimator requires --calibration-selection-manifest"
        )
    path = selection_manifest_path.expanduser().resolve(strict=True)
    selection = load_json_object(path, "calibration selection manifest")
    required = {
        "format_version",
        "selection_method",
        "seed",
        "row_count",
        "vectors_sha256",
    }
    missing = sorted(required - set(selection))
    if missing:
        raise ValueError(f"calibration selection manifest lacks fields: {missing}")
    if selection.get("format_version") != 1:
        raise ValueError("calibration selection format_version must be 1")
    method = selection.get("selection_method")
    seed = selection.get("seed")
    row_count = selection.get("row_count")
    if not isinstance(method, str) or not method:
        raise ValueError("calibration selection_method must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("calibration seed must be an integer")
    if row_count != calibration["row_count"]:
        raise ValueError("calibration selection row_count differs from hit export")
    if selection.get("vectors_sha256") != calibration["vectors_sha256"]:
        raise ValueError("calibration selection vector checksum differs from hit export")
    return {
        **selection,
        "manifest_path": str(path),
        "manifest_sha256": sha256_path(path),
    }


def iter_local_hit_rows(
    hits_path: Path,
    *,
    row_count: int,
    top_k: int,
    label_to_local: dict[int, int],
) -> Iterator[tuple[int, ...]]:
    row_struct = struct.Struct("<" + "Q" * top_k)
    with hits_path.open("rb") as handle:
        for row_index in range(row_count):
            data = handle.read(row_struct.size)
            if len(data) != row_struct.size:
                raise ValueError(f"calibration hits ended before row {row_index}")
            labels = row_struct.unpack(data)
            try:
                yield tuple(label_to_local[int(label)] for label in labels)
            except KeyError as exc:
                raise ValueError(
                    f"calibration row {row_index} contains a label outside upper graph"
                ) from exc
        if handle.read(1):
            raise ValueError("calibration hits contain trailing bytes")


def read_partition(path: Path, node_count: int) -> tuple[int, ...]:
    values: list[int] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            raise ValueError(f"partition file has an empty line at {line_number}")
        try:
            values.append(int(line))
        except ValueError as exc:
            raise ValueError(
                f"partition file line {line_number} is not an integer"
            ) from exc
    if len(values) != node_count:
        raise ValueError(
            f"partition file has {len(values)} rows; expected {node_count}"
        )
    return tuple(values)


def read_owner_binary(
    path: Path, node_count: int, num_partitions: int
) -> tuple[int, ...]:
    expected_size = node_count * 4
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"initial owner has {path.stat().st_size} bytes; expected {expected_size}"
        )
    values = tuple(
        value[0] for value in struct.iter_unpack("<i", path.read_bytes())
    )
    if any(value < 0 or value >= num_partitions for value in values):
        raise ValueError("initial owner contains an invalid partition")
    return values


def bundle_partition_file(
    source: Path, output_dir: Path, num_partitions: int
) -> Path:
    target = output_dir / f"{METIS_GRAPH_NAME}.part.{num_partitions}"
    if source.resolve() == target.resolve():
        return target
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    return target


def run_gpmetis(
    graph_path: Path,
    num_partitions: int,
    *,
    binary: str,
    imbalance_tolerance: float,
) -> tuple[Path, dict[str, Any]]:
    resolved = shutil.which(binary)
    if resolved is None:
        raise FileNotFoundError(
            f"METIS binary {binary!r} was not found; use --emit-only or "
            "--partition-file"
        )
    ufactor = metis_ufactor(imbalance_tolerance)
    command = [
        resolved,
        "-ptype=kway",
        "-objtype=cut",
        f"-seed={FIXED_METIS_SEED}",
        f"-ufactor={ufactor}",
        str(graph_path),
        str(num_partitions),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "gpmetis failed: "
            + json.dumps(
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                sort_keys=True,
            )
        )
    output = Path(f"{graph_path}.part.{num_partitions}")
    if not output.is_file():
        raise RuntimeError(f"gpmetis did not create {output}")
    return output, {
        "backend": "gpmetis",
        "binary": resolved,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "seed": FIXED_METIS_SEED,
        "ufactor": ufactor,
    }


def metis_ufactor(imbalance_tolerance: float) -> int:
    """Leave one per-mille of headroom for strict post-partition validation."""

    requested = int(math.floor(imbalance_tolerance * 1000.0 + 1e-12))
    return max(1, requested - 1)


def run_pymetis(
    graph: awp.SearchWeightedGraph,
    weights: awp.AttractionWeights,
    output_dir: Path,
    num_partitions: int,
    *,
    imbalance_tolerance: float,
) -> tuple[Path, dict[str, Any]]:
    try:
        import pymetis
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            "PyMetis is not installed; install it in the experiment environment "
            "or use --partition-file/--emit-only"
        ) from exc
    xadj = array("q", [0])
    adjncy: list[int] = []
    eweights: list[int] = []
    for row in graph.adjacency:
        for neighbour, edge_weight in row:
            adjncy.append(neighbour)
            eweights.append(edge_weight)
        xadj.append(len(adjncy))
    adjncy_buffer = array("q", adjncy)
    eweights_buffer = array("q", eweights)
    vertex_weight_buffer = array("q", weights.values)
    ufactor = metis_ufactor(imbalance_tolerance)
    options = pymetis.Options(seed=FIXED_METIS_SEED, ufactor=ufactor)
    started = time.perf_counter()
    result = pymetis.part_graph(
        num_partitions,
        adjacency=pymetis.CSRAdjacency(xadj, adjncy_buffer),
        vweights=vertex_weight_buffer,
        eweights=eweights_buffer,
        recursive=False,
        options=options,
        warn_on_copies=True,
    )
    elapsed = time.perf_counter() - started
    owner = tuple(int(value) for value in result.vertex_part)
    output = output_dir / f"{METIS_GRAPH_NAME}.part.{num_partitions}"
    with output.open("x", encoding="ascii") as handle:
        handle.write("\n".join(str(value) for value in owner) + "\n")
    return output, {
        "backend": "pymetis",
        "version": str(pymetis.version),
        "seed": FIXED_METIS_SEED,
        "ufactor": ufactor,
        "objective": "edge_cut",
        "recursive": False,
        "reported_edge_cut": int(result.edge_cuts),
        "elapsed_seconds": elapsed,
        "partition_path": str(output),
        "partition_sha256": sha256_path(output),
    }


def run_attraction_refinement(
    adjacency: tuple[tuple[int, ...], ...],
    weights: awp.AttractionWeights,
    initial_owner: tuple[int, ...],
    output_dir: Path,
    num_partitions: int,
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    result = awp.refine_owner_by_attraction(
        adjacency,
        initial_owner,
        weights,
        num_partitions,
    )
    output = output_dir / f"{METIS_GRAPH_NAME}.part.{num_partitions}"
    with output.open("x", encoding="ascii") as handle:
        handle.write("\n".join(str(value) for value in result.owner) + "\n")
    return output, {
        "backend": "attraction-refine",
        "method": "attraction_boundary_refinement_v1",
        "elapsed_seconds": time.perf_counter() - started,
        "partition_path": str(output),
        "partition_sha256": sha256_path(output),
        "initial_partition_loads": list(result.initial_partition_loads),
        "final_partition_loads": list(result.final_partition_loads),
        "initial_cut_edges": result.initial_cut_edges,
        "final_cut_edges": result.final_cut_edges,
        "cut_limit": result.cut_limit,
        "moved_node_count": result.moved_node_count,
        "moved_attraction_weight": result.moved_attraction_weight,
        "rounds": list(result.rounds),
        "semantic_sha256": result.semantic_sha256,
        "fixed_contract": {
            "maximum_cut_ratio": (
                awp.REFINEMENT_MAX_CUT_NUMERATOR
                / awp.REFINEMENT_MAX_CUT_DENOMINATOR
            ),
            "target_ceiling_ratio": (
                awp.REFINEMENT_TARGET_CEILING_NUMERATOR
                / awp.REFINEMENT_TARGET_CEILING_DENOMINATOR
            ),
            "maximum_rounds": awp.REFINEMENT_MAX_ROUNDS,
            "each_node_moves_at_most_once": True,
            "target_requires_raw_hnsw_neighbour_owner": True,
        },
    }


def run_attraction_navigation_refinement(
    adjacency: tuple[tuple[int, ...], ...],
    navigation_rows: tuple[tuple[int, ...], ...],
    weights: awp.AttractionWeights,
    initial_owner: tuple[int, ...],
    output_dir: Path,
    num_partitions: int,
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    result = awp.refine_owner_by_attraction_navigation(
        adjacency,
        navigation_rows,
        initial_owner,
        weights,
        num_partitions,
    )
    output = output_dir / f"{METIS_GRAPH_NAME}.part.{num_partitions}"
    with output.open("x", encoding="ascii") as handle:
        handle.write("\n".join(str(value) for value in result.owner) + "\n")
    return output, {
        "backend": "attraction-nav-refine",
        "method": "attraction_navigation_refinement_v1",
        "elapsed_seconds": time.perf_counter() - started,
        "partition_path": str(output),
        "partition_sha256": sha256_path(output),
        "initial_partition_loads": list(result.initial_partition_loads),
        "final_partition_loads": list(result.final_partition_loads),
        "initial_cut_edges": result.initial_cut_edges,
        "final_cut_edges": result.final_cut_edges,
        "cut_limit": result.cut_limit,
        "moved_node_count": result.moved_node_count,
        "moved_attraction_weight": result.moved_attraction_weight,
        "rounds": list(result.rounds),
        "semantic_sha256": result.semantic_sha256,
        "fixed_contract": {
            "maximum_cut_ratio": (
                awp.REFINEMENT_MAX_CUT_NUMERATOR
                / awp.REFINEMENT_MAX_CUT_DENOMINATOR
            ),
            "target_ceiling_ratio": (
                awp.REFINEMENT_TARGET_CEILING_NUMERATOR
                / awp.REFINEMENT_TARGET_CEILING_DENOMINATOR
            ),
            "maximum_rounds": awp.REFINEMENT_MAX_ROUNDS,
            "maximum_navigation_vote_loss": awp.NAV_REFINEMENT_MAX_VOTE_LOSS,
            "moved_attraction_budget_fraction": (
                awp.NAV_REFINEMENT_BUDGET_NUMERATOR
                / awp.NAV_REFINEMENT_BUDGET_DENOMINATOR
            ),
            "each_node_moves_at_most_once": True,
            "target_requires_navigation_or_raw_hnsw_evidence": True,
        },
    }


def run_partitioner(
    backend: str,
    graph_path: Path,
    raw_adjacency: tuple[tuple[int, ...], ...],
    graph: awp.SearchWeightedGraph,
    weights: awp.AttractionWeights,
    initial_owner: tuple[int, ...] | None,
    navigation_rows: tuple[tuple[int, ...], ...] | None,
    output_dir: Path,
    num_partitions: int,
    *,
    binary: str,
    imbalance_tolerance: float,
) -> tuple[Path, dict[str, Any]]:
    if backend not in PARTITIONER_BACKENDS:
        raise ValueError(f"unknown partitioner backend: {backend}")
    if backend == "attraction-refine":
        if initial_owner is None:
            raise ValueError(
                "attraction-refine requires --initial-owner-binary"
            )
        return run_attraction_refinement(
            raw_adjacency,
            weights,
            initial_owner,
            output_dir,
            num_partitions,
        )
    if backend == "attraction-nav-refine":
        if initial_owner is None or navigation_rows is None:
            raise ValueError(
                "attraction-nav-refine requires initial owner and upper "
                "self-navigation rows"
            )
        return run_attraction_navigation_refinement(
            raw_adjacency,
            navigation_rows,
            weights,
            initial_owner,
            output_dir,
            num_partitions,
        )
    if backend in {"auto", "gpmetis"} and shutil.which(binary) is not None:
        return run_gpmetis(
            graph_path,
            num_partitions,
            binary=binary,
            imbalance_tolerance=imbalance_tolerance,
        )
    if backend in {"auto", "pymetis"}:
        return run_pymetis(
            graph,
            weights,
            output_dir,
            num_partitions,
            imbalance_tolerance=imbalance_tolerance,
        )
    raise FileNotFoundError(
        f"METIS binary {binary!r} was not found; use --partitioner-backend "
        "pymetis, --emit-only, or --partition-file"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--calibration-hits", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-selection-manifest", type=Path)
    parser.add_argument("--upper-self-navigation-hits", type=Path)
    parser.add_argument("--upper-self-navigation-manifest", type=Path)
    parser.add_argument("--placement-search-ef", type=int, required=True)
    parser.add_argument("--num-partitions", type=int)
    parser.add_argument(
        "--estimator",
        choices=awp.SUPPORTED_ESTIMATORS,
        default=awp.SAMPLE_ESTIMATOR,
    )
    parser.add_argument(
        "--edge-mode",
        choices=awp.SUPPORTED_EDGE_MODES,
        default=awp.HIT_COOCCURRENCE_EDGE_MODE,
    )
    parser.add_argument(
        "--imbalance-tolerance",
        type=float,
        default=awp.DEFAULT_IMBALANCE_TOLERANCE,
    )
    parser.add_argument(
        "--heavy-atom-fraction",
        type=float,
        default=awp.DEFAULT_HEAVY_ATOM_FRACTION,
    )
    parser.add_argument("--metis-binary", default="gpmetis")
    parser.add_argument(
        "--partitioner-backend",
        choices=PARTITIONER_BACKENDS,
        default="auto",
    )
    parser.add_argument("--partition-file", type=Path)
    parser.add_argument("--initial-owner-binary", type=Path)
    parser.add_argument("--emit-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.emit_only and args.partition_file is not None:
        parser.error("--emit-only and --partition-file are mutually exclusive")
    if args.partitioner_backend in {
        "attraction-refine",
        "attraction-nav-refine",
    }:
        if args.initial_owner_binary is None:
            parser.error(
                "--partitioner-backend attraction-refine requires "
                "--initial-owner-binary"
            )
        if args.emit_only or args.partition_file is not None:
            parser.error(
                "attraction-refine cannot be combined with --emit-only or "
                "--partition-file"
            )
    if args.partitioner_backend == "attraction-nav-refine" and (
        args.upper_self_navigation_hits is None
        or args.upper_self_navigation_manifest is None
    ):
        parser.error(
            "attraction-nav-refine requires --upper-self-navigation-hits and "
            "--upper-self-navigation-manifest"
        )
    if args.placement_search_ef <= 0:
        parser.error("--placement-search-ef must be positive")
    if args.num_partitions is not None and args.num_partitions <= 0:
        parser.error("--num-partitions must be positive")
    return args


def run(args: argparse.Namespace) -> Path:
    run_started = time.perf_counter()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    artifact_path = args.artifact.expanduser().resolve(strict=True)
    hits_path = args.calibration_hits.expanduser().resolve(strict=True)
    calibration_manifest_path = args.calibration_manifest.expanduser().resolve(
        strict=True
    )
    artifact_sha256 = sha256_path(artifact_path)
    artifact, adjacency, labels, upper_graph_sha256 = load_upper_graph(artifact_path)
    artifact_load_seconds = time.perf_counter() - run_started
    logical_point_count = artifact.get("logical_point_count")
    if (
        isinstance(logical_point_count, bool)
        or not isinstance(logical_point_count, int)
        or logical_point_count <= 0
    ):
        raise ValueError("artifact.logical_point_count must be a positive integer")
    artifact_shards = artifact.get("shard_count")
    if isinstance(artifact_shards, bool) or not isinstance(artifact_shards, int):
        raise ValueError("artifact.shard_count must be an integer")
    num_partitions = args.num_partitions or artifact_shards
    if num_partitions <= 0 or num_partitions > len(labels):
        raise ValueError("num_partitions is outside the upper-node range")

    calibration = validate_calibration_manifest(
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        hits_path=hits_path,
        manifest_path=calibration_manifest_path,
        expected_search_ef=args.placement_search_ef,
    )
    calibration_selection = validate_calibration_selection(
        estimator=args.estimator,
        selection_manifest_path=args.calibration_selection_manifest,
        calibration=calibration,
    )
    self_navigation: dict[str, Any] | None = None
    navigation_rows: tuple[tuple[int, ...], ...] | None = None
    if args.upper_self_navigation_hits is not None:
        self_hits_path = args.upper_self_navigation_hits.expanduser().resolve(
            strict=True
        )
        self_manifest_path = (
            args.upper_self_navigation_manifest.expanduser().resolve(strict=True)
        )
        self_navigation = validate_calibration_manifest(
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            hits_path=self_hits_path,
            manifest_path=self_manifest_path,
            expected_search_ef=args.placement_search_ef,
        )
        if int(self_navigation["row_count"]) != len(labels):
            raise ValueError(
                "upper self-navigation row count differs from upper-node count"
            )
        navigation_rows = tuple(
            iter_local_hit_rows(
                self_hits_path,
                row_count=int(self_navigation["row_count"]),
                top_k=int(self_navigation["top_k"]),
                label_to_local={label: index for index, label in enumerate(labels)},
            )
        )
    contract_validation_seconds = (
        time.perf_counter() - run_started - artifact_load_seconds
    )
    label_to_local = {label: index for index, label in enumerate(labels)}
    measurement_started = time.perf_counter()
    measurements = awp.measure_calibration_rows(
        adjacency,
        iter_local_hit_rows(
            hits_path,
            row_count=int(calibration["row_count"]),
            top_k=int(calibration["top_k"]),
            label_to_local=label_to_local,
        ),
        edge_mode=args.edge_mode,
    )
    calibration_measurement_seconds = time.perf_counter() - measurement_started
    attraction_started = time.perf_counter()
    weights = awp.estimate_attraction_weights(
        measurements.first_hit_counts,
        calibration_count=measurements.row_count,
        logical_point_count=logical_point_count,
        estimator=args.estimator,
    )
    attraction_estimation_seconds = time.perf_counter() - attraction_started
    graph_started = time.perf_counter()
    graph = awp.build_search_weighted_graph(adjacency, measurements)
    search_weighted_graph_seconds = time.perf_counter() - graph_started
    mean_partition_load = logical_point_count / num_partitions
    heavy_atom_ratio = weights.max_weight / mean_partition_load

    output_dir.mkdir(parents=True, exist_ok=False)
    serialization_started = time.perf_counter()
    counts_path = output_dir / COUNTS_NAME
    with counts_path.open("xb") as handle:
        for value in measurements.first_hit_counts:
            handle.write(struct.pack("<Q", value))
    weights_path = output_dir / WEIGHTS_NAME
    with weights_path.open("xb") as handle:
        for value in weights.values:
            handle.write(struct.pack("<Q", value))
    graph_path = output_dir / METIS_GRAPH_NAME
    graph_sha256 = awp.write_metis_graph(graph_path, graph, weights)
    input_serialization_seconds = time.perf_counter() - serialization_started

    preflight = {
        "status": (
            "PASS"
            if heavy_atom_ratio <= args.heavy_atom_fraction + 1e-12
            else "HEAVY_ATOM_GATE_FAIL"
        ),
        "heavy_atom_ratio": heavy_atom_ratio,
        "heavy_atom_fraction_limit": args.heavy_atom_fraction,
        "max_attraction_weight": weights.max_weight,
        "mean_partition_load": mean_partition_load,
        "remediation": (
            "increase upper representative count and rebuild the natural upper graph"
        ),
    }
    write_json_new(output_dir / "preflight.json", preflight)
    if preflight["status"] != "PASS":
        error = RuntimeError(
            "one upper node exceeds the frozen heavy-atom gate: "
            f"ratio={heavy_atom_ratio:.6f}, limit={args.heavy_atom_fraction:.6f}"
        )
        write_failure(
            output_dir,
            stage="heavy_atom_preflight",
            error=error,
            evidence=preflight,
        )
        raise error

    owner: tuple[int, ...] | None = None
    partitioner: dict[str, Any]
    validation: awp.PartitionValidation | None = None
    owner_path: Path | None = None
    initial_owner_path = (
        args.initial_owner_binary.expanduser().resolve(strict=True)
        if args.initial_owner_binary is not None
        else None
    )
    initial_owner = (
        read_owner_binary(initial_owner_path, len(labels), num_partitions)
        if initial_owner_path is not None
        else None
    )
    partition_started = time.perf_counter()
    try:
        if args.emit_only:
            partitioner = {
                "backend": "external",
                "status": "METIS_INPUT_READY",
                "seed": FIXED_METIS_SEED,
            }
        else:
            if args.partition_file is not None:
                source_partition_path = args.partition_file.expanduser().resolve(
                    strict=True
                )
                partition_path = bundle_partition_file(
                    source_partition_path, output_dir, num_partitions
                )
                partitioner = {
                    "backend": "precomputed_partition_file",
                    "source_path": str(source_partition_path),
                    "source_sha256": sha256_path(source_partition_path),
                    "path": str(partition_path),
                    "sha256": sha256_path(partition_path),
                }
            else:
                partition_path, partitioner = run_partitioner(
                    args.partitioner_backend,
                    graph_path,
                    adjacency,
                    graph,
                    weights,
                    initial_owner,
                    navigation_rows,
                    output_dir,
                    num_partitions,
                    binary=args.metis_binary,
                    imbalance_tolerance=args.imbalance_tolerance,
                )
                partitioner["partition_path"] = str(partition_path)
                partitioner["partition_sha256"] = sha256_path(partition_path)
            owner = read_partition(partition_path, len(labels))
            validation = awp.validate_partition(
                owner,
                weights,
                graph,
                num_partitions,
                imbalance_tolerance=args.imbalance_tolerance,
                heavy_atom_fraction=args.heavy_atom_fraction,
            )
            if args.partitioner_backend == "attraction-nav-refine":
                initial_loads = partitioner["initial_partition_loads"]
                initial_mean = sum(initial_loads) / len(initial_loads)
                initial_cv = (
                    sum((load - initial_mean) ** 2 for load in initial_loads)
                    / len(initial_loads)
                ) ** 0.5 / initial_mean
                final_cv = (
                    sum(
                        (load - validation.load_mean) ** 2
                        for load in validation.partition_loads
                    )
                    / len(validation.partition_loads)
                ) ** 0.5 / validation.load_mean
                improvement = {
                    "initial_load_max_over_mean": max(initial_loads) / initial_mean,
                    "final_load_max_over_mean": validation.load_max_over_mean,
                    "initial_load_cv": initial_cv,
                    "final_load_cv": final_cv,
                    "max_over_mean_improves": (
                        validation.load_max_over_mean
                        < max(initial_loads) / initial_mean
                    ),
                    "cv_improves": final_cv < initial_cv,
                }
                improvement["pass"] = bool(
                    improvement["max_over_mean_improves"]
                    and improvement["cv_improves"]
                )
                partitioner["predicted_attraction_balance_gate"] = improvement
                partition_pass = bool(
                    validation.heavy_atom_pass and improvement["pass"]
                )
            else:
                partition_pass = bool(
                    validation.heavy_atom_pass and validation.balance_pass
                )
            if not partition_pass:
                raise RuntimeError(
                    "partition failed the frozen attraction-balance gates: "
                    + json.dumps(
                        {
                            "validation": asdict(validation),
                            "partitioner": partitioner,
                        },
                        sort_keys=True,
                    )
                )
            owner_path = output_dir / OWNER_NAME
            with owner_path.open("xb") as handle:
                for partition in owner:
                    handle.write(struct.pack("<i", partition))
    except Exception as error:
        write_failure(
            output_dir,
            stage="partition_generation_or_validation",
            error=error,
            evidence={
                "num_partitions": num_partitions,
                "imbalance_tolerance": args.imbalance_tolerance,
                "heavy_atom_fraction": args.heavy_atom_fraction,
            },
        )
        raise
    partition_seconds = time.perf_counter() - partition_started

    manifest = {
        "format_version": awp.FORMAT_VERSION,
        "record_type": "orion_attraction_weighted_search_graph_owner",
        "status": "METIS_INPUT_READY" if owner is None else "PASS",
        "tool": TOOL_PATH,
        "contract": {
            "upper_hnsw_mutated": False,
            "upper_graph_rebuilt_by_partitioner": False,
            "live_shard_load_read": False,
            "lower_hnsw_built_before_owner_freeze": False,
            "calibration_uses_production_upper_navigator": True,
            "calibration_search_ef_equals_placement_search_ef": True,
            "edge_weights_add_no_non_hnsw_edges": (
                args.edge_mode != awp.TOP1_STAR_EDGE_MODE
            ),
            "search_induced_overlay_edges_explicit": (
                args.edge_mode == awp.TOP1_STAR_EDGE_MODE
            ),
            "logical_shards_separate_from_physical_workers": True,
            "strict_two_percent_balance_required": (
                args.partitioner_backend != "attraction-nav-refine"
            ),
            "navigation_refinement_requires_balance_improvement_only": (
                args.partitioner_backend == "attraction-nav-refine"
            ),
        },
        "parameters": {
            "num_partitions": num_partitions,
            "placement_search_ef": args.placement_search_ef,
            "estimator": args.estimator,
            "edge_mode": args.edge_mode,
            "imbalance_tolerance": args.imbalance_tolerance,
            "heavy_atom_fraction": args.heavy_atom_fraction,
            "partitioner_backend": args.partitioner_backend,
            "dirichlet_alpha": (
                awp.DIRICHLET_ALPHA
                if args.estimator == awp.SAMPLE_ESTIMATOR
                else None
            ),
        },
        "source": {
            "artifact": str(artifact_path),
            "artifact_sha256": artifact_sha256,
            "generation": artifact.get("generation"),
            "logical_point_count": logical_point_count,
            "upper_node_count": len(labels),
            "upper_graph_sha256": upper_graph_sha256,
            "calibration": calibration,
            "calibration_selection": calibration_selection,
            "upper_self_navigation": self_navigation,
            "initial_owner": (
                None
                if initial_owner_path is None
                else {
                    "path": str(initial_owner_path),
                    "sha256": sha256_path(initial_owner_path),
                    "encoding": "i32le",
                }
            ),
        },
        "measurements": {
            "row_count": measurements.row_count,
            "top_k": measurements.top_k,
            "edge_mode": measurements.edge_mode,
            "edge_cooccurrence_nonzero_edge_count": len(
                measurements.edge_cooccurrence
            ),
            "semantic_sha256": measurements.semantic_sha256,
            "first_hit_counts_path": str(counts_path),
            "first_hit_counts_sha256": sha256_path(counts_path),
        },
        "attraction_weights": {
            "estimator": weights.estimator,
            "calibration_count": weights.calibration_count,
            "logical_point_count": weights.logical_point_count,
            "total_weight": weights.total_weight,
            "max_weight": weights.max_weight,
            "mean_weight": weights.mean_weight,
            "semantic_sha256": weights.semantic_sha256,
            "path": str(weights_path),
            "sha256": sha256_path(weights_path),
        },
        "search_weighted_graph": {
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "raw_hnsw_edge_count": graph.raw_hnsw_edge_count,
            "search_overlay_edge_count": graph.search_overlay_edge_count,
            "total_edge_weight": graph.total_edge_weight,
            "edge_mode": graph.edge_mode,
            "semantic_sha256": graph.semantic_sha256,
            "metis_path": str(graph_path),
            "metis_sha256": graph_sha256,
        },
        "preflight": preflight,
        "performance": {
            "artifact_load_seconds": artifact_load_seconds,
            "contract_validation_seconds": contract_validation_seconds,
            "calibration_measurement_seconds": calibration_measurement_seconds,
            "attraction_estimation_seconds": attraction_estimation_seconds,
            "search_weighted_graph_seconds": search_weighted_graph_seconds,
            "input_serialization_seconds": input_serialization_seconds,
            "partition_seconds": partition_seconds,
            "total_before_manifest_seconds": time.perf_counter() - run_started,
        },
        "partitioner": partitioner,
        "source_code": {
            "tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_path(Path(__file__).resolve()),
            },
            "partitioner": {
                "path": str(Path(awp.__file__).resolve()),
                "sha256": sha256_path(Path(awp.__file__).resolve()),
            },
        },
        "owner": (
            None
            if owner_path is None or validation is None
            else {
                "path": str(owner_path),
                "sha256": sha256_path(owner_path),
                "encoding": "i32le",
                "semantic_sha256": validation.semantic_sha256,
                "validation": asdict(validation),
            }
        ),
    }
    manifest_path = output_dir / MANIFEST_NAME
    write_json_new(manifest_path, manifest)
    write_checksum_listing(output_dir)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    manifest = run(parse_args(argv))
    print(json.dumps({"manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
