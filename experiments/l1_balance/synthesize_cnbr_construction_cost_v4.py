#!/usr/bin/env python3
"""Freeze a fail-closed CNBR construction-cost audit from canonical inputs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


FORMAT_VERSION = 1
RATIO_LIMIT = 0.05
AUDIT_NAME = "construction-cost-audit.json"
SOURCE_RECORD_NAME = "construction-cost-source-code.record.json"
SOURCE_COPY_NAME = "synthesize_cnbr_construction_cost_v4.py"
MEASUREMENT_AUDIT_NAME = "fresh-exporter-measurement-audit.json"
MEASUREMENT_SOURCE_RECORD_NAME = "fresh-measurement-source-code.record.json"
MEASUREMENT_STATUS_NAME = "STATUS.md"
MEASUREMENT_STATUS = "COMPLETE_FRESH_PROJECTED_EXPORTER_MEASUREMENT"
CANONICAL_BENCHMARK_LOCK = Path(
    "/proj/intelisys-PG0/exp/orion-distributed/c1-20260821-v2/benchmark.lock"
)
CPU_SET = "8-15,24-31"
TIME_PATH = Path("/usr/bin/time")
TASKSET_PATH = Path("/usr/bin/taskset")
RELEASE_EXPORTER_SUFFIX = Path("release/examples/orion_export_upper_hits")
CANONICAL_EXPORTER_SHA256 = (
    "45b9d5708ef3b93a001550aba89b9d672027d35446d3d8300288ee4ef1a7ce7f"
)
SOURCE_BUILD_MANIFESTS = {
    "sift": {
        "path": Path(
            "/proj/intelisys-PG0/exp/orion-distributed/l1-balance-20260826/"
            "sift-p32/source-build-replay-v1/build-manifest.json"
        ),
        "sha256": (
            "b6e1ef687b9b0617db0545c6395897ea8fc2865fbb62c0af0611f09492b2dfed"
        ),
    },
    "glove": {
        "path": Path(
            "/proj/intelisys-PG0/exp/orion-distributed/"
            "orion-logical-balance-20260825/production-bundles/"
            "glove-ccnb-p32-single-u48-b50-f14-g3253248/build-manifest.json"
        ),
        "sha256": (
            "8859ea9ee8994480624c3bdad4cdd448befded0013c593fb437a0e5660c086fa"
        ),
    },
}

SEMANTIC_RECORDS: dict[str, dict[str, Any]] = {
    "sift-self": {
        "hits_sha256": (
            "111b16f9c87358ee4ee494ccaa688b5fb590560ac9e6c6a9786faeb08b1ad17f"
        ),
        "rows": 31_250,
        "dimension": 128,
    },
    "sift-full": {
        "hits_sha256": (
            "465e7bd79ea20bf5770c1204815f92d00fe3304111e8386f22e0a1bcf7d80472"
        ),
        "rows": 1_000_000,
        "dimension": 128,
    },
    "glove-self": {
        "hits_sha256": (
            "8fa23af17dfeed7021ef61a2a992155fa966a5e1645570873548e2476939c856"
        ),
        "rows": 36_984,
        "dimension": 200,
    },
    "glove-full": {
        "hits_sha256": (
            "3f397cee3f9f384d1b5aa211a1f220702838d71fe9dfa015484f3d12f6869f52"
        ),
        "rows": 1_183_514,
        "dimension": 200,
    },
}

DATASETS = {
    "sift1m": {"self": "sift-self", "full": "sift-full"},
    "glove-200-angular": {"self": "glove-self", "full": "glove-full"},
}

OLD_PHASE_A_V1_SOURCE_SHA256 = {
    "generator": (
        "da88cbd9e341feb5ed877124439339285c71fbe57258bea148e542468e59e890"
    ),
    "core": "3b007a2ffaf0c3a1498980fa476896ab52ea75b736422c86528d797b1840972a",
}

PHASE_A_STAGE = "upper_only_n_native_cnbr_owners_frozen"
PHASE_B_STAGE = "post_freeze_evaluation"
PHASE_B_COMPARISON = "N_native_vs_C_CNBR"
OWNER_NAMES = {"N_native", "C_CNBR"}
REQUIRED_VALIDATION_WALL_KEY = (
    "required_input_validation_and_mass_replay_wall_seconds"
)
REQUIRED_VALIDATION_CPU_KEY = (
    "required_input_validation_and_mass_replay_process_cpu_seconds"
)
LEGACY_VALIDATION_KEYS = {
    "self_navigation_input_validation_and_mass_replay_wall_seconds",
    "self_navigation_input_validation_and_mass_replay_process_cpu_seconds",
}
PHASE_A_TIMING_CONTRACT = {
    "contract_id": "cnbr-integrated-incremental-cost-v1",
    "cost_gate_includes": [
        "external_upper_self_navigation",
        "required_input_validation_and_mass_replay",
        "formal_cnbr",
    ],
    "required_input_validation_start": (
        "after shared upper artifact deserialization and graph normalization"
    ),
    "required_input_validation_includes": [
        "projected_artifact_sha256",
        "upper_only_projection_provenance_and_redaction",
        "upper_input_manifest_and_exact_byte_parity",
        "self_navigation_manifest_and_row_validation",
        "proxy_mass_replay_and_duplicate_tie_proof",
    ],
    "reported_but_excluded_from_incremental_cost_gate": [
        "evidence_preflight",
        "shared_upper_input_materialization",
        "n_native_kmeans",
        "n_native_reference_topology_validation",
        "post_cnbr_source_stability_validation",
        "evidence_serialization_and_freeze",
    ],
}

TIME_PATTERNS = {
    "user_seconds": re.compile(r"^\s*User time \(seconds\):\s*(\S+)\s*$"),
    "system_seconds": re.compile(r"^\s*System time \(seconds\):\s*(\S+)\s*$"),
    "wall": re.compile(
        r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)\s*$"
    ),
    "maximum_resident_set_kbytes": re.compile(
        r"^\s*Maximum resident set size \(kbytes\):\s*(\S+)\s*$"
    ),
    "exit_status": re.compile(r"^\s*Exit status:\s*(\S+)\s*$"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-measurements-dir", required=True)
    parser.add_argument("--sift-phase-a-performance", required=True)
    parser.add_argument("--sift-phase-b-summary", required=True)
    parser.add_argument("--glove-phase-a-performance", required=True)
    parser.add_argument("--glove-phase-b-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def parse_measure_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure fresh projected-artifact self/full exporter construction cost"
        )
    )
    parser.add_argument("--exporter", required=True)
    parser.add_argument(
        "--benchmark-lock", default=str(CANONICAL_BENCHMARK_LOCK)
    )
    parser.add_argument("--sift-artifact", required=True)
    parser.add_argument("--sift-projection-manifest", required=True)
    parser.add_argument("--sift-self-vectors", required=True)
    parser.add_argument("--sift-full-vectors", required=True)
    parser.add_argument("--glove-artifact", required=True)
    parser.add_argument("--glove-projection-manifest", required=True)
    parser.add_argument("--glove-self-vectors", required=True)
    parser.add_argument("--glove-full-vectors", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _measurement_status_text(audit_sha256: str) -> str:
    return (
        "# Fresh projected-artifact CNBR exporter measurements\n\n"
        f"Status: `{MEASUREMENT_STATUS}`\n\n"
        f"Audit SHA-256: `{audit_sha256}`\n\n"
        "Old v3 timing is not reused; its hit hashes are used only for "
        "semantic byte-parity checks.\n"
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from error


def _absolute_file(raw: str, label: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _absolute_directory(raw: str, label: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _output_directory(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError("output-dir must be an absolute path")
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {path}")
    return path


def _require_read_only(path: Path, label: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o222:
        raise ValueError(f"{label} must be read-only: {path} has mode {mode:04o}")


def _number(value: Any, label: str, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a non-negative number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{label} must be a non-negative integer")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return value


def _single_sha256_token(path: Path, label: str) -> str:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{label} is not valid ASCII: {path}") from error
    if len(lines) != 1:
        raise ValueError(f"{label} must contain exactly one checksum line")
    fields = lines[0].split()
    if not fields or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
        raise ValueError(f"{label} does not start with a lowercase SHA-256")
    return fields[0]


def _resolve_relative_file(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ValueError(f"{label} must be a non-empty relative path")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its frozen output directory") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _aggregator_source_path() -> Path:
    return Path(__file__).resolve()


def _phase_a_source_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    return {
        "core": root / "native_cnbr_core.py",
        "generator": root / "prepare_native_cnbr_phase_a.py",
        "upper_only_projection": root / "project_upper_only_artifact.py",
        "protocol_addendum_json": root / "CNBR_CORRECTION_ADDENDUM.json",
        "protocol_addendum_markdown": root / "CNBR_CORRECTION_ADDENDUM.md",
    }


def _phase_b_source_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    return {
        "evaluator": root / "evaluate_cnbr_frozen_owners.py",
        "offline_screen": root / "run_offline_screen.py",
        "phase_a_generator": root / "prepare_native_cnbr_phase_a.py",
        "phase_a_core": root / "native_cnbr_core.py",
        "source_binding": root / "freeze_phase_b_source_binding.py",
    }


def _capture_file(path: Path, label: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"current source {label} does not exist: {path}")
    data = path.read_bytes()
    return {
        "path": path,
        "bytes": data,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _capture_current_sources() -> dict[str, Any]:
    phase_a_paths = {key: value.resolve() for key, value in _phase_a_source_paths().items()}
    phase_b_paths = {key: value.resolve() for key, value in _phase_b_source_paths().items()}
    return {
        "aggregator": _capture_file(_aggregator_source_path(), "aggregator"),
        "phase_a": {
            key: _capture_file(path, f"Phase-A {key}")
            for key, path in phase_a_paths.items()
        },
        "phase_b": {
            key: _capture_file(path, f"Phase-B {key}")
            for key, path in phase_b_paths.items()
        },
    }


def _verify_current_sources(snapshot: dict[str, Any]) -> None:
    current_paths: list[tuple[str, Path, dict[str, Any]]] = [
        ("aggregator", _aggregator_source_path().resolve(), snapshot["aggregator"])
    ]
    current_paths.extend(
        (f"Phase-A {key}", path.resolve(), snapshot["phase_a"][key])
        for key, path in _phase_a_source_paths().items()
    )
    current_paths.extend(
        (f"Phase-B {key}", path.resolve(), snapshot["phase_b"][key])
        for key, path in _phase_b_source_paths().items()
    )
    for label, path, record in current_paths:
        if (
            path != record["path"]
            or not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or sha256_path(path) != record["sha256"]
        ):
            raise ValueError(f"{label} source changed during aggregation")


def _verify_file_snapshot(record: dict[str, Any], label: str) -> None:
    path = Path(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or sha256_path(path) != record["sha256"]
    ):
        raise ValueError(f"{label} changed during measurement")


def _parse_elapsed(raw: str, label: str) -> float:
    parts = raw.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"{label} has unsupported elapsed time {raw!r}")
    try:
        values = [float(value) for value in parts]
    except ValueError as error:
        raise ValueError(f"{label} has invalid elapsed time {raw!r}") from error
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"{label} has invalid elapsed time {raw!r}")
    if values[-1] >= 60 or (len(values) == 3 and values[-2] >= 60):
        raise ValueError(f"{label} has out-of-range elapsed time {raw!r}")
    if len(values) == 2:
        return values[0] * 60 + values[1]
    return values[0] * 3600 + values[1] * 60 + values[2]


def _parse_time_file(path: Path, label: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8: {path}") from error
    found: dict[str, list[str]] = {key: [] for key in TIME_PATTERNS}
    for line in lines:
        for key, pattern in TIME_PATTERNS.items():
            match = pattern.fullmatch(line)
            if match is not None:
                found[key].append(match.group(1))
    duplicates = {key: values for key, values in found.items() if len(values) != 1}
    if duplicates:
        raise ValueError(f"{label} lacks exactly one required /usr/bin/time field")
    try:
        user_seconds = float(found["user_seconds"][0])
        system_seconds = float(found["system_seconds"][0])
        maximum_rss = int(found["maximum_resident_set_kbytes"][0])
        exit_status = int(found["exit_status"][0])
    except ValueError as error:
        raise ValueError(f"{label} contains a malformed /usr/bin/time value") from error
    _number(user_seconds, f"{label} user time")
    _number(system_seconds, f"{label} system time")
    _number(maximum_rss, f"{label} maximum RSS", integer=True)
    if exit_status != 0:
        raise ValueError(f"{label} exporter exit status is not zero: {exit_status}")
    wall_seconds = _parse_elapsed(found["wall"][0], label)
    return {
        "external_wall_seconds": wall_seconds,
        "user_seconds": user_seconds,
        "system_seconds": system_seconds,
        "process_cpu_seconds": user_seconds + system_seconds,
        "maximum_resident_set_kbytes": maximum_rss,
        "exit_status": exit_status,
    }


def _require_mode_0444(path: Path, label: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o444:
        raise ValueError(f"{label} must have mode 0444: {path} has mode {mode:04o}")


def _file_binding(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _external_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_external_binding(record: Any, label: str) -> Path:
    value = _mapping(record, label)
    path = _absolute_file(value.get("path"), f"{label}.path")
    if (
        value.get("sha256") != sha256_path(path)
        or value.get("size_bytes") != path.stat().st_size
    ):
        raise ValueError(f"{label} checksum binding drifted")
    return path


def _validate_projection_manifest(
    manifest_path: Path, artifact_path: Path, label: str
) -> tuple[str, dict[str, Any]]:
    _require_read_only(manifest_path, label)
    sidecar_path = manifest_path.with_name(manifest_path.name + ".sha256")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"{label} sidecar is missing: {sidecar_path}")
    _require_read_only(sidecar_path, f"{label} sidecar")
    digest = sha256_path(manifest_path)
    if _single_sha256_token(sidecar_path, f"{label} sidecar") != digest:
        raise ValueError(f"{label} checksum binding failed")
    value = _mapping(_load_json(manifest_path, label), label)
    if (
        value.get("format_version") != FORMAT_VERSION
        or value.get("record_type") != "orion_upper_only_projection"
    ):
        raise ValueError(f"{label} is not an upper-only projection record")
    output = _mapping(value.get("output_artifact"), f"{label}.output_artifact")
    _require_same_path(output.get("path"), artifact_path, f"{label} artifact path")
    if (
        output.get("sha256") != sha256_path(artifact_path)
        or output.get("size_bytes") != artifact_path.stat().st_size
    ):
        raise ValueError(f"{label} projected artifact checksum binding failed")
    artifact_sidecar = artifact_path.with_name(artifact_path.name + ".sha256")
    _require_same_path(
        output.get("sha256_sidecar"),
        artifact_sidecar,
        f"{label} projected artifact sidecar path",
    )
    if not artifact_sidecar.is_file():
        raise FileNotFoundError(
            f"{label} projected artifact sidecar is missing: {artifact_sidecar}"
        )
    _require_read_only(artifact_sidecar, f"{label} projected artifact sidecar")
    if _single_sha256_token(
        artifact_sidecar, f"{label} projected artifact sidecar"
    ) != output.get("sha256"):
        raise ValueError(f"{label} projected artifact sidecar checksum drifted")
    contract = _mapping(value.get("contract"), f"{label}.contract")
    proof = _mapping(value.get("redaction_proof"), f"{label}.redaction_proof")
    if (
        contract.get("historical_memberships_removed") is not True
        or contract.get("historical_layout_metadata_removed") is not True
        or proof.get("every_output_membership_is_exact_neutral_sentinel")
        is not True
        or proof.get("historical_membership_values_emitted") is not False
        or proof.get("historical_membership_distribution_emitted") is not False
    ):
        raise ValueError(f"{label} does not prove neutral upper-only projection")
    projection_source = _mapping(
        value.get("projection_source"), f"{label}.projection_source"
    )
    current_projection_source = _phase_a_source_paths()[
        "upper_only_projection"
    ].resolve()
    _require_same_path(
        projection_source.get("path"),
        current_projection_source,
        f"{label} projection source path",
    )
    if (
        projection_source.get("sha256") != sha256_path(current_projection_source)
        or projection_source.get("size_bytes")
        != current_projection_source.stat().st_size
    ):
        raise ValueError(f"{label} projection source is not current")
    provenance = _mapping(
        value.get("upper_build_provenance"), f"{label}.upper_build_provenance"
    )
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("source_manifest_sha256")))
        is None
        or not isinstance(provenance.get("source_manifest_size_bytes"), int)
        or provenance["source_manifest_size_bytes"] <= 0
    ):
        raise ValueError(f"{label} lacks checksum-bound source-build provenance")
    parameters = _mapping(
        provenance.get("parameters"), f"{label}.upper_build_provenance.parameters"
    )
    expected_parameters = {
        "upper_sample_seed",
        "upper_m",
        "upper_ef_construction",
        "upper_graph_seed",
    }
    if set(parameters) != expected_parameters or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in parameters.values()
    ):
        raise ValueError(f"{label} source-build parameter binding drifted")
    return digest, value


def _validate_neutral_projected_artifact(
    artifact_path: Path, expected: dict[str, Any], label: str
) -> None:
    artifact = _mapping(_load_json(artifact_path, label), label)
    nodes = _sequence(artifact.get("upper_nodes"), f"{label}.upper_nodes")
    schema = _mapping(artifact.get("vector_schema"), f"{label}.vector_schema")
    if (
        artifact.get("layout_sha256") != "0" * 64
        or artifact.get("physical_point_count")
        != artifact.get("logical_point_count")
        or len(nodes) != expected["rows"]
        or schema.get("dimension") != expected["dimension"]
        or any(
            not isinstance(node, dict) or node.get("shard_membership") != [0]
            for node in nodes
        )
    ):
        raise ValueError(f"{label} is not the required neutral projected artifact")


def _validate_source_build_lineage(
    source_build_path: Path,
    projection_value: dict[str, Any],
    full_vectors_path: Path,
    label: str,
) -> dict[str, Any]:
    source_build = _mapping(_load_json(source_build_path, label), label)
    dataset = _mapping(source_build.get("dataset"), f"{label}.dataset")
    outputs = _mapping(source_build.get("outputs"), f"{label}.outputs")
    files = _mapping(outputs.get("files"), f"{label}.outputs.files")
    production_name = outputs.get("production_artifact")
    if not isinstance(production_name, str) or production_name not in files:
        raise ValueError(f"{label} does not name a production artifact output")
    production = _mapping(
        files[production_name], f"{label}.outputs.files.{production_name}"
    )
    projection_source = _mapping(
        projection_value.get("source_artifact"),
        f"{label} projection source artifact",
    )
    if (
        projection_source.get("sha256") != production.get("sha256")
        or projection_source.get("size_bytes") != production.get("size_bytes")
    ):
        raise ValueError(
            f"{label} projection source is not the source-build production artifact"
        )
    vector_name = full_vectors_path.name
    if vector_name not in files:
        raise ValueError(f"{label} does not bind normalized full vectors {vector_name}")
    vector_output = _mapping(
        files[vector_name], f"{label}.outputs.files.{vector_name}"
    )
    if (
        vector_output.get("sha256") != sha256_path(full_vectors_path)
        or vector_output.get("size_bytes") != full_vectors_path.stat().st_size
    ):
        raise ValueError(f"{label} normalized full-vector checksum binding drifted")
    if "path" in vector_output:
        _require_same_path(
            vector_output.get("path"),
            full_vectors_path,
            f"{label} normalized full-vector path",
        )
    dataset_sha256 = dataset.get("sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(dataset_sha256)) is None:
        raise ValueError(f"{label} lacks a checksum-bound dataset identity")
    return {
        "dataset_sha256": dataset_sha256,
        "dataset_dimension": dataset.get("dimension"),
        "dataset_train_rows_used": dataset.get("train_rows_used"),
        "production_artifact_output": production_name,
        "production_artifact_sha256": production.get("sha256"),
        "normalized_full_vectors_output": vector_name,
        "normalized_full_vectors_sha256": vector_output.get("sha256"),
    }


def _measurement_input_records(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    dataset_inputs = {
        "sift": {
            "artifact": args.sift_artifact,
            "projection_manifest": args.sift_projection_manifest,
            "self_vectors": args.sift_self_vectors,
            "full_vectors": args.sift_full_vectors,
        },
        "glove": {
            "artifact": args.glove_artifact,
            "projection_manifest": args.glove_projection_manifest,
            "self_vectors": args.glove_self_vectors,
            "full_vectors": args.glove_full_vectors,
        },
    }
    records: dict[str, dict[str, Any]] = {}
    for prefix, raw in dataset_inputs.items():
        artifact = _absolute_file(raw["artifact"], f"{prefix} projected artifact")
        projection = _absolute_file(
            raw["projection_manifest"], f"{prefix} projection manifest"
        )
        projection_sha256, projection_value = _validate_projection_manifest(
            projection, artifact, f"{prefix} projection manifest"
        )
        expected_source_build = SOURCE_BUILD_MANIFESTS[prefix]
        source_build_path = _absolute_file(
            str(expected_source_build["path"]),
            f"{prefix} canonical source-build manifest",
        )
        source_build_sha256 = sha256_path(source_build_path)
        if source_build_sha256 != expected_source_build["sha256"]:
            raise ValueError(f"{prefix} canonical source-build manifest drifted")
        projection_provenance = _mapping(
            projection_value.get("upper_build_provenance"),
            f"{prefix} projection upper-build provenance",
        )
        if (
            projection_provenance.get("source_manifest_sha256")
            != source_build_sha256
            or projection_provenance.get("source_manifest_size_bytes")
            != source_build_path.stat().st_size
        ):
            raise ValueError(
                f"{prefix} projection does not bind the canonical source-build manifest"
            )
        full_vectors_path = _absolute_file(
            raw["full_vectors"], f"{prefix} full vectors"
        )
        source_build_lineage = _validate_source_build_lineage(
            source_build_path,
            projection_value,
            full_vectors_path,
            f"{prefix} canonical source-build manifest",
        )
        self_expected = SEMANTIC_RECORDS[f"{prefix}-self"]
        full_expected = SEMANTIC_RECORDS[f"{prefix}-full"]
        _validate_neutral_projected_artifact(
            artifact, self_expected, f"{prefix} projected artifact"
        )
        if self_expected["dimension"] != full_expected["dimension"]:
            raise ValueError(f"{prefix} self/full dimension contract drifted")
        for scope, expected in (("self", self_expected), ("full", full_expected)):
            vectors = _absolute_file(
                raw[f"{scope}_vectors"], f"{prefix} {scope} vectors"
            )
            expected_size = expected["rows"] * expected["dimension"] * 4
            if vectors.stat().st_size != expected_size:
                raise ValueError(
                    f"{prefix} {scope} vector size differs from rows x dimension x f32"
                )
            records[f"{prefix}-{scope}"] = {
                "artifact": _external_binding(artifact),
                "projection_manifest": {
                    **_external_binding(projection),
                    "sha256": projection_sha256,
                },
                "source_build_manifest": _external_binding(source_build_path),
                "source_build_lineage": source_build_lineage,
                "vectors": _external_binding(vectors),
            }
    return records


def _measurement_command(
    exporter: Path,
    artifact: Path,
    vectors: Path,
    expected: dict[str, Any],
    hits_path: Path,
    manifest_path: Path,
    time_path: Path,
) -> list[str]:
    return [
        str(TIME_PATH),
        "-v",
        "-o",
        str(time_path),
        str(TASKSET_PATH),
        "-c",
        CPU_SET,
        str(exporter),
        str(artifact),
        str(vectors),
        str(expected["rows"]),
        str(expected["dimension"]),
        "10",
        "100",
        str(hits_path),
        str(manifest_path),
    ]


def _run_measurement_record(
    root: Path,
    record_name: str,
    exporter: Path,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    expected = SEMANTIC_RECORDS[record_name]
    artifact = Path(inputs["artifact"]["path"])
    vectors = Path(inputs["vectors"]["path"])
    hits_path = root / f"{record_name}.hits.u64le"
    manifest_path = root / f"{record_name}.manifest.json"
    time_path = root / f"{record_name}.time.txt"
    stdout_path = root / f"{record_name}.stdout.log"
    stderr_path = root / f"{record_name}.stderr.log"
    hits_sidecar = root / f"{record_name}.hits.sha256"
    command = _measurement_command(
        exporter,
        artifact,
        vectors,
        expected,
        hits_path,
        manifest_path,
        time_path,
    )
    with stdout_path.open("xb") as stdout_handle, stderr_path.open(
        "xb"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"fresh projected exporter {record_name} failed with exit "
            f"status {completed.returncode}"
        )
    for path, label in (
        (hits_path, "hits"),
        (manifest_path, "manifest"),
        (time_path, "time record"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{record_name} exporter did not create {label}")
    timing = _parse_time_file(time_path, f"fresh {record_name} time file")
    hits_sha256 = sha256_path(hits_path)
    hits_size_bytes = hits_path.stat().st_size
    expected_size = expected["rows"] * 10 * 8
    if (
        hits_sha256 != expected["hits_sha256"]
        or hits_size_bytes != expected_size
    ):
        raise ValueError(
            f"fresh {record_name} hits differ from frozen semantic checksum/size"
        )
    manifest = _mapping(
        _load_json(manifest_path, f"fresh {record_name} manifest"),
        f"fresh {record_name} manifest",
    )
    expected_manifest = {
        "artifact_sha256": inputs["artifact"]["sha256"],
        "vectors_sha256": inputs["vectors"]["sha256"],
        "row_count": expected["rows"],
        "dimension": expected["dimension"],
        "top_k": 10,
        "search_ef": 100,
        "hits_sha256": hits_sha256,
        "hits_size_bytes": hits_size_bytes,
    }
    mismatches = {
        key: {"observed": manifest.get(key), "expected": value}
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"fresh {record_name} manifest contract drifted: {mismatches}")
    _require_same_path(
        manifest.get("artifact_path"), artifact, f"fresh {record_name} artifact path"
    )
    _require_same_path(
        manifest.get("vectors_path"), vectors, f"fresh {record_name} vector path"
    )
    _require_same_path(
        manifest.get("hits_path"), hits_path, f"fresh {record_name} hits path"
    )
    hits_sidecar.write_text(
        f"{hits_sha256}  {hits_path}\n", encoding="ascii"
    )
    return {
        "artifact": inputs["artifact"],
        "projection_manifest": inputs["projection_manifest"],
        "source_build_manifest": inputs["source_build_manifest"],
        "source_build_lineage": inputs["source_build_lineage"],
        "vectors": inputs["vectors"],
        "parameters": {
            "rows": expected["rows"],
            "dimension": expected["dimension"],
            "top_k": 10,
            "search_ef": 100,
        },
        "command": command,
        "timing": timing,
        "semantic_hits_sha256": expected["hits_sha256"],
        "files": {
            "hits": _file_binding(root, hits_path),
            "manifest": _file_binding(root, manifest_path),
            "time": _file_binding(root, time_path),
            "hits_sha256": _file_binding(root, hits_sidecar),
            "stdout": _file_binding(root, stdout_path),
            "stderr": _file_binding(root, stderr_path),
        },
    }


def _verify_external_inputs_unchanged(
    exporter: dict[str, Any],
    time_binary: dict[str, Any],
    taskset_binary: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> None:
    _validate_external_binding(exporter, "release exporter")
    _validate_external_binding(time_binary, "canonical /usr/bin/time")
    _validate_external_binding(taskset_binary, "canonical taskset")
    for record_name, record in records.items():
        for key in (
            "artifact",
            "projection_manifest",
            "source_build_manifest",
            "vectors",
        ):
            _validate_external_binding(
                record[key], f"{record_name} measurement {key}"
            )


def run_fresh_measurements(args: argparse.Namespace) -> tuple[Path, str]:
    output_dir = _output_directory(args.output_dir)
    runner_source = _capture_file(
        _aggregator_source_path(), "fresh measurement runner"
    )
    exporter = _absolute_file(args.exporter, "release exporter")
    if tuple(exporter.parts[-3:]) != tuple(RELEASE_EXPORTER_SUFFIX.parts):
        raise ValueError("exporter must be the release/examples/orion_export_upper_hits binary")
    if not os.access(exporter, os.X_OK):
        raise ValueError(f"release exporter is not executable: {exporter}")
    exporter_record = _external_binding(exporter)
    if exporter_record["sha256"] != CANONICAL_EXPORTER_SHA256:
        raise ValueError("release exporter checksum is not canonical")
    benchmark_lock = _absolute_file(args.benchmark_lock, "benchmark lock")
    if benchmark_lock != CANONICAL_BENCHMARK_LOCK.resolve():
        raise ValueError("benchmark-lock is not the canonical lock path")
    if not TIME_PATH.is_file() or not TASKSET_PATH.is_file():
        raise FileNotFoundError("canonical /usr/bin/time or taskset is unavailable")
    if not os.access(TIME_PATH, os.X_OK) or not os.access(TASKSET_PATH, os.X_OK):
        raise ValueError("canonical /usr/bin/time or taskset is not executable")
    time_binary_record = _external_binding(TIME_PATH)
    taskset_binary_record = _external_binding(TASKSET_PATH)
    input_records = _measurement_input_records(args)
    serial_order = ["sift-self", "sift-full", "glove-self", "glove-full"]

    with benchmark_lock.open("rb") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("canonical benchmark lock is already held") from error
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite output directory: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=False)
        source_dir = output_dir / "source"
        source_dir.mkdir()
        measurements: dict[str, dict[str, Any]] = {}
        for record_name in serial_order:
            measurements[record_name] = _run_measurement_record(
                output_dir,
                record_name,
                exporter,
                input_records[record_name],
            )

        _verify_external_inputs_unchanged(
            exporter_record,
            time_binary_record,
            taskset_binary_record,
            input_records,
        )
        _verify_file_snapshot(runner_source, "fresh measurement runner source")
        source_copy_path = source_dir / SOURCE_COPY_NAME
        source_copy_path.write_bytes(runner_source["bytes"])
        measurement_source_record = {
            "format_version": FORMAT_VERSION,
            "record_type": "cnbr_fresh_measurement_runner_source_code",
            "files": {
                "runner": {
                    "path": source_copy_path.relative_to(output_dir).as_posix(),
                    "sha256": runner_source["sha256"],
                    "size_bytes": runner_source["size_bytes"],
                }
            },
        }
        measurement_source_record_path = (
            output_dir / MEASUREMENT_SOURCE_RECORD_NAME
        )
        measurement_source_record_path.write_bytes(
            _json_bytes(measurement_source_record)
        )
        source_record_binding = _file_binding(
            output_dir, measurement_source_record_path
        )
        audit = {
            "format_version": FORMAT_VERSION,
            "record_type": "cnbr_fresh_projected_exporter_measurements_v4",
            "status": MEASUREMENT_STATUS,
            "controls": {
                "benchmark_lock": str(benchmark_lock),
                "kernel_flock_held_during_all_measurements": True,
                "cpu_set": CPU_SET,
                "time_binary": time_binary_record,
                "taskset_binary": taskset_binary_record,
                "serial_order": serial_order,
                "release_exporter": exporter_record,
                "old_v3_timing_reused": False,
                "projected_artifact_measurement": True,
                "all_semantic_hits_match_frozen_checksums": True,
            },
            "semantic_reference": {
                "scope": "hits_bytes_only_not_timing_or_artifact_identity",
                "records": {
                    name: value["hits_sha256"]
                    for name, value in SEMANTIC_RECORDS.items()
                },
            },
            "runner_source_code": {
                "record": source_record_binding,
                "files": measurement_source_record["files"],
            },
            "records": measurements,
            "outputs": {
                "audit": MEASUREMENT_AUDIT_NAME,
                "audit_sidecar": f"{MEASUREMENT_AUDIT_NAME}.sha256",
                "source_record": MEASUREMENT_SOURCE_RECORD_NAME,
                "source_dir": "source",
                "status": MEASUREMENT_STATUS_NAME,
            },
        }
        audit_path = output_dir / MEASUREMENT_AUDIT_NAME
        audit_path.write_bytes(_json_bytes(audit))
        audit_sha256 = sha256_path(audit_path)
        sidecar_path = output_dir / f"{MEASUREMENT_AUDIT_NAME}.sha256"
        sidecar_path.write_text(audit_sha256 + "\n", encoding="ascii")
        status_path = output_dir / MEASUREMENT_STATUS_NAME
        status_path.write_text(
            _measurement_status_text(audit_sha256),
            encoding="utf-8",
        )
        _verify_external_inputs_unchanged(
            exporter_record,
            time_binary_record,
            taskset_binary_record,
            input_records,
        )
        _verify_file_snapshot(runner_source, "fresh measurement runner source")
        for path in output_dir.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o444)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    _validate_fresh_measurements(str(output_dir))
    return audit_path, audit_sha256


def _validated_internal_file(
    root: Path, record: Any, label: str
) -> Path:
    value = _mapping(record, label)
    path = _resolve_relative_file(root, value.get("path"), f"{label}.path")
    _require_mode_0444(path, label)
    if (
        value.get("sha256") != sha256_path(path)
        or value.get("size_bytes") != path.stat().st_size
    ):
        raise ValueError(f"{label} checksum binding drifted")
    return path


def _validate_fresh_measurements(raw: str) -> dict[str, Any]:
    root = _absolute_directory(raw, "fresh-measurements-dir")
    for path in root.rglob("*"):
        if path.is_file():
            _require_mode_0444(path, "fresh measurement output")
    audit_path = root / MEASUREMENT_AUDIT_NAME
    sidecar_path = root / f"{MEASUREMENT_AUDIT_NAME}.sha256"
    status_path = root / MEASUREMENT_STATUS_NAME
    if not audit_path.is_file() or not sidecar_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("fresh measurement audit, sidecar, or STATUS is missing")
    audit_sha256 = sha256_path(audit_path)
    if _single_sha256_token(sidecar_path, "fresh measurement audit sidecar") != audit_sha256:
        raise ValueError("fresh measurement audit checksum drifted")
    audit = _mapping(_load_json(audit_path, "fresh measurement audit"), "fresh measurement audit")
    if (
        audit.get("format_version") != FORMAT_VERSION
        or audit.get("record_type")
        != "cnbr_fresh_projected_exporter_measurements_v4"
        or audit.get("status") != MEASUREMENT_STATUS
    ):
        raise ValueError("fresh measurement audit status or schema drifted")
    expected_outputs = {
        "audit": MEASUREMENT_AUDIT_NAME,
        "audit_sidecar": f"{MEASUREMENT_AUDIT_NAME}.sha256",
        "source_record": MEASUREMENT_SOURCE_RECORD_NAME,
        "source_dir": "source",
        "status": MEASUREMENT_STATUS_NAME,
    }
    if audit.get("outputs") != expected_outputs:
        raise ValueError("fresh measurement output contract drifted")
    try:
        status_text = status_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError("fresh measurement STATUS is not valid UTF-8") from error
    if status_text != _measurement_status_text(audit_sha256):
        raise ValueError("fresh measurement STATUS content drifted")
    controls = _mapping(audit.get("controls"), "fresh measurement controls")
    expected_order = ["sift-self", "sift-full", "glove-self", "glove-full"]
    if (
        controls.get("benchmark_lock") != str(CANONICAL_BENCHMARK_LOCK.resolve())
        or controls.get("kernel_flock_held_during_all_measurements") is not True
        or controls.get("cpu_set") != CPU_SET
        or controls.get("serial_order") != expected_order
        or controls.get("old_v3_timing_reused") is not False
        or controls.get("projected_artifact_measurement") is not True
        or controls.get("all_semantic_hits_match_frozen_checksums") is not True
    ):
        raise ValueError("fresh measurement execution controls drifted")
    time_binary_record = _mapping(
        controls.get("time_binary"), "fresh measurement /usr/bin/time"
    )
    time_binary_path = _validate_external_binding(
        time_binary_record, "fresh measurement /usr/bin/time"
    )
    taskset_binary_record = _mapping(
        controls.get("taskset_binary"), "fresh measurement taskset"
    )
    taskset_binary_path = _validate_external_binding(
        taskset_binary_record, "fresh measurement taskset"
    )
    if (
        time_binary_path != TIME_PATH.resolve()
        or taskset_binary_path != TASKSET_PATH.resolve()
        or not os.access(time_binary_path, os.X_OK)
        or not os.access(taskset_binary_path, os.X_OK)
    ):
        raise ValueError("fresh measurement timing tool identity drifted")
    exporter_record = _mapping(
        controls.get("release_exporter"), "fresh measurement release exporter"
    )
    exporter_path = _validate_external_binding(
        exporter_record, "fresh measurement release exporter"
    )
    if (
        tuple(exporter_path.parts[-3:]) != tuple(RELEASE_EXPORTER_SUFFIX.parts)
        or not os.access(exporter_path, os.X_OK)
        or exporter_record.get("sha256") != CANONICAL_EXPORTER_SHA256
    ):
        raise ValueError("fresh measurement exporter is not the release exporter")

    runner_source = _mapping(
        audit.get("runner_source_code"), "fresh measurement runner source"
    )
    runner_files = _mapping(
        runner_source.get("files"), "fresh measurement runner source files"
    )
    if set(runner_files) != {"runner"}:
        raise ValueError("fresh measurement runner source membership drifted")
    runner_record = _mapping(
        runner_files["runner"], "fresh measurement runner source binding"
    )
    if runner_record.get("path") != f"source/{SOURCE_COPY_NAME}":
        raise ValueError("fresh measurement runner source path drifted")
    runner_copy_path = _validated_internal_file(
        root, runner_files["runner"], "fresh measurement runner source copy"
    )
    current_runner = _capture_file(
        _aggregator_source_path(), "current fresh measurement runner"
    )
    if (
        runner_record.get("sha256") != current_runner["sha256"]
        or runner_record.get("size_bytes") != current_runner["size_bytes"]
        or sha256_path(runner_copy_path) != current_runner["sha256"]
    ):
        raise ValueError(
            "fresh measurement runner source is not the current v4 source"
        )
    runner_source_record = _mapping(
        runner_source.get("record"), "fresh measurement runner source record binding"
    )
    if runner_source_record.get("path") != MEASUREMENT_SOURCE_RECORD_NAME:
        raise ValueError("fresh measurement runner source record path drifted")
    source_record_path = _validated_internal_file(
        root,
        runner_source_record,
        "fresh measurement runner source record",
    )
    source_record = _mapping(
        _load_json(source_record_path, "fresh measurement runner source record"),
        "fresh measurement runner source record",
    )
    if (
        source_record.get("format_version") != FORMAT_VERSION
        or source_record.get("record_type")
        != "cnbr_fresh_measurement_runner_source_code"
        or source_record.get("files") != runner_files
    ):
        raise ValueError("fresh measurement runner source record binding drifted")

    semantic_reference = _mapping(
        audit.get("semantic_reference"), "fresh measurement semantic reference"
    )
    if (
        semantic_reference.get("scope")
        != "hits_bytes_only_not_timing_or_artifact_identity"
        or semantic_reference.get("records")
        != {
            name: value["hits_sha256"]
            for name, value in SEMANTIC_RECORDS.items()
        }
    ):
        raise ValueError("fresh measurement semantic reference drifted")

    audit_records = _mapping(audit.get("records"), "fresh measurement records")
    if set(audit_records) != set(SEMANTIC_RECORDS):
        raise ValueError("fresh measurement record membership drifted")
    records: dict[str, dict[str, Any]] = {}
    for record_name, expected in SEMANTIC_RECORDS.items():
        record = _mapping(
            audit_records[record_name], f"fresh measurement {record_name}"
        )
        artifact = _validate_external_binding(
            record.get("artifact"), f"fresh measurement {record_name} artifact"
        )
        projection = _validate_external_binding(
            record.get("projection_manifest"),
            f"fresh measurement {record_name} projection manifest",
        )
        projection_sha256, projection_value = _validate_projection_manifest(
            projection,
            artifact,
            f"fresh measurement {record_name} projection manifest",
        )
        if record["projection_manifest"].get("sha256") != projection_sha256:
            raise ValueError(f"fresh measurement {record_name} projection SHA drifted")
        source_build_record = _mapping(
            record.get("source_build_manifest"),
            f"fresh measurement {record_name} source-build manifest",
        )
        source_build = _validate_external_binding(
            source_build_record,
            f"fresh measurement {record_name} source-build manifest",
        )
        prefix = record_name.split("-", 1)[0]
        expected_source_build = SOURCE_BUILD_MANIFESTS[prefix]
        if (
            source_build != Path(expected_source_build["path"]).resolve()
            or source_build_record.get("sha256")
            != expected_source_build["sha256"]
        ):
            raise ValueError(
                f"fresh measurement {record_name} source-build identity drifted"
            )
        projection_provenance = _mapping(
            projection_value.get("upper_build_provenance"),
            f"fresh measurement {record_name} projection provenance",
        )
        if (
            projection_provenance.get("source_manifest_sha256")
            != source_build_record["sha256"]
            or projection_provenance.get("source_manifest_size_bytes")
            != source_build_record["size_bytes"]
        ):
            raise ValueError(
                f"fresh measurement {record_name} projection/source-build binding drifted"
            )
        vectors = _validate_external_binding(
            record.get("vectors"), f"fresh measurement {record_name} vectors"
        )
        source_build_lineage = _mapping(
            record.get("source_build_lineage"),
            f"fresh measurement {record_name} source-build lineage",
        )
        if record_name.endswith("-full"):
            expected_lineage = _validate_source_build_lineage(
                source_build,
                projection_value,
                vectors,
                f"fresh measurement {record_name} source-build manifest",
            )
            if source_build_lineage != expected_lineage:
                raise ValueError(
                    f"fresh measurement {record_name} source-build lineage drifted"
                )
        parameters = _mapping(
            record.get("parameters"), f"fresh measurement {record_name} parameters"
        )
        expected_parameters = {
            "rows": expected["rows"],
            "dimension": expected["dimension"],
            "top_k": 10,
            "search_ef": 100,
        }
        if parameters != expected_parameters:
            raise ValueError(f"fresh measurement {record_name} parameters drifted")
        files = _mapping(
            record.get("files"), f"fresh measurement {record_name} files"
        )
        expected_file_names = {
            "hits",
            "manifest",
            "time",
            "hits_sha256",
            "stdout",
            "stderr",
        }
        if set(files) != expected_file_names:
            raise ValueError(f"fresh measurement {record_name} file membership drifted")
        expected_paths = {
            "hits": f"{record_name}.hits.u64le",
            "manifest": f"{record_name}.manifest.json",
            "time": f"{record_name}.time.txt",
            "hits_sha256": f"{record_name}.hits.sha256",
            "stdout": f"{record_name}.stdout.log",
            "stderr": f"{record_name}.stderr.log",
        }
        if any(
            not isinstance(files[key], dict)
            or files[key].get("path") != expected_path
            for key, expected_path in expected_paths.items()
        ):
            raise ValueError(
                f"fresh measurement {record_name} canonical file paths drifted"
            )
        paths = {
            key: _validated_internal_file(
                root, value, f"fresh measurement {record_name} {key}"
            )
            for key, value in files.items()
        }
        expected_command = _measurement_command(
            Path(exporter_record["path"]),
            artifact,
            vectors,
            expected,
            paths["hits"],
            paths["manifest"],
            paths["time"],
        )
        if record.get("command") != expected_command:
            raise ValueError(f"fresh measurement {record_name} command drifted")
        hits_sha256 = sha256_path(paths["hits"])
        hits_size_bytes = paths["hits"].stat().st_size
        if (
            hits_sha256 != expected["hits_sha256"]
            or hits_size_bytes != expected["rows"] * 10 * 8
            or record.get("semantic_hits_sha256") != expected["hits_sha256"]
            or paths["hits_sha256"].read_text(encoding="ascii")
            != f"{expected['hits_sha256']}  {paths['hits']}\n"
        ):
            raise ValueError(f"fresh measurement {record_name} semantic hits drifted")
        manifest = _mapping(
            _load_json(paths["manifest"], f"fresh measurement {record_name} manifest"),
            f"fresh measurement {record_name} manifest",
        )
        expected_manifest = {
            "artifact_sha256": record["artifact"]["sha256"],
            "vectors_sha256": record["vectors"]["sha256"],
            "row_count": expected["rows"],
            "dimension": expected["dimension"],
            "top_k": 10,
            "search_ef": 100,
            "hits_sha256": expected["hits_sha256"],
            "hits_size_bytes": expected["rows"] * 10 * 8,
        }
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            raise ValueError(f"fresh measurement {record_name} manifest contract drifted")
        _require_same_path(
            manifest.get("artifact_path"),
            artifact,
            f"fresh measurement {record_name} artifact path",
        )
        _require_same_path(
            manifest.get("vectors_path"),
            vectors,
            f"fresh measurement {record_name} vectors path",
        )
        _require_same_path(
            manifest.get("hits_path"),
            paths["hits"],
            f"fresh measurement {record_name} hits path",
        )
        timing = _parse_time_file(
            paths["time"], f"fresh measurement {record_name} time file"
        )
        if record.get("timing") != timing:
            raise ValueError(f"fresh measurement {record_name} timing copy drifted")
        records[record_name] = {
            **timing,
            "rows": expected["rows"],
            "dimension": expected["dimension"],
            "top_k": 10,
            "search_ef": 100,
            "hits_sha256": expected["hits_sha256"],
            "hits_size_bytes": expected["rows"] * 10 * 8,
            "artifact_path": str(artifact),
            "artifact_sha256": record["artifact"]["sha256"],
            "projection_manifest": str(projection),
            "projection_manifest_sha256": projection_sha256,
            "source_build_manifest": str(source_build),
            "source_build_manifest_sha256": source_build_record["sha256"],
            "source_build_lineage": source_build_lineage,
            "vectors_path": str(vectors),
            "vectors_sha256": record["vectors"]["sha256"],
            "time_file": str(paths["time"]),
            "time_file_sha256": files["time"]["sha256"],
            "manifest": str(paths["manifest"]),
            "manifest_sha256": files["manifest"]["sha256"],
            "hits_file": str(paths["hits"]),
            "hits_sha256_file": str(paths["hits_sha256"]),
        }
    for dataset, labels in DATASETS.items():
        if (
            records[labels["self"]]["artifact_sha256"]
            != records[labels["full"]]["artifact_sha256"]
        ):
            raise ValueError(f"fresh {dataset} self/full artifact identity differs")
        if (
            records[labels["self"]]["source_build_manifest_sha256"]
            != records[labels["full"]]["source_build_manifest_sha256"]
            or records[labels["self"]]["source_build_lineage"]
            != records[labels["full"]]["source_build_lineage"]
        ):
            raise ValueError(f"fresh {dataset} self/full source-build lineage differs")
    expected_files = {
        audit_path.resolve(),
        sidecar_path.resolve(),
        status_path.resolve(),
        source_record_path.resolve(),
        runner_copy_path.resolve(),
    }
    for record_name in SEMANTIC_RECORDS:
        files = _mapping(
            audit_records[record_name].get("files"),
            f"fresh measurement {record_name} files",
        )
        expected_files.update(
            _resolve_relative_file(
                root,
                record.get("path"),
                f"fresh measurement {record_name} output path",
            ).resolve()
            for record in files.values()
        )
    actual_files = {
        path.resolve() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("fresh measurement directory has extra or missing files")
    actual_directories = {
        path.resolve() for path in root.rglob("*") if path.is_dir()
    }
    if actual_directories != {(root / "source").resolve()}:
        raise ValueError("fresh measurement directory layout drifted")
    return {
        "root": root,
        "audit": str(audit_path),
        "audit_sha256": audit_sha256,
        "audit_status": audit["status"],
        "exporter": exporter_record,
        "records": records,
    }


def _source_record_matches(
    root: Path,
    record: Any,
    current: dict[str, Any],
    label: str,
) -> None:
    value = _mapping(record, label)
    if (
        value.get("sha256") != current["sha256"]
        or value.get("size_bytes") != current["size_bytes"]
    ):
        raise ValueError(f"{label} does not match the current repository source")
    frozen_path = _resolve_relative_file(root, value.get("path"), f"{label}.path")
    _require_read_only(frozen_path, label)
    if (
        frozen_path.stat().st_size != current["size_bytes"]
        or sha256_path(frozen_path) != current["sha256"]
    ):
        raise ValueError(f"{label} frozen source copy checksum drifted")


def _validate_phase_a(
    raw: str,
    dataset: str,
    fresh_self: dict[str, Any],
    current_sources: dict[str, Any],
) -> dict[str, Any]:
    performance_path = _absolute_file(raw, f"{dataset} Phase-A performance")
    if performance_path.name != "phase-a.performance.json":
        raise ValueError(f"{dataset} Phase-A performance has a non-canonical filename")
    _require_read_only(performance_path, f"{dataset} Phase-A performance")
    performance_sha256 = sha256_path(performance_path)
    performance = _mapping(
        _load_json(performance_path, f"{dataset} Phase-A performance"),
        f"{dataset} Phase-A performance",
    )
    if (
        performance.get("record_type") != "phase_a_construction_performance"
        or performance.get("candidate_set") != "formal"
    ):
        raise ValueError(f"{dataset} Phase-A performance is not a formal record")
    timing_contract = _mapping(
        performance.get("timing_contract"), f"{dataset} Phase-A timing contract"
    )
    if timing_contract != PHASE_A_TIMING_CONTRACT:
        raise ValueError(
            f"{dataset} Phase-A timing contract does not match "
            f"{PHASE_A_TIMING_CONTRACT['contract_id']}"
        )

    shared = _mapping(performance.get("shared"), f"{dataset} Phase-A shared timing")
    if shared.get("source_snapshot_verified_before_freeze") is not True:
        raise ValueError(f"{dataset} Phase-A source snapshot was not verified before freeze")
    source_snapshot = _mapping(
        shared.get("source_snapshot"), f"{dataset} Phase-A source snapshot"
    )
    if set(source_snapshot) != set(current_sources["phase_a"]):
        raise ValueError(f"{dataset} Phase-A source snapshot membership drifted")
    source_snapshot_records: dict[str, dict[str, Any]] = {}
    for key, current in current_sources["phase_a"].items():
        record = _mapping(
            source_snapshot.get(key), f"{dataset} Phase-A source snapshot {key}"
        )
        source_snapshot_records[key] = record
        if (
            record.get("sha256") != current["sha256"]
            or record.get("size_bytes") != current["size_bytes"]
        ):
            raise ValueError(
                f"{dataset} Phase-A source snapshot {key} is not current"
            )
    if (
        source_snapshot_records["generator"].get("sha256")
        == OLD_PHASE_A_V1_SOURCE_SHA256["generator"]
        and source_snapshot_records["core"].get("sha256")
        == OLD_PHASE_A_V1_SOURCE_SHA256["core"]
    ):
        raise ValueError(f"{dataset} old Phase-A v1 timing/source is not admissible")

    legacy_validation_keys = sorted(LEGACY_VALIDATION_KEYS.intersection(shared))
    if legacy_validation_keys:
        raise ValueError(
            f"{dataset} Phase-A contains inadmissible legacy self-navigation "
            f"timing keys: {legacy_validation_keys}"
        )
    if REQUIRED_VALIDATION_WALL_KEY not in shared:
        raise ValueError(
            f"{dataset} Phase-A omits required validation and mass replay wall timing"
        )
    validation_wall = _number(
        shared[REQUIRED_VALIDATION_WALL_KEY],
        f"{dataset} Phase-A validation/mass replay wall",
    )
    validation_cpu = _number(
        shared.get(REQUIRED_VALIDATION_CPU_KEY),
        f"{dataset} Phase-A validation/mass replay process CPU",
    )

    owners = _mapping(performance.get("owners"), f"{dataset} Phase-A performance owners")
    if set(owners) != OWNER_NAMES:
        raise ValueError(f"{dataset} Phase-A performance owners are not exactly N_native/C_CNBR")
    for owner_name in OWNER_NAMES:
        _mapping(owners[owner_name], f"{dataset} Phase-A owner {owner_name}")

    manifest_name = performance.get("phase_a_manifest")
    if manifest_name != "phase-a.manifest.json":
        raise ValueError(f"{dataset} Phase-A performance does not name the canonical manifest")
    manifest_path = performance_path.parent / manifest_name
    sidecar_path = manifest_path.with_name(manifest_path.name + ".sha256")
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"{dataset} Phase-A manifest or sidecar is missing")
    _require_read_only(manifest_path, f"{dataset} Phase-A manifest")
    _require_read_only(sidecar_path, f"{dataset} Phase-A manifest sidecar")
    manifest_sha256 = sha256_path(manifest_path)
    if (
        performance.get("phase_a_manifest_sha256") != manifest_sha256
        or _single_sha256_token(sidecar_path, f"{dataset} Phase-A manifest sidecar")
        != manifest_sha256
    ):
        raise ValueError(f"{dataset} Phase-A manifest checksum binding failed")
    manifest = _mapping(
        _load_json(manifest_path, f"{dataset} Phase-A manifest"),
        f"{dataset} Phase-A manifest",
    )
    if (
        manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("stage") != PHASE_A_STAGE
        or manifest.get("candidate_set") != "formal"
    ):
        raise ValueError(f"{dataset} Phase-A manifest is not the formal frozen stage")
    manifest_owners = _mapping(manifest.get("owners"), f"{dataset} Phase-A manifest owners")
    if set(manifest_owners) != OWNER_NAMES:
        raise ValueError(f"{dataset} Phase-A manifest owners are not exactly N_native/C_CNBR")
    outputs = _mapping(manifest.get("outputs"), f"{dataset} Phase-A outputs")
    performance_output = _mapping(
        outputs.get("performance_record"),
        f"{dataset} Phase-A performance output",
    )
    if performance_output.get("path") != performance_path.name:
        raise ValueError(f"{dataset} Phase-A manifest does not bind its performance record")

    source_code = _mapping(manifest.get("source_code"), f"{dataset} Phase-A source code")
    supporting = _mapping(
        source_code.get("supporting_files"),
        f"{dataset} Phase-A supporting source files",
    )
    frozen_sources = {
        "core": source_code.get("core"),
        "generator": source_code.get("generator"),
        **supporting,
    }
    if set(frozen_sources) != set(current_sources["phase_a"]):
        raise ValueError(f"{dataset} Phase-A frozen source membership drifted")
    for key, current in current_sources["phase_a"].items():
        _source_record_matches(
            performance_path.parent,
            frozen_sources[key],
            current,
            f"{dataset} Phase-A frozen source {key}",
        )
    source_record = _mapping(
        source_code.get("record"), f"{dataset} Phase-A source record"
    )
    source_record_path = _resolve_relative_file(
        performance_path.parent,
        source_record.get("path"),
        f"{dataset} Phase-A source record path",
    )
    _require_read_only(source_record_path, f"{dataset} Phase-A source record")
    if (
        source_record.get("sha256") != sha256_path(source_record_path)
        or source_record.get("size_bytes") != source_record_path.stat().st_size
    ):
        raise ValueError(f"{dataset} Phase-A source record checksum drifted")
    source_record_value = _mapping(
        _load_json(source_record_path, f"{dataset} Phase-A source record"),
        f"{dataset} Phase-A source record",
    )
    if source_record_value.get("files") != frozen_sources:
        raise ValueError(f"{dataset} Phase-A source record/file binding drifted")

    construction = _mapping(
        manifest.get("construction_inputs"), f"{dataset} Phase-A construction inputs"
    )
    artifact = _mapping(
        construction.get("artifact"), f"{dataset} Phase-A projected artifact"
    )
    _require_same_path(
        artifact.get("path"),
        Path(fresh_self["artifact_path"]),
        f"{dataset} Phase-A projected artifact path",
    )
    if (
        artifact.get("sha256") != fresh_self["artifact_sha256"]
        or construction.get("artifact_sha256") != fresh_self["artifact_sha256"]
    ):
        raise ValueError(
            f"{dataset} fresh exporter timing is bound to a different artifact"
        )
    projection = _mapping(
        construction.get("upper_only_projection"),
        f"{dataset} Phase-A upper-only projection",
    )
    _require_same_path(
        projection.get("manifest_path"),
        Path(fresh_self["projection_manifest"]),
        f"{dataset} Phase-A projection manifest path",
    )
    if (
        projection.get("manifest_sha256")
        != fresh_self["projection_manifest_sha256"]
    ):
        raise ValueError(
            f"{dataset} Phase-A projection manifest differs from fresh measurement"
        )
    ordered_vectors = _mapping(
        construction.get("ordered_vectors"),
        f"{dataset} Phase-A ordered upper vectors",
    )
    _require_same_path(
        ordered_vectors.get("path"),
        Path(fresh_self["vectors_path"]),
        f"{dataset} Phase-A upper vector path",
    )
    if ordered_vectors.get("sha256") != fresh_self["vectors_sha256"]:
        raise ValueError(
            f"{dataset} Phase-A upper vectors differ from fresh measurement"
        )
    self_navigation = _mapping(
        construction.get("self_navigation"),
        f"{dataset} Phase-A self-navigation input",
    )
    expected_navigation = {
        "row_count": fresh_self["rows"],
        "top_k": 10,
        "search_ef": 100,
        "sha256": fresh_self["hits_sha256"],
        "size_bytes": fresh_self["hits_size_bytes"],
        "manifest_sha256": fresh_self["manifest_sha256"],
    }
    navigation_mismatches = {
        key: {"observed": self_navigation.get(key), "expected": value}
        for key, value in expected_navigation.items()
        if self_navigation.get(key) != value
    }
    if navigation_mismatches:
        raise ValueError(
            f"{dataset} Phase-A is not bound to fresh projected self navigation: "
            f"{navigation_mismatches}"
        )
    if (
        construction.get("upper_node_count") != fresh_self["rows"]
        or construction.get("dimension") != fresh_self["dimension"]
        or construction.get("upper_navigation_hits_sha256")
        != fresh_self["hits_sha256"]
        or construction.get("upper_navigation_manifest_sha256")
        != fresh_self["manifest_sha256"]
        or shared.get("upper_rows") != fresh_self["rows"]
    ):
        raise ValueError(
            f"{dataset} Phase-A dataset identity differs from fresh measurement"
        )

    for owner_name, owner in owners.items():
        for key in (
            "wall_seconds",
            "process_cpu_seconds",
            "peak_rss_bytes_after",
            "upper_rows",
            "edges_examined",
            "proposal_count",
            "commit_count",
            "round_count",
        ):
            _number(
                owner.get(key),
                f"{dataset} Phase-A {owner_name} {key}",
                integer=key
                in {
                    "peak_rss_bytes_after",
                    "upper_rows",
                    "edges_examined",
                    "proposal_count",
                    "commit_count",
                    "round_count",
                },
            )
    if owners["C_CNBR"].get("upper_rows") != fresh_self["rows"]:
        raise ValueError(f"{dataset} Phase-A C_CNBR row count drifted")
    return {
        "performance_path": performance_path,
        "performance_sha256": performance_sha256,
        "performance": performance,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "manifest": manifest,
        "required_validation_wall_seconds": float(validation_wall),
        "required_validation_process_cpu_seconds": float(validation_cpu),
        "owners": owners,
    }


def _require_same_path(raw: Any, expected: Path, label: str) -> None:
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    if Path(raw).resolve() != expected.resolve():
        raise ValueError(f"{label} is bound to a different input")


def _numeric_construction_fields(performance: dict[str, Any]) -> dict[str, Any]:
    return {
        f"construction_{key}": value
        for key, value in performance.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _validate_gate_record(record: dict[str, Any], label: str, *, candidate: bool) -> None:
    required = [
        "identity_all_pass",
        "graph_topology_all_pass",
        "query_topology_all_pass",
        "topology_all_pass",
        "screen_eligible",
        "materialization_eligible",
    ]
    if candidate:
        required.append("selected_adoption_candidate")
    failures = [key for key in required if record.get(key) is not True]
    if failures:
        raise ValueError(f"{label} is not identity/topology/materialization eligible: {failures}")


def _validate_phase_b_sources(
    root: Path,
    screen: dict[str, Any],
    dataset: str,
    current_sources: dict[str, Any],
) -> None:
    bundle = _mapping(
        screen.get("evaluator_source_code"), f"{dataset} Phase-B evaluator source code"
    )
    files = _mapping(bundle.get("files"), f"{dataset} Phase-B evaluator source files")
    if set(files) != set(current_sources["phase_b"]):
        raise ValueError(f"{dataset} Phase-B frozen source membership drifted")
    for key, current in current_sources["phase_b"].items():
        _source_record_matches(
            root,
            files[key],
            current,
            f"{dataset} Phase-B frozen source {key}",
        )
    record = _mapping(bundle.get("record"), f"{dataset} Phase-B source record")
    record_path = _resolve_relative_file(
        root, record.get("path"), f"{dataset} Phase-B source record path"
    )
    _require_read_only(record_path, f"{dataset} Phase-B source record")
    if (
        record.get("sha256") != sha256_path(record_path)
        or record.get("size_bytes") != record_path.stat().st_size
    ):
        raise ValueError(f"{dataset} Phase-B source record checksum drifted")
    record_value = _mapping(
        _load_json(record_path, f"{dataset} Phase-B source record"),
        f"{dataset} Phase-B source record",
    )
    if record_value.get("files") != files:
        raise ValueError(f"{dataset} Phase-B source record/file binding drifted")


def _validate_phase_b(
    raw: str,
    dataset: str,
    phase_a: dict[str, Any],
    fresh_full: dict[str, Any],
    current_sources: dict[str, Any],
) -> dict[str, Any]:
    summary_path = _absolute_file(raw, f"{dataset} Phase-B summary")
    if summary_path.name != "summary.json":
        raise ValueError(f"{dataset} Phase-B summary has a non-canonical filename")
    _require_read_only(summary_path, f"{dataset} Phase-B summary")
    summary_sha256 = sha256_path(summary_path)
    summary = _sequence(
        _load_json(summary_path, f"{dataset} Phase-B summary"),
        f"{dataset} Phase-B summary",
    )
    if len(summary) != 2 or not all(isinstance(row, dict) for row in summary):
        raise ValueError(f"{dataset} Phase-B summary must contain exactly two rows")
    rows = {row.get("name"): row for row in summary}
    if set(rows) != OWNER_NAMES:
        raise ValueError(f"{dataset} Phase-B summary rows are not exactly N_native/C_CNBR")

    screen_path = summary_path.parent / "screen-manifest.json"
    sidecar_path = screen_path.with_name(screen_path.name + ".sha256")
    if not screen_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"{dataset} Phase-B screen manifest or sidecar is missing")
    _require_read_only(screen_path, f"{dataset} Phase-B screen manifest")
    _require_read_only(sidecar_path, f"{dataset} Phase-B screen manifest sidecar")
    screen_sha256 = sha256_path(screen_path)
    if _single_sha256_token(sidecar_path, f"{dataset} Phase-B screen sidecar") != screen_sha256:
        raise ValueError(f"{dataset} Phase-B screen checksum binding failed")
    screen = _mapping(
        _load_json(screen_path, f"{dataset} Phase-B screen manifest"),
        f"{dataset} Phase-B screen manifest",
    )
    if (
        screen.get("format_version") != FORMAT_VERSION
        or screen.get("stage") != PHASE_B_STAGE
        or screen.get("candidate_set") != "formal"
        or screen.get("comparison") != PHASE_B_COMPARISON
        or screen.get("identity_all_pass") is not True
    ):
        raise ValueError(f"{dataset} Phase-B screen is not the canonical formal evaluation")
    identity_gates = _mapping(
        screen.get("identity_gates"), f"{dataset} Phase-B identity gates"
    )
    if not identity_gates or any(value is not True for value in identity_gates.values()):
        raise ValueError(f"{dataset} Phase-B identity gates are not all true")
    outputs = _mapping(screen.get("outputs"), f"{dataset} Phase-B outputs")
    expected_outputs = {
        "summary_json": summary_path.name,
        "screen_manifest": screen_path.name,
        "screen_manifest_sidecar": sidecar_path.name,
    }
    output_mismatches = {
        key: {"observed": outputs.get(key), "expected": value}
        for key, value in expected_outputs.items()
        if outputs.get(key) != value
    }
    if output_mismatches:
        raise ValueError(f"{dataset} Phase-B output naming drifted: {output_mismatches}")
    _require_same_path(
        screen.get("phase_a_performance_record"),
        phase_a["performance_path"],
        f"{dataset} Phase-B Phase-A performance path",
    )
    if screen.get("phase_a_performance_record_sha256") != phase_a["performance_sha256"]:
        raise ValueError(f"{dataset} Phase-B is bound to a different Phase-A performance")
    _require_same_path(
        screen.get("phase_a_manifest"),
        phase_a["manifest_path"],
        f"{dataset} Phase-B Phase-A manifest path",
    )
    if screen.get("phase_a_manifest_sha256") != phase_a["manifest_sha256"]:
        raise ValueError(f"{dataset} Phase-B is bound to a different Phase-A manifest")
    if screen.get("phase_a_shared_construction_performance") != phase_a["performance"]["shared"]:
        raise ValueError(f"{dataset} Phase-B copied stale Phase-A shared timing")

    reference = _mapping(screen.get("reference"), f"{dataset} Phase-B reference")
    candidate = _mapping(screen.get("candidate"), f"{dataset} Phase-B candidate")
    candidates = _mapping(screen.get("candidates"), f"{dataset} Phase-B candidates")
    if (
        reference.get("name") != "N_native"
        or candidate.get("name") != "C_CNBR"
        or set(candidates) != {"C_CNBR"}
        or candidates["C_CNBR"] != candidate
    ):
        raise ValueError(f"{dataset} Phase-B formal owner identity drifted")
    _validate_gate_record(reference, f"{dataset} Phase-B N_native", candidate=False)
    _validate_gate_record(candidate, f"{dataset} Phase-B C_CNBR", candidate=True)

    for owner_name, screen_record in (
        ("N_native", reference),
        ("C_CNBR", candidate),
    ):
        construction = _mapping(
            screen_record.get("construction_performance"),
            f"{dataset} Phase-B {owner_name} construction performance",
        )
        if construction != phase_a["owners"][owner_name]:
            raise ValueError(
                f"{dataset} Phase-B {owner_name} construction timing differs from Phase-A"
            )
        expected_fields = _numeric_construction_fields(construction)
        actual_fields = {
            key: value
            for key, value in rows[owner_name].items()
            if isinstance(key, str) and key.startswith("construction_")
        }
        if actual_fields != expected_fields:
            raise ValueError(
                f"{dataset} Phase-B summary {owner_name} construction timing mismatch"
            )
    _validate_gate_record(rows["C_CNBR"], f"{dataset} Phase-B summary C_CNBR", candidate=True)

    post_freeze = _mapping(
        screen.get("post_freeze_evaluation_inputs"),
        f"{dataset} Phase-B post-freeze inputs",
    )
    _require_same_path(
        post_freeze.get("attachments"),
        Path(fresh_full["hits_file"]),
        f"{dataset} Phase-B full attachments path",
    )
    _require_same_path(
        post_freeze.get("attachments_manifest"),
        Path(fresh_full["manifest"]),
        f"{dataset} Phase-B full attachment manifest path",
    )
    if (
        post_freeze.get("attachments_sha256") != fresh_full["hits_sha256"]
        or post_freeze.get("attachments_manifest_sha256")
        != fresh_full["manifest_sha256"]
    ):
        raise ValueError(
            f"{dataset} Phase-B attachments are not fresh projected full hits"
        )

    _validate_phase_b_sources(
        summary_path.parent, screen, dataset, current_sources
    )
    return {
        "summary_path": summary_path,
        "summary_sha256": summary_sha256,
        "screen_path": screen_path,
        "screen_sha256": screen_sha256,
        "summary_rows": rows,
        "screen": screen,
        "candidate": candidate,
    }


def _public_source_bindings(current_sources: dict[str, Any]) -> dict[str, Any]:
    return {
        stage: {
            key: {
                "path": str(record["path"]),
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
            for key, record in current_sources[stage].items()
        }
        for stage in ("phase_a", "phase_b")
    }


def _build_audit(
    fresh_measurements: dict[str, Any],
    phase_a_records: dict[str, dict[str, Any]],
    phase_b_records: dict[str, dict[str, Any]],
    current_sources: dict[str, Any],
    source_record: dict[str, Any],
    source_record_sha256: str,
    source_record_size: int,
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    all_pass = True
    for dataset, labels in DATASETS.items():
        self_record = fresh_measurements["records"][labels["self"]]
        full_record = fresh_measurements["records"][labels["full"]]
        phase_a = phase_a_records[dataset]
        phase_b = phase_b_records[dataset]
        cnbr = phase_a["owners"]["C_CNBR"]
        numerator = (
            self_record["external_wall_seconds"]
            + phase_a["required_validation_wall_seconds"]
            + cnbr["wall_seconds"]
        )
        denominator = full_record["external_wall_seconds"]
        if denominator <= 0:
            raise ValueError(f"{dataset} canonical full-attachment wall time is not positive")
        ratio = numerator / denominator
        gate_pass = ratio <= RATIO_LIMIT
        all_pass = all_pass and gate_pass
        incremental_cpu = (
            self_record["process_cpu_seconds"]
            + phase_a["required_validation_process_cpu_seconds"]
            + cnbr["process_cpu_seconds"]
        )
        datasets[dataset] = {
            "external_self_navigation": self_record,
            "required_input_validation_and_mass_replay": {
                "wall_seconds": phase_a["required_validation_wall_seconds"],
                "process_cpu_seconds": phase_a[
                    "required_validation_process_cpu_seconds"
                ],
                "source_performance_record": str(phase_a["performance_path"]),
                "source_performance_record_sha256": phase_a[
                    "performance_sha256"
                ],
            },
            "C_CNBR_construction": {
                "wall_seconds": cnbr["wall_seconds"],
                "process_cpu_seconds": cnbr["process_cpu_seconds"],
                "peak_rss_bytes_after": cnbr["peak_rss_bytes_after"],
                "upper_rows": cnbr["upper_rows"],
                "edges_examined": cnbr["edges_examined"],
                "proposal_count": cnbr["proposal_count"],
                "commit_count": cnbr["commit_count"],
                "round_count": cnbr["round_count"],
                "source_summary": str(phase_b["summary_path"]),
                "source_summary_sha256": phase_b["summary_sha256"],
                "source_screen_manifest": str(phase_b["screen_path"]),
                "source_screen_manifest_sha256": phase_b["screen_sha256"],
            },
            "external_full_attachment": full_record,
            "strict_ratio_terms": {
                "external_self_navigation_wall_seconds": self_record[
                    "external_wall_seconds"
                ],
                "required_input_validation_and_mass_replay_wall_seconds": phase_a[
                    "required_validation_wall_seconds"
                ],
                "C_CNBR_wall_seconds": cnbr["wall_seconds"],
                "external_full_attachment_wall_seconds": full_record[
                    "external_wall_seconds"
                ],
            },
            "incremental_wall_seconds": numerator,
            "incremental_process_cpu_seconds": incremental_cpu,
            "strict_incremental_ratio": ratio,
            "strict_incremental_percent": ratio * 100,
            "strict_incremental_gate_pass": gate_pass,
            "fresh_projected_exporter_measurements_used": True,
            "old_v3_exporter_timing_reused": False,
            "old_phase_a_timing_used": False,
            "required_validation_and_mass_replay_included": True,
            "shared_n_native_kmeans_excluded": True,
        }
    status = (
        "COMPLETE_FRESH_PROJECTED_INCREMENTAL_GATE_PASS"
        if all_pass
        else "COMPLETE_FRESH_PROJECTED_INCREMENTAL_GATE_FAIL"
    )
    return {
        "format_version": FORMAT_VERSION,
        "record_type": "cnbr_construction_cost_v4",
        "status": status,
        "scope": {
            "claim": "incremental_upper_only_construction_cost",
            "datasets_must_pass_independently": True,
            "end_to_end_C_CNBR_vs_HashAll_evaluated": False,
            "end_to_end_status": "NOT_EVALUATED",
        },
        "ratio_contract": {
            "formula": (
                "(external_self_navigation_wall_seconds + "
                "required_input_validation_and_mass_replay_wall_seconds + "
                "C_CNBR_wall_seconds) / external_full_attachment_wall_seconds"
            ),
            "limit": RATIO_LIMIT,
            "operator": "<=",
            "independent_datasets": list(DATASETS),
            "shared_n_native_kmeans_excluded": True,
        },
        "fresh_projected_exporter_measurements": {
            "directory": str(fresh_measurements["root"]),
            "audit": fresh_measurements["audit"],
            "audit_sha256": fresh_measurements["audit_sha256"],
            "audit_status": fresh_measurements["audit_status"],
            "release_exporter": fresh_measurements["exporter"],
            "old_v3_timing_reused": False,
        },
        "current_repository_source_bindings": _public_source_bindings(
            current_sources
        ),
        "aggregator_source_code": {
            "record": {
                "path": SOURCE_RECORD_NAME,
                "sha256": source_record_sha256,
                "size_bytes": source_record_size,
            },
            "files": source_record["files"],
        },
        "datasets": datasets,
        "incremental_upper_only_gate": "PASS" if all_pass else "FAIL",
        "all_datasets_pass": all_pass,
        "end_to_end_c_cnbr_vs_hashall": {
            "status": "NOT_EVALUATED",
            "same_condition_evidence_claimed": False,
        },
        "outputs": {
            "audit": AUDIT_NAME,
            "audit_sidecar": f"{AUDIT_NAME}.sha256",
            "source_record": SOURCE_RECORD_NAME,
            "source_dir": "source",
            "status": MEASUREMENT_STATUS_NAME,
        },
    }


def _source_record(current_sources: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    aggregator = current_sources["aggregator"]
    value = {
        "format_version": FORMAT_VERSION,
        "record_type": "cnbr_construction_cost_v4_aggregator_source_code",
        "files": {
            "aggregator": {
                "path": f"source/{SOURCE_COPY_NAME}",
                "sha256": aggregator["sha256"],
                "size_bytes": aggregator["size_bytes"],
            }
        },
    }
    return value, _json_bytes(value)


def _publish(
    output_dir: Path,
    aggregator_bytes: bytes,
    source_record_bytes: bytes,
    audit_bytes: bytes,
) -> tuple[Path, str]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    source_dir = output_dir / "source"
    source_dir.mkdir()
    source_copy_path = source_dir / SOURCE_COPY_NAME
    source_record_path = output_dir / SOURCE_RECORD_NAME
    audit_path = output_dir / AUDIT_NAME
    sidecar_path = output_dir / f"{AUDIT_NAME}.sha256"
    status_path = output_dir / MEASUREMENT_STATUS_NAME
    with source_copy_path.open("xb") as handle:
        handle.write(aggregator_bytes)
    with source_record_path.open("xb") as handle:
        handle.write(source_record_bytes)
    with audit_path.open("xb") as handle:
        handle.write(audit_bytes)
    audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    with sidecar_path.open("xb") as handle:
        handle.write((audit_sha256 + "\n").encode("ascii"))
    audit_value = _mapping(json.loads(audit_bytes), "construction-cost audit")
    with status_path.open("x", encoding="utf-8") as handle:
        handle.write(
            "# CNBR construction-cost v4\n\n"
            f"Status: `{audit_value.get('status')}`\n\n"
            f"Audit SHA-256: `{audit_sha256}`\n\n"
            "The end-to-end C_CNBR/HashAll construction comparison remains "
            "not evaluated.\n"
        )
    if hashlib.sha256(source_copy_path.read_bytes()).hexdigest() != hashlib.sha256(
        aggregator_bytes
    ).hexdigest():
        raise ValueError("aggregator source copy changed while publishing")
    if sha256_path(source_record_path) != hashlib.sha256(source_record_bytes).hexdigest():
        raise ValueError("aggregator source record changed while publishing")
    if sha256_path(audit_path) != audit_sha256:
        raise ValueError("construction-cost audit changed while publishing")
    for path in (
        source_copy_path,
        source_record_path,
        audit_path,
        sidecar_path,
        status_path,
    ):
        os.chmod(path, 0o444)
    return audit_path, audit_sha256


def run(args: argparse.Namespace) -> tuple[Path, str]:
    output_dir = _output_directory(args.output_dir)
    current_sources = _capture_current_sources()
    fresh_measurements = _validate_fresh_measurements(
        args.fresh_measurements_dir
    )
    phase_a_records: dict[str, dict[str, Any]] = {}
    phase_b_records: dict[str, dict[str, Any]] = {}
    for dataset, phase_a_raw, phase_b_raw in (
        (
            "sift1m",
            args.sift_phase_a_performance,
            args.sift_phase_b_summary,
        ),
        (
            "glove-200-angular",
            args.glove_phase_a_performance,
            args.glove_phase_b_summary,
        ),
    ):
        labels = DATASETS[dataset]
        phase_a = _validate_phase_a(
            phase_a_raw,
            dataset,
            fresh_measurements["records"][labels["self"]],
            current_sources,
        )
        phase_b = _validate_phase_b(
            phase_b_raw,
            dataset,
            phase_a,
            fresh_measurements["records"][labels["full"]],
            current_sources,
        )
        phase_a_records[dataset] = phase_a
        phase_b_records[dataset] = phase_b

    source_record, source_record_bytes = _source_record(current_sources)
    source_record_sha256 = hashlib.sha256(source_record_bytes).hexdigest()
    audit = _build_audit(
        fresh_measurements,
        phase_a_records,
        phase_b_records,
        current_sources,
        source_record,
        source_record_sha256,
        len(source_record_bytes),
    )
    audit_bytes = _json_bytes(audit)
    _verify_current_sources(current_sources)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    return _publish(
        output_dir,
        current_sources["aggregator"]["bytes"],
        source_record_bytes,
        audit_bytes,
    )


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "measure":
        audit_path, audit_sha256 = run_fresh_measurements(
            parse_measure_args(raw[1:])
        )
        print(f"fresh_measurement_audit={audit_path}")
        print(f"fresh_measurement_audit_sha256={audit_sha256}")
        return
    if raw and raw[0] == "aggregate":
        raw = raw[1:]
    audit_path, audit_sha256 = run(parse_args(raw))
    print(f"construction_cost_audit={audit_path}")
    print(f"construction_cost_audit_sha256={audit_sha256}")


if __name__ == "__main__":
    main()
