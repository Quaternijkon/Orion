#!/usr/bin/env python3
"""Validate a frozen aggregate CNBR construction-cost-v4 PASS artifact.

The aggregator performs the expensive evidence replay once.  Adoption-path
consumers use this module to fail closed on any missing file, checksum/source
drift, non-PASS status, or ratio-contract inconsistency, and then record the
exact aggregate audit SHA-256 in their own immutable evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from experiments.l1_balance import synthesize_cnbr_construction_cost_v4 as v4


PASS_STATUS = "COMPLETE_FRESH_PROJECTED_INCREMENTAL_GATE_PASS"
EXPECTED_DATASETS = tuple(v4.DATASETS)
EXPECTED_RATIO_CONTRACT = {
    "formula": (
        "(external_self_navigation_wall_seconds + "
        "required_input_validation_and_mass_replay_wall_seconds + "
        "C_CNBR_wall_seconds) / external_full_attachment_wall_seconds"
    ),
    "limit": v4.RATIO_LIMIT,
    "operator": "<=",
    "independent_datasets": list(EXPECTED_DATASETS),
    "shared_n_native_kmeans_excluded": True,
}
EXPECTED_OUTPUTS = {
    "audit": v4.AUDIT_NAME,
    "audit_sidecar": f"{v4.AUDIT_NAME}.sha256",
    "source_record": v4.SOURCE_RECORD_NAME,
    "source_dir": "source",
    "status": v4.MEASUREMENT_STATUS_NAME,
}


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _normalize_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a valid SHA-256 string")
    return normalized


def _require_file(raw: str | os.PathLike[str] | Path, label: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _require_read_only(path: Path, label: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o222:
        raise ValueError(f"{label} is not frozen read-only: mode={mode:o}")


def _single_sha256(path: Path, label: str) -> str:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{label} is not valid ASCII") from error
    if len(lines) != 1 or len(lines[0].split()) != 1:
        raise ValueError(f"{label} must contain exactly one SHA-256 token")
    return _normalize_sha256(lines[0], label)


def _resolve_internal(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ValueError(f"{label} must be a non-empty relative path")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the construction-cost directory") from error
    return _require_file(path, label)


def _validate_file_binding(
    path: Path, record: Mapping[str, Any], label: str
) -> str:
    digest = _normalize_sha256(record.get("sha256"), f"{label} SHA-256")
    size = record.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{label} size_bytes is invalid")
    if path.stat().st_size != size or sha256_path(path) != digest:
        raise ValueError(f"{label} checksum or size drifted")
    return digest


def _validate_external_file(
    raw_path: Any, raw_sha256: Any, label: str
) -> tuple[Path, str]:
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = _require_file(raw_path, label)
    digest = _normalize_sha256(raw_sha256, f"{label} SHA-256")
    _require_read_only(path, label)
    if sha256_path(path) != digest:
        raise ValueError(f"{label} checksum drifted")
    return path, digest


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def _status_text(audit_sha256: str) -> str:
    return (
        "# CNBR construction-cost v4\n\n"
        f"Status: `{PASS_STATUS}`\n\n"
        f"Audit SHA-256: `{audit_sha256}`\n\n"
        "The end-to-end C_CNBR/HashAll construction comparison remains "
        "not evaluated.\n"
    )


def _validate_aggregator_source(root: Path, audit: Mapping[str, Any]) -> str:
    bundle = _mapping(
        audit.get("aggregator_source_code"), "aggregator source-code binding"
    )
    record_binding = _mapping(
        bundle.get("record"), "aggregator source-code record binding"
    )
    if record_binding.get("path") != v4.SOURCE_RECORD_NAME:
        raise ValueError("construction-cost source-record path drifted")
    record_path = _resolve_internal(
        root, record_binding.get("path"), "construction-cost source record"
    )
    _require_read_only(record_path, "construction-cost source record")
    record_sha256 = _validate_file_binding(
        record_path, record_binding, "construction-cost source record"
    )
    record = _load_json(record_path, "construction-cost source record")
    if (
        record.get("format_version") != v4.FORMAT_VERSION
        or record.get("record_type")
        != "cnbr_construction_cost_v4_aggregator_source_code"
    ):
        raise ValueError("construction-cost source-record schema drifted")
    files = _mapping(record.get("files"), "construction-cost source files")
    if set(files) != {"aggregator"} or bundle.get("files") != files:
        raise ValueError("construction-cost source file membership drifted")
    aggregator_binding = _mapping(files["aggregator"], "aggregator source binding")
    expected_relative = f"source/{v4.SOURCE_COPY_NAME}"
    if aggregator_binding.get("path") != expected_relative:
        raise ValueError("construction-cost aggregator source path drifted")
    aggregator_copy = _resolve_internal(
        root, aggregator_binding.get("path"), "frozen aggregator source"
    )
    _require_read_only(aggregator_copy, "frozen aggregator source")
    frozen_sha256 = _validate_file_binding(
        aggregator_copy, aggregator_binding, "frozen aggregator source"
    )
    current = Path(v4.__file__).resolve()
    if sha256_path(current) != frozen_sha256:
        raise ValueError("construction-cost aggregator source differs from current v4")
    return record_sha256


def _validate_current_source_bindings(audit: Mapping[str, Any]) -> None:
    bindings = _mapping(
        audit.get("current_repository_source_bindings"),
        "construction-cost repository source bindings",
    )
    expected_paths = {
        "phase_a": v4._phase_a_source_paths(),
        "phase_b": v4._phase_b_source_paths(),
    }
    if set(bindings) != set(expected_paths):
        raise ValueError("construction-cost repository source stages drifted")
    for stage, paths in expected_paths.items():
        stage_bindings = _mapping(bindings.get(stage), f"{stage} source bindings")
        if set(stage_bindings) != set(paths):
            raise ValueError(f"construction-cost {stage} source membership drifted")
        for name, expected_path in paths.items():
            record = _mapping(
                stage_bindings.get(name), f"{stage} source binding {name}"
            )
            raw_path = record.get("path")
            if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
                raise ValueError(f"construction-cost {stage} {name} path is invalid")
            path = _require_file(raw_path, f"construction-cost {stage} {name}")
            if path != expected_path.resolve():
                raise ValueError(f"construction-cost {stage} {name} path drifted")
            _validate_file_binding(path, record, f"construction-cost {stage} {name}")


def validate_construction_cost_v4_pass(
    raw_path: str | os.PathLike[str] | Path,
    *,
    expected_formal_screen_sha256_by_dataset: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return immutable public evidence only after every v4 PASS gate replays."""

    audit_path = _require_file(raw_path, "construction-cost-v4 audit")
    if audit_path.name != v4.AUDIT_NAME:
        raise ValueError(
            f"construction-cost-v4 audit must be named {v4.AUDIT_NAME}"
        )
    root = audit_path.parent.resolve()
    sidecar_path = root / f"{v4.AUDIT_NAME}.sha256"
    status_path = root / v4.MEASUREMENT_STATUS_NAME
    for path, label in (
        (audit_path, "construction-cost-v4 audit"),
        (sidecar_path, "construction-cost-v4 audit sidecar"),
        (status_path, "construction-cost-v4 status"),
    ):
        _require_file(path, label)
        _require_read_only(path, label)
    audit_sha256 = sha256_path(audit_path)
    if _single_sha256(sidecar_path, "construction-cost-v4 audit sidecar") != audit_sha256:
        raise ValueError("construction-cost-v4 audit checksum sidecar mismatch")

    audit = _load_json(audit_path, "construction-cost-v4 audit")
    if (
        audit.get("format_version") != v4.FORMAT_VERSION
        or audit.get("record_type") != "cnbr_construction_cost_v4"
    ):
        raise ValueError("construction-cost-v4 audit schema drifted")
    if audit.get("status") != PASS_STATUS:
        raise ValueError("construction-cost-v4 status is not PASS")
    if (
        audit.get("incremental_upper_only_gate") != "PASS"
        or audit.get("all_datasets_pass") is not True
    ):
        raise ValueError("construction-cost-v4 aggregate gate is not PASS")
    if audit.get("ratio_contract") != EXPECTED_RATIO_CONTRACT:
        raise ValueError("construction-cost-v4 ratio contract drifted")
    if audit.get("outputs") != EXPECTED_OUTPUTS:
        raise ValueError("construction-cost-v4 output contract drifted")
    scope = _mapping(audit.get("scope"), "construction-cost-v4 scope")
    if (
        scope.get("claim") != "incremental_upper_only_construction_cost"
        or scope.get("datasets_must_pass_independently") is not True
        or scope.get("end_to_end_C_CNBR_vs_HashAll_evaluated") is not False
        or scope.get("end_to_end_status") != "NOT_EVALUATED"
    ):
        raise ValueError("construction-cost-v4 scope drifted")
    end_to_end = _mapping(
        audit.get("end_to_end_c_cnbr_vs_hashall"),
        "construction-cost-v4 end-to-end scope",
    )
    if end_to_end != {
        "status": "NOT_EVALUATED",
        "same_condition_evidence_claimed": False,
    }:
        raise ValueError("construction-cost-v4 end-to-end claim drifted")

    source_record_sha256 = _validate_aggregator_source(root, audit)
    _validate_current_source_bindings(audit)
    if status_path.read_text(encoding="utf-8") != _status_text(audit_sha256):
        raise ValueError("construction-cost-v4 STATUS content drifted")

    fresh = _mapping(
        audit.get("fresh_projected_exporter_measurements"),
        "construction-cost-v4 fresh measurements",
    )
    if (
        fresh.get("audit_status") != v4.MEASUREMENT_STATUS
        or fresh.get("old_v3_timing_reused") is not False
    ):
        raise ValueError("construction-cost-v4 fresh measurement status drifted")
    _validate_external_file(
        fresh.get("audit"),
        fresh.get("audit_sha256"),
        "construction-cost-v4 fresh measurement audit",
    )
    exporter = _mapping(
        fresh.get("release_exporter"),
        "construction-cost-v4 release exporter binding",
    )
    exporter_path_raw = exporter.get("path")
    if not isinstance(exporter_path_raw, str) or not Path(exporter_path_raw).is_absolute():
        raise ValueError("construction-cost-v4 release exporter path is invalid")
    exporter_path = _require_file(
        exporter_path_raw, "construction-cost-v4 release exporter"
    )
    _validate_file_binding(
        exporter_path, exporter, "construction-cost-v4 release exporter"
    )

    datasets = _mapping(audit.get("datasets"), "construction-cost-v4 datasets")
    if set(datasets) != set(EXPECTED_DATASETS):
        raise ValueError("construction-cost-v4 dataset set drifted")
    if expected_formal_screen_sha256_by_dataset is not None:
        if set(expected_formal_screen_sha256_by_dataset) != set(EXPECTED_DATASETS):
            raise ValueError("expected formal-screen dataset set drifted")
        expected_screens = {
            dataset: _normalize_sha256(
                digest, f"expected {dataset} formal Phase-B screen SHA-256"
            )
            for dataset, digest in expected_formal_screen_sha256_by_dataset.items()
        }
    else:
        expected_screens = None

    public_datasets: dict[str, Any] = {}
    for dataset in EXPECTED_DATASETS:
        record = _mapping(datasets.get(dataset), f"construction-cost {dataset}")
        terms = _mapping(
            record.get("strict_ratio_terms"),
            f"construction-cost {dataset} ratio terms",
        )
        expected_term_keys = {
            "external_self_navigation_wall_seconds",
            "required_input_validation_and_mass_replay_wall_seconds",
            "C_CNBR_wall_seconds",
            "external_full_attachment_wall_seconds",
        }
        if set(terms) != expected_term_keys:
            raise ValueError(f"construction-cost {dataset} ratio terms drifted")
        values = {key: _number(value, f"{dataset} {key}") for key, value in terms.items()}
        denominator = values["external_full_attachment_wall_seconds"]
        if denominator <= 0:
            raise ValueError(f"construction-cost {dataset} denominator is not positive")
        replayed_ratio = (
            values["external_self_navigation_wall_seconds"]
            + values["required_input_validation_and_mass_replay_wall_seconds"]
            + values["C_CNBR_wall_seconds"]
        ) / denominator
        recorded_ratio = _number(
            record.get("strict_incremental_ratio"),
            f"construction-cost {dataset} strict ratio",
        )
        recorded_percent = _number(
            record.get("strict_incremental_percent"),
            f"construction-cost {dataset} strict percent",
        )
        if not math.isclose(recorded_ratio, replayed_ratio, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"construction-cost {dataset} strict ratio does not replay")
        if not math.isclose(recorded_percent, replayed_ratio * 100.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"construction-cost {dataset} strict percent does not replay")
        if (
            replayed_ratio > v4.RATIO_LIMIT
            or record.get("strict_incremental_gate_pass") is not True
            or record.get("fresh_projected_exporter_measurements_used") is not True
            or record.get("old_v3_exporter_timing_reused") is not False
            or record.get("old_phase_a_timing_used") is not False
            or record.get("required_validation_and_mass_replay_included") is not True
            or record.get("shared_n_native_kmeans_excluded") is not True
        ):
            raise ValueError(f"construction-cost {dataset} did not pass the strict gate")

        validation = _mapping(
            record.get("required_input_validation_and_mass_replay"),
            f"construction-cost {dataset} validation record",
        )
        _validate_external_file(
            validation.get("source_performance_record"),
            validation.get("source_performance_record_sha256"),
            f"construction-cost {dataset} Phase-A performance",
        )
        cnbr = _mapping(
            record.get("C_CNBR_construction"),
            f"construction-cost {dataset} C_CNBR record",
        )
        _, summary_sha256 = _validate_external_file(
            cnbr.get("source_summary"),
            cnbr.get("source_summary_sha256"),
            f"construction-cost {dataset} Phase-B summary",
        )
        screen_path, screen_sha256 = _validate_external_file(
            cnbr.get("source_screen_manifest"),
            cnbr.get("source_screen_manifest_sha256"),
            f"construction-cost {dataset} Phase-B screen",
        )
        if expected_screens is not None and screen_sha256 != expected_screens[dataset]:
            raise ValueError(
                f"construction-cost {dataset} is bound to a different formal screen"
            )
        public_datasets[dataset] = {
            "strict_incremental_ratio": recorded_ratio,
            "strict_incremental_percent": recorded_percent,
            "strict_incremental_gate": "PASS",
            "phase_b_screen": str(screen_path),
            "phase_b_screen_sha256": screen_sha256,
            "phase_b_summary_sha256": summary_sha256,
        }

    expected_files = {
        v4.AUDIT_NAME,
        f"{v4.AUDIT_NAME}.sha256",
        v4.SOURCE_RECORD_NAME,
        v4.MEASUREMENT_STATUS_NAME,
        f"source/{v4.SOURCE_COPY_NAME}",
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("construction-cost-v4 frozen file set drifted")
    actual_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }
    if actual_directories != {"source"}:
        raise ValueError("construction-cost-v4 directory layout drifted")

    return {
        "status": "PASS",
        "record_type": "cnbr_construction_cost_v4_adoption_gate",
        "audit": str(audit_path),
        "audit_sha256": audit_sha256,
        "aggregate_status": PASS_STATUS,
        "incremental_upper_only_gate": "PASS",
        "all_datasets_pass": True,
        "ratio_limit": v4.RATIO_LIMIT,
        "aggregator_source_record_sha256": source_record_sha256,
        "datasets": public_datasets,
    }
