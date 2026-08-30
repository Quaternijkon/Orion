from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance import cnbr_construction_cost_gate as gate
from experiments.l1_balance import synthesize_cnbr_construction_cost_v4 as v4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _freeze(path: Path) -> None:
    os.chmod(path, 0o444)


def _refresh_audit(root: Path, audit: dict[str, object]) -> None:
    audit_path = root / v4.AUDIT_NAME
    os.chmod(audit_path, 0o644)
    _write_json(audit_path, audit)
    _freeze(audit_path)
    sidecar = root / f"{v4.AUDIT_NAME}.sha256"
    os.chmod(sidecar, 0o644)
    sidecar.write_text(_sha256(audit_path) + "\n", encoding="ascii")
    _freeze(sidecar)
    status = root / v4.MEASUREMENT_STATUS_NAME
    os.chmod(status, 0o644)
    status.write_text(gate._status_text(_sha256(audit_path)), encoding="utf-8")
    _freeze(status)


def make_cost_audit(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "cost-v4"
    source_dir = root / "source"
    source_dir.mkdir(parents=True)

    aggregator_copy = source_dir / v4.SOURCE_COPY_NAME
    aggregator_copy.write_bytes(Path(v4.__file__).read_bytes())
    aggregator_binding = {
        "path": f"source/{v4.SOURCE_COPY_NAME}",
        "sha256": _sha256(aggregator_copy),
        "size_bytes": aggregator_copy.stat().st_size,
    }
    source_record = {
        "format_version": 1,
        "record_type": "cnbr_construction_cost_v4_aggregator_source_code",
        "files": {"aggregator": aggregator_binding},
    }
    source_record_path = root / v4.SOURCE_RECORD_NAME
    _write_json(source_record_path, source_record)

    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    fresh_audit = evidence_root / "fresh-exporter-measurement-audit.json"
    _write_json(fresh_audit, {"status": v4.MEASUREMENT_STATUS})
    exporter = evidence_root / "orion_export_upper_hits"
    exporter.write_bytes(b"fixture exporter\n")

    datasets: dict[str, object] = {}
    expected_screens: dict[str, str] = {}
    for index, dataset in enumerate(gate.EXPECTED_DATASETS, start=1):
        phase_a = evidence_root / f"{index}-phase-a.performance.json"
        summary = evidence_root / f"{index}-summary.json"
        screen = evidence_root / f"{index}-screen-manifest.json"
        _write_json(phase_a, {"dataset": dataset, "stage": "phase-a"})
        _write_json(summary, {"dataset": dataset, "stage": "phase-b-summary"})
        _write_json(screen, {"dataset": dataset, "stage": "phase-b-screen"})
        expected_screens[dataset] = _sha256(screen)
        terms = {
            "external_self_navigation_wall_seconds": 1.0,
            "required_input_validation_and_mass_replay_wall_seconds": 1.0,
            "C_CNBR_wall_seconds": 2.0,
            "external_full_attachment_wall_seconds": 100.0,
        }
        datasets[dataset] = {
            "strict_ratio_terms": terms,
            "strict_incremental_ratio": 0.04,
            "strict_incremental_percent": 4.0,
            "strict_incremental_gate_pass": True,
            "fresh_projected_exporter_measurements_used": True,
            "old_v3_exporter_timing_reused": False,
            "old_phase_a_timing_used": False,
            "required_validation_and_mass_replay_included": True,
            "shared_n_native_kmeans_excluded": True,
            "required_input_validation_and_mass_replay": {
                "source_performance_record": str(phase_a.resolve()),
                "source_performance_record_sha256": _sha256(phase_a),
            },
            "C_CNBR_construction": {
                "source_summary": str(summary.resolve()),
                "source_summary_sha256": _sha256(summary),
                "source_screen_manifest": str(screen.resolve()),
                "source_screen_manifest_sha256": _sha256(screen),
            },
        }

    source_bindings = {
        "phase_a": {
            key: _binding(path.resolve())
            for key, path in v4._phase_a_source_paths().items()
        },
        "phase_b": {
            key: _binding(path.resolve())
            for key, path in v4._phase_b_source_paths().items()
        },
    }
    audit = {
        "format_version": 1,
        "record_type": "cnbr_construction_cost_v4",
        "status": gate.PASS_STATUS,
        "scope": {
            "claim": "incremental_upper_only_construction_cost",
            "datasets_must_pass_independently": True,
            "end_to_end_C_CNBR_vs_HashAll_evaluated": False,
            "end_to_end_status": "NOT_EVALUATED",
        },
        "ratio_contract": gate.EXPECTED_RATIO_CONTRACT,
        "fresh_projected_exporter_measurements": {
            "directory": str(evidence_root),
            "audit": str(fresh_audit.resolve()),
            "audit_sha256": _sha256(fresh_audit),
            "audit_status": v4.MEASUREMENT_STATUS,
            "release_exporter": _binding(exporter),
            "old_v3_timing_reused": False,
        },
        "current_repository_source_bindings": source_bindings,
        "aggregator_source_code": {
            "record": {
                "path": v4.SOURCE_RECORD_NAME,
                "sha256": _sha256(source_record_path),
                "size_bytes": source_record_path.stat().st_size,
            },
            "files": source_record["files"],
        },
        "datasets": datasets,
        "incremental_upper_only_gate": "PASS",
        "all_datasets_pass": True,
        "end_to_end_c_cnbr_vs_hashall": {
            "status": "NOT_EVALUATED",
            "same_condition_evidence_claimed": False,
        },
        "outputs": gate.EXPECTED_OUTPUTS,
    }
    audit_path = root / v4.AUDIT_NAME
    _write_json(audit_path, audit)
    sidecar = root / f"{v4.AUDIT_NAME}.sha256"
    sidecar.write_text(_sha256(audit_path) + "\n", encoding="ascii")
    status = root / v4.MEASUREMENT_STATUS_NAME
    status.write_text(gate._status_text(_sha256(audit_path)), encoding="utf-8")
    for path in root.rglob("*"):
        if path.is_file():
            _freeze(path)
    for path in evidence_root.iterdir():
        _freeze(path)
    return audit_path, {"audit": audit, "screens": expected_screens}


def test_accepts_exact_frozen_dual_dataset_pass(tmp_path: Path) -> None:
    audit_path, fixture = make_cost_audit(tmp_path)
    result = gate.validate_construction_cost_v4_pass(
        audit_path,
        expected_formal_screen_sha256_by_dataset=fixture["screens"],
    )
    assert result["status"] == "PASS"
    assert result["audit_sha256"] == _sha256(audit_path)
    assert set(result["datasets"]) == set(gate.EXPECTED_DATASETS)
    assert all(
        row["strict_incremental_gate"] == "PASS"
        for row in result["datasets"].values()
    )


def test_rejects_missing_or_mismatched_sidecar(tmp_path: Path) -> None:
    audit_path, _fixture = make_cost_audit(tmp_path)
    sidecar = audit_path.with_name(audit_path.name + ".sha256")
    os.chmod(sidecar, 0o644)
    sidecar.write_text("0" * 64 + "\n", encoding="ascii")
    _freeze(sidecar)
    with pytest.raises(ValueError, match="sidecar mismatch"):
        gate.validate_construction_cost_v4_pass(audit_path)


def test_rejects_self_consistent_fail_status(tmp_path: Path) -> None:
    audit_path, fixture = make_cost_audit(tmp_path)
    audit = fixture["audit"]
    audit["status"] = "COMPLETE_FRESH_PROJECTED_INCREMENTAL_GATE_FAIL"
    audit["incremental_upper_only_gate"] = "FAIL"
    audit["all_datasets_pass"] = False
    _refresh_audit(audit_path.parent, audit)
    with pytest.raises(ValueError, match="status is not PASS"):
        gate.validate_construction_cost_v4_pass(audit_path)


def test_rejects_self_consistent_ratio_above_limit(tmp_path: Path) -> None:
    audit_path, fixture = make_cost_audit(tmp_path)
    audit = fixture["audit"]
    sift = audit["datasets"]["sift1m"]
    sift["strict_ratio_terms"]["C_CNBR_wall_seconds"] = 4.0
    sift["strict_incremental_ratio"] = 0.06
    sift["strict_incremental_percent"] = 6.0
    _refresh_audit(audit_path.parent, audit)
    with pytest.raises(ValueError, match="did not pass the strict gate"):
        gate.validate_construction_cost_v4_pass(audit_path)


def test_rejects_formal_screen_binding_drift(tmp_path: Path) -> None:
    audit_path, fixture = make_cost_audit(tmp_path)
    expected = dict(fixture["screens"])
    expected["glove-200-angular"] = "f" * 64
    with pytest.raises(ValueError, match="different formal screen"):
        gate.validate_construction_cost_v4_pass(
            audit_path,
            expected_formal_screen_sha256_by_dataset=expected,
        )


def test_rejects_current_source_drift_even_with_refreshed_outer_checksum(
    tmp_path: Path,
) -> None:
    audit_path, fixture = make_cost_audit(tmp_path)
    audit = fixture["audit"]
    audit["current_repository_source_bindings"]["phase_a"]["core"][
        "sha256"
    ] = "e" * 64
    _refresh_audit(audit_path.parent, audit)
    with pytest.raises(ValueError, match="checksum or size drifted"):
        gate.validate_construction_cost_v4_pass(audit_path)
