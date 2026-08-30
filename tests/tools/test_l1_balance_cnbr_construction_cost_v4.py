from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import textwrap
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance import (  # noqa: E402
    synthesize_cnbr_construction_cost_v4 as synthesize,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _freeze(path: Path) -> None:
    os.chmod(path, 0o444)


def _rewrite_frozen_json(
    path: Path, value: object, *, refresh_sidecar: bool = False
) -> None:
    os.chmod(path, 0o644)
    path.write_bytes(_json_bytes(value))
    _freeze(path)
    if refresh_sidecar:
        sidecar = path.with_name(path.name + ".sha256")
        os.chmod(sidecar, 0o644)
        sidecar.write_text(_sha256(path) + "\n", encoding="ascii")
        _freeze(sidecar)


def _external_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _internal_binding(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _semantic_records() -> dict[str, dict[str, Any]]:
    shapes = {
        "sift-self": (2, 2),
        "sift-full": (4, 2),
        "glove-self": (3, 3),
        "glove-full": (5, 3),
    }
    return {
        name: {
            "hits_sha256": hashlib.sha256(bytes(rows * 10 * 8)).hexdigest(),
            "rows": rows,
            "dimension": dimension,
        }
        for name, (rows, dimension) in shapes.items()
    }


@pytest.fixture()
def patched_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    records = _semantic_records()
    benchmark_lock = tmp_path / "benchmark.lock"
    benchmark_lock.write_bytes(b"")
    taskset = tmp_path / "taskset"
    taskset.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import sys

            if len(sys.argv) < 4 or sys.argv[1] != "-c":
                raise SystemExit(64)
            os.execv(sys.argv[3], sys.argv[3:])
            """
        ),
        encoding="utf-8",
    )
    os.chmod(taskset, 0o755)
    monkeypatch.setattr(synthesize, "SEMANTIC_RECORDS", records)
    monkeypatch.setattr(
        synthesize, "CANONICAL_BENCHMARK_LOCK", benchmark_lock
    )
    monkeypatch.setattr(synthesize, "TASKSET_PATH", taskset)
    return {
        "records": records,
        "benchmark_lock": benchmark_lock,
        "taskset": taskset,
        "monkeypatch": monkeypatch,
    }


def _make_projection_inputs(
    root: Path, patched_contract: dict[str, Any]
) -> dict[str, dict[str, Path]]:
    records = patched_contract["records"]
    result: dict[str, dict[str, Path]] = {}
    source_build_records: dict[str, dict[str, Any]] = {}
    for prefix in ("sift", "glove"):
        dataset_root = root / prefix
        dataset_root.mkdir(parents=True)
        self_record = records[f"{prefix}-self"]
        full_record = records[f"{prefix}-full"]
        artifact_path = dataset_root / "upper-only.json"
        artifact = {
            "format_version": 1,
            "layout_sha256": "0" * 64,
            "logical_point_count": self_record["rows"],
            "physical_point_count": self_record["rows"],
            "vector_schema": {"dimension": self_record["dimension"]},
            "upper_nodes": [
                {"label": index, "shard_membership": [0]}
                for index in range(self_record["rows"])
            ],
        }
        _write_json(artifact_path, artifact)
        artifact_sidecar = artifact_path.with_name(artifact_path.name + ".sha256")
        artifact_sidecar.write_text(_sha256(artifact_path) + "\n", encoding="ascii")

        self_vectors = dataset_root / "self.f32le"
        full_vectors = dataset_root / "full.f32le"
        self_vectors.write_bytes(
            bytes(self_record["rows"] * self_record["dimension"] * 4)
        )
        full_vectors.write_bytes(
            bytes(full_record["rows"] * full_record["dimension"] * 4)
        )
        source_artifact = dataset_root / "source-production.json"
        source_artifact.write_bytes(f"{prefix} source artifact\n".encode("ascii"))
        source_build_path = dataset_root / "build-manifest.json"
        source_build = {
            "dataset": {
                "sha256": ("b" if prefix == "sift" else "c") * 64,
                "dimension": full_record["dimension"],
                "train_rows_used": full_record["rows"],
            },
            "outputs": {
                "production_artifact": source_artifact.name,
                "files": {
                    source_artifact.name: _external_binding(source_artifact),
                    full_vectors.name: _external_binding(full_vectors),
                },
            },
        }
        _write_json(source_build_path, source_build)
        source_build_records[prefix] = {
            "path": source_build_path,
            "sha256": _sha256(source_build_path),
        }

        projection_path = dataset_root / "upper-only.manifest.json"
        projection_source = synthesize._phase_a_source_paths()[
            "upper_only_projection"
        ].resolve()
        projection = {
            "format_version": 1,
            "record_type": "orion_upper_only_projection",
            "output_artifact": {
                **_external_binding(artifact_path),
                "sha256_sidecar": str(artifact_sidecar.resolve()),
            },
            "source_artifact": _external_binding(source_artifact),
            "projection_source": _external_binding(projection_source),
            "contract": {
                "historical_memberships_removed": True,
                "historical_layout_metadata_removed": True,
            },
            "redaction_proof": {
                "every_output_membership_is_exact_neutral_sentinel": True,
                "historical_membership_values_emitted": False,
                "historical_membership_distribution_emitted": False,
            },
            "upper_build_provenance": {
                "source_manifest_sha256": _sha256(source_build_path),
                "source_manifest_size_bytes": source_build_path.stat().st_size,
                "parameters": {
                    "upper_sample_seed": 1,
                    "upper_m": 2,
                    "upper_ef_construction": 3,
                    "upper_graph_seed": 4,
                },
            },
        }
        _write_json(projection_path, projection)
        projection_sidecar = projection_path.with_name(
            projection_path.name + ".sha256"
        )
        projection_sidecar.write_text(
            _sha256(projection_path) + "\n", encoding="ascii"
        )
        for path in (
            artifact_path,
            artifact_sidecar,
            projection_path,
            projection_sidecar,
            source_artifact,
            source_build_path,
            self_vectors,
            full_vectors,
        ):
            _freeze(path)
        result[prefix] = {
            "artifact": artifact_path,
            "projection": projection_path,
            "source_build": source_build_path,
            "self_vectors": self_vectors,
            "full_vectors": full_vectors,
        }
    patched_contract["monkeypatch"].setattr(
        synthesize, "SOURCE_BUILD_MANIFESTS", source_build_records
    )
    return result


def _make_fake_exporter(root: Path) -> Path:
    exporter = root / "release/examples/orion_export_upper_hits"
    exporter.parent.mkdir(parents=True)
    exporter.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import hashlib
            import json
            from pathlib import Path
            import sys

            (
                artifact_raw,
                vectors_raw,
                rows_raw,
                dimension_raw,
                top_k_raw,
                search_ef_raw,
                hits_raw,
                manifest_raw,
            ) = sys.argv[1:]
            artifact = Path(artifact_raw)
            vectors = Path(vectors_raw)
            hits = Path(hits_raw)
            manifest = Path(manifest_raw)
            rows = int(rows_raw)
            dimension = int(dimension_raw)
            top_k = int(top_k_raw)
            search_ef = int(search_ef_raw)
            payload = bytes(rows * top_k * 8)
            hits.write_bytes(payload)
            record = {
                "artifact_path": str(artifact),
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "vectors_path": str(vectors),
                "vectors_sha256": hashlib.sha256(vectors.read_bytes()).hexdigest(),
                "row_count": rows,
                "dimension": dimension,
                "top_k": top_k,
                "search_ef": search_ef,
                "hits_path": str(hits),
                "hits_sha256": hashlib.sha256(payload).hexdigest(),
                "hits_size_bytes": len(payload),
            }
            manifest.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )
    os.chmod(exporter, 0o755)
    return exporter


def _measurement_args(
    tmp_path: Path,
    patched_contract: dict[str, Any],
    output_dir: Path,
) -> argparse.Namespace:
    inputs = _make_projection_inputs(
        tmp_path / "projected-inputs", patched_contract
    )
    exporter = _make_fake_exporter(tmp_path / "toolchain")
    patched_contract["monkeypatch"].setattr(
        synthesize, "CANONICAL_EXPORTER_SHA256", _sha256(exporter)
    )
    return argparse.Namespace(
        exporter=str(exporter),
        benchmark_lock=str(patched_contract["benchmark_lock"]),
        sift_artifact=str(inputs["sift"]["artifact"]),
        sift_projection_manifest=str(inputs["sift"]["projection"]),
        sift_self_vectors=str(inputs["sift"]["self_vectors"]),
        sift_full_vectors=str(inputs["sift"]["full_vectors"]),
        glove_artifact=str(inputs["glove"]["artifact"]),
        glove_projection_manifest=str(inputs["glove"]["projection"]),
        glove_self_vectors=str(inputs["glove"]["self_vectors"]),
        glove_full_vectors=str(inputs["glove"]["full_vectors"]),
        output_dir=str(output_dir),
    )


def _time_text(wall_seconds: float) -> str:
    minutes = int(wall_seconds // 60)
    seconds = wall_seconds - minutes * 60
    return (
        "User time (seconds): 0.40\n"
        "System time (seconds): 0.10\n"
        f"Elapsed (wall clock) time (h:mm:ss or m:ss): {minutes}:{seconds:05.2f}\n"
        "Maximum resident set size (kbytes): 128\n"
        "Exit status: 0\n"
    )


def _make_synthetic_fresh_measurements(
    root: Path, patched_contract: dict[str, Any]
) -> dict[str, Any]:
    records = patched_contract["records"]
    inputs = _make_projection_inputs(root.parent / "aggregate-inputs", patched_contract)
    exporter = _make_fake_exporter(root.parent / "aggregate-toolchain")
    patched_contract["monkeypatch"].setattr(
        synthesize, "CANONICAL_EXPORTER_SHA256", _sha256(exporter)
    )
    root.mkdir(parents=True)
    source_dir = root / "source"
    source_dir.mkdir()
    runner_copy = source_dir / synthesize.SOURCE_COPY_NAME
    runner_copy.write_bytes(synthesize._aggregator_source_path().read_bytes())
    runner_files = {"runner": _internal_binding(root, runner_copy)}
    source_record_path = root / synthesize.MEASUREMENT_SOURCE_RECORD_NAME
    _write_json(
        source_record_path,
        {
            "format_version": 1,
            "record_type": "cnbr_fresh_measurement_runner_source_code",
            "files": runner_files,
        },
    )

    wall_seconds = {
        "sift-self": 1.0,
        "sift-full": 100.0,
        "glove-self": 2.0,
        "glove-full": 200.0,
    }
    audit_records: dict[str, Any] = {}
    internal_paths: dict[str, dict[str, Path]] = {}
    for record_name, expected in records.items():
        prefix, scope = record_name.split("-", 1)
        input_record = inputs[prefix]
        artifact = input_record["artifact"]
        projection = input_record["projection"]
        source_build = input_record["source_build"]
        vectors = input_record[f"{scope}_vectors"]
        hits = root / f"{record_name}.hits.u64le"
        hits.write_bytes(bytes(expected["rows"] * 10 * 8))
        manifest = root / f"{record_name}.manifest.json"
        manifest_value = {
            "artifact_path": str(artifact.resolve()),
            "artifact_sha256": _sha256(artifact),
            "vectors_path": str(vectors.resolve()),
            "vectors_sha256": _sha256(vectors),
            "row_count": expected["rows"],
            "dimension": expected["dimension"],
            "top_k": 10,
            "search_ef": 100,
            "hits_path": str(hits.resolve()),
            "hits_sha256": expected["hits_sha256"],
            "hits_size_bytes": hits.stat().st_size,
        }
        _write_json(manifest, manifest_value)
        time_path = root / f"{record_name}.time.txt"
        time_path.write_text(
            _time_text(wall_seconds[record_name]), encoding="utf-8"
        )
        hits_sidecar = root / f"{record_name}.hits.sha256"
        hits_sidecar.write_text(
            f"{expected['hits_sha256']}  {hits}\n", encoding="ascii"
        )
        stdout = root / f"{record_name}.stdout.log"
        stderr = root / f"{record_name}.stderr.log"
        stdout.write_bytes(b"")
        stderr.write_bytes(b"")
        timing = synthesize._parse_time_file(time_path, record_name)
        file_paths = {
            "hits": hits,
            "manifest": manifest,
            "time": time_path,
            "hits_sha256": hits_sidecar,
            "stdout": stdout,
            "stderr": stderr,
        }
        internal_paths[record_name] = file_paths
        audit_records[record_name] = {
            "artifact": _external_binding(artifact),
            "projection_manifest": _external_binding(projection),
            "source_build_manifest": _external_binding(source_build),
            "source_build_lineage": synthesize._validate_source_build_lineage(
                source_build,
                json.loads(projection.read_text(encoding="utf-8")),
                input_record["full_vectors"],
                f"{prefix} synthetic source build",
            ),
            "vectors": _external_binding(vectors),
            "parameters": {
                "rows": expected["rows"],
                "dimension": expected["dimension"],
                "top_k": 10,
                "search_ef": 100,
            },
            "command": synthesize._measurement_command(
                exporter,
                artifact,
                vectors,
                expected,
                hits,
                manifest,
                time_path,
            ),
            "timing": timing,
            "semantic_hits_sha256": expected["hits_sha256"],
            "files": {
                key: _internal_binding(root, path)
                for key, path in file_paths.items()
            },
        }

    audit = {
        "format_version": 1,
        "record_type": "cnbr_fresh_projected_exporter_measurements_v4",
        "status": synthesize.MEASUREMENT_STATUS,
        "controls": {
            "benchmark_lock": str(
                patched_contract["benchmark_lock"].resolve()
            ),
            "kernel_flock_held_during_all_measurements": True,
            "cpu_set": synthesize.CPU_SET,
            "time_binary": _external_binding(synthesize.TIME_PATH),
            "taskset_binary": _external_binding(
                patched_contract["taskset"]
            ),
            "serial_order": [
                "sift-self",
                "sift-full",
                "glove-self",
                "glove-full",
            ],
            "release_exporter": _external_binding(exporter),
            "old_v3_timing_reused": False,
            "projected_artifact_measurement": True,
            "all_semantic_hits_match_frozen_checksums": True,
        },
        "semantic_reference": {
            "scope": "hits_bytes_only_not_timing_or_artifact_identity",
            "records": {
                name: value["hits_sha256"] for name, value in records.items()
            },
        },
        "runner_source_code": {
            "record": _internal_binding(root, source_record_path),
            "files": runner_files,
        },
        "records": audit_records,
        "outputs": {
            "audit": synthesize.MEASUREMENT_AUDIT_NAME,
            "audit_sidecar": f"{synthesize.MEASUREMENT_AUDIT_NAME}.sha256",
            "source_record": synthesize.MEASUREMENT_SOURCE_RECORD_NAME,
            "source_dir": "source",
            "status": synthesize.MEASUREMENT_STATUS_NAME,
        },
    }
    audit_path = root / synthesize.MEASUREMENT_AUDIT_NAME
    _write_json(audit_path, audit)
    audit_sidecar = root / f"{synthesize.MEASUREMENT_AUDIT_NAME}.sha256"
    audit_sidecar.write_text(_sha256(audit_path) + "\n", encoding="ascii")
    status_path = root / synthesize.MEASUREMENT_STATUS_NAME
    status_path.write_text(
        synthesize._measurement_status_text(_sha256(audit_path)),
        encoding="utf-8",
    )
    for path in root.rglob("*"):
        if path.is_file():
            _freeze(path)
    return {
        "root": root,
        "audit": audit_path,
        "files": internal_paths,
        "validated": synthesize._validate_fresh_measurements(str(root)),
    }


def _frozen_source_bundle(
    root: Path, sources: dict[str, dict[str, Any]], record_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_dir = root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for key, current in sources.items():
        copy_path = source_dir / f"{key}.source"
        copy_path.write_bytes(current["bytes"])
        _freeze(copy_path)
        records[key] = _internal_binding(root, copy_path)
    record_path = root / record_name
    _write_json(record_path, {"files": records})
    _freeze(record_path)
    return records, _internal_binding(root, record_path)


def _make_phase_a(
    root: Path,
    fresh_self: dict[str, Any],
    current_sources: dict[str, Any],
    *,
    validation_wall_seconds: float,
    cnbr_wall_seconds: float,
) -> dict[str, Any]:
    root.mkdir(parents=True)
    frozen_sources, source_record = _frozen_source_bundle(
        root,
        current_sources["phase_a"],
        "phase-a-source-code.record.json",
    )
    rows = fresh_self["rows"]
    owners = {
        "N_native": {
            "wall_seconds": 8.0,
            "process_cpu_seconds": 7.0,
            "peak_rss_bytes_after": 1_024,
            "upper_rows": rows,
            "edges_examined": 11,
            "proposal_count": 12,
            "commit_count": 13,
            "round_count": 14,
        },
        "C_CNBR": {
            "wall_seconds": cnbr_wall_seconds,
            "process_cpu_seconds": cnbr_wall_seconds - 0.25,
            "peak_rss_bytes_after": 2_048,
            "upper_rows": rows,
            "edges_examined": 21,
            "proposal_count": 22,
            "commit_count": 23,
            "round_count": 24,
        },
    }
    shared = {
        "source_snapshot_verified_before_freeze": True,
        "source_snapshot": {
            key: {
                "sha256": current["sha256"],
                "size_bytes": current["size_bytes"],
            }
            for key, current in current_sources["phase_a"].items()
        },
        "required_input_validation_and_mass_replay_wall_seconds": (
            validation_wall_seconds
        ),
        "required_input_validation_and_mass_replay_process_cpu_seconds": (
            validation_wall_seconds / 2
        ),
        "upper_rows": rows,
    }
    performance_path = root / "phase-a.performance.json"
    manifest_path = root / "phase-a.manifest.json"
    source_code = {
        "core": frozen_sources["core"],
        "generator": frozen_sources["generator"],
        "supporting_files": {
            key: value
            for key, value in frozen_sources.items()
            if key not in {"core", "generator"}
        },
        "record": source_record,
    }
    manifest = {
        "format_version": 1,
        "stage": synthesize.PHASE_A_STAGE,
        "candidate_set": "formal",
        "owners": {"N_native": {}, "C_CNBR": {}},
        "outputs": {
            "performance_record": {"path": performance_path.name}
        },
        "source_code": source_code,
        "construction_inputs": {
            "artifact": {
                "path": fresh_self["artifact_path"],
                "sha256": fresh_self["artifact_sha256"],
                "size_bytes": Path(fresh_self["artifact_path"]).stat().st_size,
            },
            "artifact_sha256": fresh_self["artifact_sha256"],
            "upper_only_projection": {
                "manifest_path": fresh_self["projection_manifest"],
                "manifest_sha256": fresh_self[
                    "projection_manifest_sha256"
                ],
            },
            "ordered_vectors": {
                "path": fresh_self["vectors_path"],
                "sha256": fresh_self["vectors_sha256"],
            },
            "self_navigation": {
                "row_count": fresh_self["rows"],
                "top_k": 10,
                "search_ef": 100,
                "sha256": fresh_self["hits_sha256"],
                "size_bytes": fresh_self["hits_size_bytes"],
                "manifest_sha256": fresh_self["manifest_sha256"],
            },
            "upper_node_count": fresh_self["rows"],
            "dimension": fresh_self["dimension"],
            "upper_navigation_hits_sha256": fresh_self["hits_sha256"],
            "upper_navigation_manifest_sha256": fresh_self[
                "manifest_sha256"
            ],
        },
    }
    _write_json(manifest_path, manifest)
    manifest_sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    manifest_sidecar.write_text(_sha256(manifest_path) + "\n", encoding="ascii")
    performance = {
        "record_type": "phase_a_construction_performance",
        "candidate_set": "formal",
        "phase_a_manifest": manifest_path.name,
        "phase_a_manifest_sha256": _sha256(manifest_path),
        "timing_contract": synthesize.PHASE_A_TIMING_CONTRACT,
        "shared": shared,
        "owners": owners,
    }
    _write_json(performance_path, performance)
    for path in (manifest_path, manifest_sidecar, performance_path):
        _freeze(path)
    return {
        "root": root,
        "performance": performance_path,
        "manifest": manifest_path,
        "manifest_sidecar": manifest_sidecar,
        "performance_value": performance,
        "manifest_value": manifest,
    }


def _gate_fields(*, candidate: bool) -> dict[str, bool]:
    value = {
        "identity_all_pass": True,
        "graph_topology_all_pass": True,
        "query_topology_all_pass": True,
        "topology_all_pass": True,
        "screen_eligible": True,
        "materialization_eligible": True,
    }
    if candidate:
        value["selected_adoption_candidate"] = True
    return value


def _construction_summary_fields(owner: dict[str, Any]) -> dict[str, Any]:
    return {
        f"construction_{key}": value
        for key, value in owner.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _make_phase_b(
    root: Path,
    phase_a: dict[str, Any],
    fresh_full: dict[str, Any],
    current_sources: dict[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True)
    frozen_sources, source_record = _frozen_source_bundle(
        root,
        current_sources["phase_b"],
        "phase-b-source-code.record.json",
    )
    performance = json.loads(
        phase_a["performance"].read_text(encoding="utf-8")
    )
    owners = performance["owners"]
    reference = {
        "name": "N_native",
        **_gate_fields(candidate=False),
        "construction_performance": owners["N_native"],
    }
    candidate = {
        "name": "C_CNBR",
        **_gate_fields(candidate=True),
        "construction_performance": owners["C_CNBR"],
    }
    summary = [
        {
            "name": "N_native",
            **_construction_summary_fields(owners["N_native"]),
        },
        {
            "name": "C_CNBR",
            **_gate_fields(candidate=True),
            **_construction_summary_fields(owners["C_CNBR"]),
        },
    ]
    summary_path = root / "summary.json"
    _write_json(summary_path, summary)
    screen_path = root / "screen-manifest.json"
    screen = {
        "format_version": 1,
        "stage": synthesize.PHASE_B_STAGE,
        "candidate_set": "formal",
        "comparison": synthesize.PHASE_B_COMPARISON,
        "identity_all_pass": True,
        "identity_gates": {"frozen_owner_identity": True},
        "outputs": {
            "summary_json": summary_path.name,
            "screen_manifest": screen_path.name,
            "screen_manifest_sidecar": screen_path.name + ".sha256",
        },
        "phase_a_performance_record": str(phase_a["performance"].resolve()),
        "phase_a_performance_record_sha256": _sha256(
            phase_a["performance"]
        ),
        "phase_a_manifest": str(phase_a["manifest"].resolve()),
        "phase_a_manifest_sha256": _sha256(phase_a["manifest"]),
        "phase_a_shared_construction_performance": performance["shared"],
        "reference": reference,
        "candidate": candidate,
        "candidates": {"C_CNBR": candidate},
        "post_freeze_evaluation_inputs": {
            "attachments": fresh_full["hits_file"],
            "attachments_manifest": fresh_full["manifest"],
            "attachments_sha256": fresh_full["hits_sha256"],
            "attachments_manifest_sha256": fresh_full["manifest_sha256"],
        },
        "evaluator_source_code": {
            "files": frozen_sources,
            "record": source_record,
        },
    }
    _write_json(screen_path, screen)
    screen_sidecar = screen_path.with_name(screen_path.name + ".sha256")
    screen_sidecar.write_text(_sha256(screen_path) + "\n", encoding="ascii")
    for path in (summary_path, screen_path, screen_sidecar):
        _freeze(path)
    return {
        "root": root,
        "summary": summary_path,
        "screen": screen_path,
        "screen_sidecar": screen_sidecar,
    }


def _make_evidence_bundle(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> dict[str, Any]:
    fresh = _make_synthetic_fresh_measurements(
        tmp_path / "fresh-measurements", patched_contract
    )
    current_sources = synthesize._capture_current_sources()
    validated = fresh["validated"]
    sift_phase_a = _make_phase_a(
        tmp_path / "sift-phase-a",
        validated["records"]["sift-self"],
        current_sources,
        validation_wall_seconds=1.0,
        cnbr_wall_seconds=2.0,
    )
    glove_phase_a = _make_phase_a(
        tmp_path / "glove-phase-a",
        validated["records"]["glove-self"],
        current_sources,
        validation_wall_seconds=2.0,
        cnbr_wall_seconds=3.0,
    )
    sift_phase_b = _make_phase_b(
        tmp_path / "sift-phase-b",
        sift_phase_a,
        validated["records"]["sift-full"],
        current_sources,
    )
    glove_phase_b = _make_phase_b(
        tmp_path / "glove-phase-b",
        glove_phase_a,
        validated["records"]["glove-full"],
        current_sources,
    )
    return {
        "fresh": fresh,
        "phase_a": {
            "sift1m": sift_phase_a,
            "glove-200-angular": glove_phase_a,
        },
        "phase_b": {
            "sift1m": sift_phase_b,
            "glove-200-angular": glove_phase_b,
        },
    }


def _aggregate_args(bundle: dict[str, Any], output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        fresh_measurements_dir=str(bundle["fresh"]["root"]),
        sift_phase_a_performance=str(
            bundle["phase_a"]["sift1m"]["performance"]
        ),
        sift_phase_b_summary=str(bundle["phase_b"]["sift1m"]["summary"]),
        glove_phase_a_performance=str(
            bundle["phase_a"]["glove-200-angular"]["performance"]
        ),
        glove_phase_b_summary=str(
            bundle["phase_b"]["glove-200-angular"]["summary"]
        ),
        output_dir=str(output_dir),
    )


def test_fresh_projected_measurement_runner_succeeds_and_freezes_files(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    output_dir = tmp_path / "fresh-run"
    args = _measurement_args(tmp_path, patched_contract, output_dir)

    audit_path, audit_sha256 = synthesize.run_fresh_measurements(args)

    assert audit_path == output_dir / synthesize.MEASUREMENT_AUDIT_NAME
    assert _sha256(audit_path) == audit_sha256
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == synthesize.MEASUREMENT_STATUS
    assert audit["controls"]["old_v3_timing_reused"] is False
    assert set(audit["records"]) == set(patched_contract["records"])
    assert synthesize._validate_fresh_measurements(str(output_dir))[
        "audit_sha256"
    ] == audit_sha256
    files = [path for path in output_dir.rglob("*") if path.is_file()]
    assert files
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in files)


def test_fresh_projected_measurement_runner_refuses_existing_output(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    output_dir = tmp_path / "fresh-run"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")
    args = _measurement_args(tmp_path, patched_contract, output_dir)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        synthesize.run_fresh_measurements(args)

    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"
    assert list(output_dir.iterdir()) == [sentinel]


def test_fresh_projected_measurement_runner_preserves_failed_attempt(
    tmp_path: Path,
    patched_contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "fresh-run"
    args = _measurement_args(tmp_path, patched_contract, output_dir)

    def fail_exporter(*_args: Any, **_kwargs: Any) -> argparse.Namespace:
        return argparse.Namespace(returncode=17)

    monkeypatch.setattr(synthesize.subprocess, "run", fail_exporter)

    with pytest.raises(RuntimeError, match="sift-self failed.*17"):
        synthesize.run_fresh_measurements(args)

    assert output_dir.is_dir()
    assert (output_dir / "sift-self.stdout.log").is_file()
    assert (output_dir / "sift-self.stderr.log").is_file()
    assert not (output_dir / synthesize.MEASUREMENT_AUDIT_NAME).exists()
    assert not (output_dir / synthesize.MEASUREMENT_STATUS_NAME).exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        synthesize.run_fresh_measurements(args)


def test_aggregate_uses_strict_formula_and_freezes_new_output(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    output_dir = tmp_path / "construction-cost-v4"

    audit_path, audit_sha256 = synthesize.run(
        _aggregate_args(bundle, output_dir)
    )

    assert _sha256(audit_path) == audit_sha256
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    sift = audit["datasets"]["sift1m"]
    glove = audit["datasets"]["glove-200-angular"]
    assert sift["strict_ratio_terms"] == {
        "external_self_navigation_wall_seconds": 1.0,
        "required_input_validation_and_mass_replay_wall_seconds": 1.0,
        "C_CNBR_wall_seconds": 2.0,
        "external_full_attachment_wall_seconds": 100.0,
    }
    assert glove["strict_ratio_terms"] == {
        "external_self_navigation_wall_seconds": 2.0,
        "required_input_validation_and_mass_replay_wall_seconds": 2.0,
        "C_CNBR_wall_seconds": 3.0,
        "external_full_attachment_wall_seconds": 200.0,
    }
    assert sift["strict_incremental_ratio"] == pytest.approx(0.04)
    assert glove["strict_incremental_ratio"] == pytest.approx(0.035)
    assert audit["status"] == "COMPLETE_FRESH_PROJECTED_INCREMENTAL_GATE_PASS"
    assert audit["all_datasets_pass"] is True
    assert audit["ratio_contract"]["formula"] == (
        "(external_self_navigation_wall_seconds + "
        "required_input_validation_and_mass_replay_wall_seconds + "
        "C_CNBR_wall_seconds) / external_full_attachment_wall_seconds"
    )
    assert sift["required_input_validation_and_mass_replay"] == {
        "wall_seconds": 1.0,
        "process_cpu_seconds": 0.5,
        "source_performance_record": str(
            bundle["phase_a"]["sift1m"]["performance"].resolve()
        ),
        "source_performance_record_sha256": _sha256(
            bundle["phase_a"]["sift1m"]["performance"]
        ),
    }
    assert "self_navigation_input_validation_and_mass_replay" not in sift
    assert sift["required_validation_and_mass_replay_included"] is True
    assert sift["shared_n_native_kmeans_excluded"] is True
    assert sift["old_phase_a_timing_used"] is False
    expected_files = {
        synthesize.AUDIT_NAME,
        f"{synthesize.AUDIT_NAME}.sha256",
        synthesize.SOURCE_RECORD_NAME,
        f"source/{synthesize.SOURCE_COPY_NAME}",
        synthesize.MEASUREMENT_STATUS_NAME,
    }
    files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert files == expected_files
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o444
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    assert (output_dir / f"{synthesize.AUDIT_NAME}.sha256").read_text(
        encoding="ascii"
    ) == audit_sha256 + "\n"


def test_aggregate_refuses_existing_output_and_preserves_sentinel(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    output_dir = tmp_path / "construction-cost-v4"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert sentinel.read_text(encoding="utf-8") == "do not overwrite\n"
    assert list(output_dir.iterdir()) == [sentinel]


@pytest.mark.parametrize("drift", ["audit", "time", "manifest"])
def test_aggregate_rejects_fresh_measurement_checksum_drift_before_output(
    tmp_path: Path,
    patched_contract: dict[str, Any],
    drift: str,
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    if drift == "audit":
        target = bundle["fresh"]["audit"]
    else:
        target = bundle["fresh"]["files"]["sift-self"][drift]
    os.chmod(target, 0o644)
    target.write_bytes(target.read_bytes() + b"\n")
    _freeze(target)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="checksum.*drift"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_aggregate_rejects_self_consistent_stale_measurement_runner(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    fresh_root = bundle["fresh"]["root"]
    runner_copy = fresh_root / "source" / synthesize.SOURCE_COPY_NAME
    os.chmod(runner_copy, 0o644)
    runner_copy.write_bytes(b"stale but internally checksum-bound runner\n")
    _freeze(runner_copy)

    source_record_path = fresh_root / synthesize.MEASUREMENT_SOURCE_RECORD_NAME
    source_record = json.loads(source_record_path.read_text(encoding="utf-8"))
    source_record["files"]["runner"] = _internal_binding(
        fresh_root, runner_copy
    )
    _rewrite_frozen_json(source_record_path, source_record)

    audit_path = bundle["fresh"]["audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["runner_source_code"]["files"] = source_record["files"]
    audit["runner_source_code"]["record"] = _internal_binding(
        fresh_root, source_record_path
    )
    _rewrite_frozen_json(audit_path, audit, refresh_sidecar=True)
    status_path = fresh_root / synthesize.MEASUREMENT_STATUS_NAME
    os.chmod(status_path, 0o644)
    status_path.write_text(
        synthesize._measurement_status_text(_sha256(audit_path)),
        encoding="utf-8",
    )
    _freeze(status_path)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="not the current v4 source"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_aggregate_rejects_unbound_fresh_measurement_file(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    extra = bundle["fresh"]["root"] / "unbound-old-timing.txt"
    extra.write_text("must not be accepted\n", encoding="utf-8")
    _freeze(extra)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="extra or missing files"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_aggregate_rejects_unbound_fresh_measurement_directory(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    (bundle["fresh"]["root"] / "unbound-directory").mkdir()
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="directory layout drifted"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_aggregate_rejects_fresh_measurement_status_drift(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    status_path = bundle["fresh"]["root"] / synthesize.MEASUREMENT_STATUS_NAME
    os.chmod(status_path, 0o644)
    status_path.write_text("Status: `fabricated`\n", encoding="utf-8")
    _freeze(status_path)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="STATUS content drifted"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_aggregate_rejects_phase_a_without_current_source_snapshot(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    performance_path = bundle["phase_a"]["sift1m"]["performance"]
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    del performance["shared"]["source_snapshot"]
    _rewrite_frozen_json(performance_path, performance)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="source snapshot"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_aggregate_rejects_phase_a_projected_artifact_mismatch(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    phase_a = bundle["phase_a"]["sift1m"]
    manifest = json.loads(phase_a["manifest"].read_text(encoding="utf-8"))
    manifest["construction_inputs"]["artifact"]["sha256"] = "f" * 64
    _rewrite_frozen_json(
        phase_a["manifest"], manifest, refresh_sidecar=True
    )
    performance = json.loads(
        phase_a["performance"].read_text(encoding="utf-8")
    )
    performance["phase_a_manifest_sha256"] = _sha256(phase_a["manifest"])
    _rewrite_frozen_json(phase_a["performance"], performance)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="different artifact"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_aggregate_rejects_phase_b_bound_to_different_phase_a(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    screen_path = bundle["phase_b"]["sift1m"]["screen"]
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    screen["phase_a_performance_record_sha256"] = "e" * 64
    _rewrite_frozen_json(screen_path, screen, refresh_sidecar=True)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="different Phase-A performance"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_aggregate_rejects_omitted_validation_and_mass_replay_timing(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    performance_path = bundle["phase_a"]["sift1m"]["performance"]
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    del performance["shared"][
        "required_input_validation_and_mass_replay_wall_seconds"
    ]
    _rewrite_frozen_json(performance_path, performance)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="omits required validation"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_aggregate_rejects_wrong_phase_a_timing_contract(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    performance_path = bundle["phase_a"]["sift1m"]["performance"]
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    performance["timing_contract"]["contract_id"] = "legacy-cost-contract"
    _rewrite_frozen_json(performance_path, performance)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="timing contract does not match"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_aggregate_rejects_legacy_self_navigation_timing_keys(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    performance_path = bundle["phase_a"]["sift1m"]["performance"]
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    performance["shared"][
        "self_navigation_input_validation_and_mass_replay_wall_seconds"
    ] = 1.0
    performance["shared"][
        "self_navigation_input_validation_and_mass_replay_process_cpu_seconds"
    ] = 0.5
    _rewrite_frozen_json(performance_path, performance)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="inadmissible legacy self-navigation"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_changed_validation_timing_changes_strict_result(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    phase_a = bundle["phase_a"]["sift1m"]
    performance = json.loads(
        phase_a["performance"].read_text(encoding="utf-8")
    )
    performance["shared"][
        "required_input_validation_and_mass_replay_wall_seconds"
    ] = 3.0
    _rewrite_frozen_json(phase_a["performance"], performance)
    screen_path = bundle["phase_b"]["sift1m"]["screen"]
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    screen["phase_a_performance_record_sha256"] = _sha256(
        phase_a["performance"]
    )
    screen["phase_a_shared_construction_performance"] = performance["shared"]
    _rewrite_frozen_json(screen_path, screen, refresh_sidecar=True)
    output_dir = tmp_path / "construction-cost-v4"

    audit_path, _audit_sha = synthesize.run(
        _aggregate_args(bundle, output_dir)
    )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    sift = audit["datasets"]["sift1m"]
    assert sift["strict_incremental_ratio"] == pytest.approx(0.06)
    assert sift["strict_incremental_gate_pass"] is False
    assert audit["status"] == "COMPLETE_FRESH_PROJECTED_INCREMENTAL_GATE_FAIL"
    assert audit["all_datasets_pass"] is False


def test_aggregate_rejects_construction_timing_mismatch(
    tmp_path: Path, patched_contract: dict[str, Any]
) -> None:
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    summary_path = bundle["phase_b"]["sift1m"]["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    c_cnbr = next(row for row in summary if row["name"] == "C_CNBR")
    c_cnbr["construction_wall_seconds"] += 0.5
    _rewrite_frozen_json(summary_path, summary)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="construction timing mismatch"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()


def test_aggregate_rechecks_aggregator_source_before_publishing(
    tmp_path: Path,
    patched_contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surrogate_source = tmp_path / "surrogate-aggregator.py"
    surrogate_source.write_bytes(b"original aggregator source\n")
    monkeypatch.setattr(
        synthesize, "_aggregator_source_path", lambda: surrogate_source
    )
    bundle = _make_evidence_bundle(tmp_path, patched_contract)
    original_build_audit = synthesize._build_audit

    def build_then_drift(*args: Any, **kwargs: Any) -> dict[str, Any]:
        audit = original_build_audit(*args, **kwargs)
        surrogate_source.write_bytes(b"changed aggregator source\n")
        return audit

    monkeypatch.setattr(synthesize, "_build_audit", build_then_drift)
    output_dir = tmp_path / "construction-cost-v4"

    with pytest.raises(ValueError, match="aggregator source changed"):
        synthesize.run(_aggregate_args(bundle, output_dir))

    assert not output_dir.exists()
