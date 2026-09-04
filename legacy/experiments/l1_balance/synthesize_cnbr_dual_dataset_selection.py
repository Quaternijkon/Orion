#!/usr/bin/env python3
"""Audit SIFT/GloVe grid screens and freeze the unique CNBR selection proof."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance import evaluate_cnbr_frozen_owners as phase_b  # noqa: E402


GRID_NAMES = (
    "CNBR_9_4",
    "CNBR_2_1",
    "CNBR_7_4",
    "CNBR_8_5",
    "CNBR_3_2",
    "CNBR_7_5",
)
SELECTED_GRID_NAME = "CNBR_9_4"
FORMAL_NAME = "C_CNBR"
REFERENCE_NAME = "N_native"

PHASE_B_OUTPUTS = {
    "summary_csv": "summary.csv",
    "summary_json": "summary.json",
    "screen_manifest": "screen-manifest.json",
    "screen_manifest_sidecar": "screen-manifest.json.sha256",
    "source_dir": "source",
    "source_record": "phase-b-source-code.record.json",
}

POST_FREEZE_FILES = (
    ("dataset_manifest", "dataset_manifest_sha256"),
    ("source_build_manifest", "source_build_manifest_sha256"),
    ("attachments", "attachments_sha256"),
    ("attachments_manifest", "attachments_manifest_sha256"),
    ("query_hits", "query_hits_sha256"),
    ("query_hits_manifest", "query_hits_manifest_sha256"),
    ("ground_truth", "ground_truth_sha256"),
    ("ground_truth_manifest", "ground_truth_manifest_sha256"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sift-grid", required=True)
    parser.add_argument("--glove-grid", required=True)
    parser.add_argument("--sift-formal", required=True)
    parser.add_argument("--glove-formal", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_file(raw: str, label: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _require_read_only(path: Path, label: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o222:
        raise ValueError(f"{label} is not frozen read-only: mode={mode:o}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _relative_frozen_file(base_dir: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ValueError(f"{label} must be a relative frozen path")
    path = (base_dir / raw).resolve()
    try:
        path.relative_to(base_dir.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its frozen output directory") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _bound_file(
    value: Mapping[str, Any], path_key: str, sha_key: str, label: str
) -> Path:
    raw = value.get(path_key)
    if not isinstance(raw, str):
        raise ValueError(f"{label} path is missing")
    path = _absolute_file(raw, f"{label} path")
    digest = value.get(sha_key)
    if not isinstance(digest, str) or sha256_path(path) != digest:
        raise ValueError(f"{label} checksum binding drifted")
    return path


def _validate_evaluator_source_code(
    screen_path: Path, screen: Mapping[str, Any], label: str
) -> None:
    binding = screen.get("evaluator_source_code")
    if not isinstance(binding, Mapping) or set(binding) != {"record", "files"}:
        raise ValueError(f"{label} evaluator source binding schema drifted")
    files = binding.get("files")
    record_binding = binding.get("record")
    current_sources = phase_b._phase_b_source_paths()
    if not isinstance(files, Mapping) or set(files) != set(current_sources):
        raise ValueError(f"{label} evaluator source file set drifted")
    if not isinstance(record_binding, Mapping):
        raise ValueError(f"{label} evaluator source record binding is missing")
    record_path = _relative_frozen_file(
        screen_path.parent,
        record_binding.get("path"),
        f"{label} evaluator source record",
    )
    record_sha256 = sha256_path(record_path)
    if (
        set(record_binding) != {"path", "sha256", "size_bytes"}
        or record_binding.get("sha256") != record_sha256
        or record_binding.get("size_bytes") != record_path.stat().st_size
    ):
        raise ValueError(f"{label} evaluator source record checksum drifted")
    _require_read_only(record_path, f"{label} evaluator source record")
    record = _object(
        json.loads(record_path.read_text(encoding="utf-8")),
        f"{label} evaluator source record",
    )
    if (
        set(record) != {"format_version", "record_type", "files", "runtime"}
        or record.get("format_version") != 1
        or record.get("record_type") != "phase_b_evaluator_source_code"
        or record.get("files") != files
        or record.get("runtime")
        != {
            "python_version": sys.version,
            "numpy_version": phase_b.np.__version__,
        }
    ):
        raise ValueError(f"{label} evaluator source record content drifted")
    frozen_source_paths: set[Path] = set()
    for name, current_path in current_sources.items():
        file_binding = files.get(name)
        if not isinstance(file_binding, Mapping) or set(file_binding) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"{label} evaluator source {name} binding drifted")
        frozen_path = _relative_frozen_file(
            screen_path.parent,
            file_binding.get("path"),
            f"{label} evaluator source {name}",
        )
        frozen_source_paths.add(frozen_path)
        frozen_sha256 = sha256_path(frozen_path)
        if (
            file_binding.get("sha256") != frozen_sha256
            or file_binding.get("size_bytes") != frozen_path.stat().st_size
            or sha256_path(current_path.resolve()) != frozen_sha256
        ):
            raise ValueError(f"{label} evaluator source {name} checksum drifted")
        _require_read_only(frozen_path, f"{label} evaluator source {name}")
    source_dir = screen_path.parent / PHASE_B_OUTPUTS["source_dir"]
    if not source_dir.is_dir() or {
        path.resolve() for path in source_dir.iterdir() if path.is_file()
    } != frozen_source_paths:
        raise ValueError(f"{label} evaluator source directory drifted")


def _expected_summary_csv(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    fields = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _validate_phase_b_outputs(
    screen_path: Path,
    screen: Mapping[str, Any],
    rows: list[dict[str, Any]],
    label: str,
) -> None:
    if screen.get("outputs") != PHASE_B_OUTPUTS:
        raise ValueError(f"{label} output schema drifted")
    if screen_path.name != PHASE_B_OUTPUTS["screen_manifest"]:
        raise ValueError(f"{label} screen filename drifted")
    summary_json = _relative_frozen_file(
        screen_path.parent,
        PHASE_B_OUTPUTS["summary_json"],
        f"{label} summary JSON",
    )
    summary_csv = _relative_frozen_file(
        screen_path.parent,
        PHASE_B_OUTPUTS["summary_csv"],
        f"{label} summary CSV",
    )
    expected_json = (
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if summary_json.read_bytes() != expected_json:
        raise ValueError(f"{label} summary JSON does not replay")
    if summary_csv.read_bytes() != _expected_summary_csv(rows):
        raise ValueError(f"{label} summary CSV does not replay")
    _require_read_only(summary_json, f"{label} summary JSON")
    _require_read_only(summary_csv, f"{label} summary CSV")


def _deep_replay_screen(
    path: Path,
    screen: dict[str, Any],
    *,
    candidate_set: str,
    dataset: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-open every frozen input and recompute the complete Phase-B screen."""

    _validate_evaluator_source_code(path, screen, label)
    post_freeze = screen.get("post_freeze_evaluation_inputs")
    expected_post_keys = {
        key for pair in POST_FREEZE_FILES for key in pair
    } | {"dataset_sha256", "query_row_count"}
    if not isinstance(post_freeze, Mapping) or set(post_freeze) != expected_post_keys:
        raise ValueError(f"{label} post-freeze input schema drifted")
    bound_files = {
        path_key: _bound_file(
            post_freeze, path_key, sha_key, f"{label} {path_key}"
        )
        for path_key, sha_key in POST_FREEZE_FILES
    }
    source_build = _object(
        json.loads(bound_files["source_build_manifest"].read_text(encoding="utf-8")),
        f"{label} source-build binding",
    )
    if source_build.get("record_type") != phase_b.source_binding_contract.RECORD_TYPE:
        raise ValueError(f"{label} does not use the frozen Phase-B source binding")

    phase_a_path = _absolute_file(
        screen.get("phase_a_manifest", ""), f"{label} Phase-A manifest"
    )
    if sha256_path(phase_a_path) != screen.get("phase_a_manifest_sha256"):
        raise ValueError(f"{label} Phase-A manifest checksum binding drifted")
    attachments_manifest = _object(
        json.loads(bound_files["attachments_manifest"].read_text(encoding="utf-8")),
        f"{label} attachment manifest",
    )
    ground_truth_manifest = _object(
        json.loads(bound_files["ground_truth_manifest"].read_text(encoding="utf-8")),
        f"{label} ground-truth manifest",
    )
    row_count = int(attachments_manifest.get("row_count", 0))
    attachment_k = int(attachments_manifest.get("top_k", 0))
    ground_truth_width = int(ground_truth_manifest.get("width", 0))
    if row_count <= 0 or attachment_k != phase_b.UPPER_NAVIGATION_TOP_K:
        raise ValueError(f"{label} attachment shape drifted")
    if ground_truth_width <= 0:
        raise ValueError(f"{label} ground-truth width drifted")
    replay_args = argparse.Namespace(
        phase_a_manifest=str(phase_a_path),
        dataset_manifest=str(bound_files["dataset_manifest"]),
        source_build_manifest=str(bound_files["source_build_manifest"]),
        attachments=str(bound_files["attachments"]),
        attachments_manifest=str(bound_files["attachments_manifest"]),
        row_count=row_count,
        attachment_k=attachment_k,
        query_hits=str(bound_files["query_hits"]),
        query_hits_manifest=str(bound_files["query_hits_manifest"]),
        ground_truth=str(bound_files["ground_truth"]),
        ground_truth_manifest=str(bound_files["ground_truth_manifest"]),
        ground_truth_width=ground_truth_width,
    )
    frozen = phase_b.validate_phase_a(replay_args)
    if frozen.candidate_set != candidate_set:
        raise ValueError(f"{label} Phase-A candidate set drifted")
    inputs = phase_b.open_phase_b_inputs(replay_args, frozen)
    dataset_manifest = _object(
        json.loads(inputs.dataset_manifest_path.read_text(encoding="utf-8")),
        f"{label} dataset manifest",
    )
    dataset_record = _object(
        dataset_manifest.get("dataset"), f"{label} dataset identity"
    )
    if dataset_record.get("name") != dataset:
        raise ValueError(
            f"{label} dataset identity drifted: {dataset_record.get('name')!r}"
        )
    if int(post_freeze.get("query_row_count", 0)) != int(
        inputs.query_manifest["row_count"]
    ):
        raise ValueError(f"{label} query-row count drifted")
    rows, replayed = phase_b.evaluate_phase_b(frozen, inputs)

    actual_core = copy.deepcopy(screen)
    actual_core.pop("evaluator_source_code", None)
    actual_identity = _object(actual_core.get("identity_gates"), "identity_gates")
    actual_identity.pop("phase_b_evaluator_source_code_sha256", None)
    actual_outputs = _object(actual_core.get("outputs"), "outputs")
    actual_outputs.pop("source_dir", None)
    actual_outputs.pop("source_record", None)
    if actual_core != replayed:
        mismatched = sorted(
            key
            for key in set(actual_core) | set(replayed)
            if actual_core.get(key) != replayed.get(key)
        )
        raise ValueError(f"{label} deep replay mismatch in fields: {mismatched}")
    _validate_phase_b_outputs(path, screen, rows, label)
    return replayed, {
        "dataset_manifest_path": str(inputs.dataset_manifest_path),
        "dataset_manifest_sha256": inputs.dataset_manifest_sha256,
        "dataset_sha256": inputs.dataset_sha256,
        "source_build_manifest_path": str(inputs.source_build_manifest_path),
        "source_build_manifest_sha256": inputs.source_build_manifest_sha256,
    }


def _load_screen(
    raw: str, label: str, candidate_set: str, dataset: str
) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    path = _absolute_file(raw, label)
    digest = sha256_path(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != digest:
        raise ValueError(f"{label} checksum sidecar mismatch")
    _require_read_only(path, label)
    _require_read_only(sidecar, f"{label} checksum sidecar")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if (
        value.get("format_version") != 1
        or value.get("stage") != "post_freeze_evaluation"
        or value.get("candidate_set") != candidate_set
        or value.get("identity_all_pass") is not True
    ):
        raise ValueError(f"{label} is not an eligible canonical Phase-B screen")
    replayed, provenance = _deep_replay_screen(
        path,
        value,
        candidate_set=candidate_set,
        dataset=dataset,
        label=label,
    )
    return path, digest, replayed, provenance


def _candidate(screen: dict[str, Any], name: str, label: str) -> dict[str, Any]:
    candidates = screen.get("candidates")
    if not isinstance(candidates, dict) or not isinstance(candidates.get(name), dict):
        raise ValueError(f"{label} lacks candidate {name}")
    return candidates[name]


def _same_formal_grid_candidate(
    formal: dict[str, Any], grid: dict[str, Any], dataset: str
) -> None:
    formal_reference = formal.get("reference")
    grid_reference = grid.get("reference")
    if not isinstance(formal_reference, dict) or not isinstance(grid_reference, dict):
        raise ValueError(f"{dataset} reference record is missing")
    formal_candidate = formal.get("candidate")
    grid_candidate = _candidate(grid, SELECTED_GRID_NAME, f"{dataset} grid")
    if not isinstance(formal_candidate, dict):
        raise ValueError(f"{dataset} formal candidate is missing")
    reference_parity_fields = (
        "owner_sha256",
        "topology_metrics",
        "actual_metrics",
        "materialization_parity",
        "graph_topology_gates",
        "query_topology_gates",
        "graph_topology_all_pass",
        "query_topology_all_pass",
        "topology_all_pass",
        "physical_copy_load_improves_over_reference",
        "screen_eligible",
        "materialization_eligible",
    )
    candidate_parity_fields = reference_parity_fields + (
        "base_owner_sha256",
        "physical_copy_load_gate",
        "selected_adoption_candidate",
    )
    checks = {
        "reference_owner_sha256": (
            formal_reference.get("owner_sha256"), grid_reference.get("owner_sha256")
        ),
        "candidate_owner_sha256": (
            formal_candidate.get("owner_sha256"), grid_candidate.get("owner_sha256")
        ),
        **{
            f"reference_{field}": (
                formal_reference.get(field),
                grid_reference.get(field),
            )
            for field in reference_parity_fields
        },
        **{
            f"candidate_{field}": (
                formal_candidate.get(field),
                grid_candidate.get(field),
            )
            for field in candidate_parity_fields
        },
    }
    mismatches = {
        name: {"formal": left, "grid": right}
        for name, (left, right) in checks.items()
        if left != right
    }
    if mismatches:
        raise ValueError(f"{dataset} formal/grid 9/4 parity failed: {mismatches}")
    if formal_candidate.get("materialization_eligible") is not True:
        raise ValueError(f"{dataset} formal C_CNBR is not materialization eligible")
    if grid_candidate.get("screen_eligible") is not True:
        raise ValueError(f"{dataset} grid CNBR_9_4 is not screen eligible")


def run(args: argparse.Namespace) -> tuple[Path, str]:
    source_path = Path(__file__).resolve()
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        raise ValueError("output-dir must be an absolute path")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    replay_source_sha256 = {
        name: sha256_path(path)
        for name, path in phase_b._phase_b_source_paths().items()
    }

    inputs: dict[str, dict[str, Any]] = {}
    screens: dict[str, dict[str, dict[str, Any]]] = {}
    provenance: dict[str, dict[str, dict[str, Any]]] = {}
    screen_paths: set[Path] = set()
    for dataset, grid_raw, formal_raw in (
        ("sift1m", args.sift_grid, args.sift_formal),
        ("glove-200-angular", args.glove_grid, args.glove_formal),
    ):
        grid_path, grid_sha, grid, grid_provenance = _load_screen(
            grid_raw, f"{dataset} grid", "frozen-grid", dataset
        )
        formal_path, formal_sha, formal, formal_provenance = _load_screen(
            formal_raw, f"{dataset} formal", "formal", dataset
        )
        for path in (grid_path, formal_path):
            if path in screen_paths:
                raise ValueError("selection requires four distinct Phase-B screens")
            screen_paths.add(path)
        candidates = grid.get("candidates")
        if not isinstance(candidates, dict) or set(candidates) != set(GRID_NAMES):
            raise ValueError(f"{dataset} grid candidate family drifted")
        _same_formal_grid_candidate(formal, grid, dataset)
        grid_post = _object(
            grid.get("post_freeze_evaluation_inputs"),
            f"{dataset} grid post-freeze inputs",
        )
        formal_post = _object(
            formal.get("post_freeze_evaluation_inputs"),
            f"{dataset} formal post-freeze inputs",
        )
        semantic_input_fields = (
            "dataset_sha256",
            "attachments_sha256",
            "query_hits_sha256",
            "query_row_count",
            "ground_truth_sha256",
        )
        if any(
            grid_post.get(field) != formal_post.get(field)
            for field in semantic_input_fields
        ):
            raise ValueError(f"{dataset} formal/grid evaluation corpus drifted")
        if (
            grid_provenance["dataset_sha256"]
            != formal_provenance["dataset_sha256"]
        ):
            raise ValueError(f"{dataset} formal/grid dataset provenance drifted")
        inputs[dataset] = {
            "grid_screen": str(grid_path),
            "grid_screen_sha256": grid_sha,
            "grid_phase_a_manifest_sha256": grid.get("phase_a_manifest_sha256"),
            "formal_screen": str(formal_path),
            "formal_screen_sha256": formal_sha,
            "formal_phase_a_manifest_sha256": formal.get("phase_a_manifest_sha256"),
        }
        screens[dataset] = {"grid": grid, "formal": formal}
        provenance[dataset] = {
            "grid": grid_provenance,
            "formal": formal_provenance,
        }

    if (
        provenance["sift1m"]["grid"]["dataset_sha256"]
        == provenance["glove-200-angular"]["grid"]["dataset_sha256"]
    ):
        raise ValueError("SIFT and GloVe screens bind the same dataset bytes")

    rows: list[dict[str, Any]] = []
    dual_pass: list[str] = []
    for name in GRID_NAMES:
        dataset_pass: dict[str, bool] = {}
        row: dict[str, Any] = {"name": name}
        for dataset in ("sift1m", "glove-200-angular"):
            candidate = _candidate(screens[dataset]["grid"], name, f"{dataset} grid")
            passed = candidate.get("screen_eligible") is True
            dataset_pass[dataset] = passed
            metrics = candidate.get("actual_metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"{dataset} {name} lacks actual metrics")
            prefix = "sift" if dataset == "sift1m" else "glove"
            row.update(
                {
                    f"{prefix}_screen_eligible": passed,
                    f"{prefix}_graph_topology_all_pass": candidate.get(
                        "graph_topology_all_pass"
                    ),
                    f"{prefix}_query_topology_all_pass": candidate.get(
                        "query_topology_all_pass"
                    ),
                    f"{prefix}_physical_copy_load_max_over_mean": metrics.get(
                        "physical_copy_load_max_over_mean"
                    ),
                    f"{prefix}_routed_shards_mean": metrics.get("routed_shards_mean"),
                    f"{prefix}_gt_routing_coverage_mean": metrics.get(
                        "gt_routing_coverage_mean"
                    ),
                    f"{prefix}_expansion_ratio": metrics.get("expansion_ratio"),
                }
            )
        row["dual_dataset_all_gate_pass"] = all(dataset_pass.values())
        if row["dual_dataset_all_gate_pass"]:
            dual_pass.append(name)
        rows.append(row)

    if dual_pass != [SELECTED_GRID_NAME]:
        raise ValueError(
            "frozen family no longer has unique dual-dataset winner CNBR_9_4: "
            f"{dual_pass}"
        )
    if sha256_path(source_path) != source_sha256:
        raise ValueError("selection synthesizer source changed during evaluation")
    if any(
        sha256_path(phase_b._phase_b_source_paths()[name]) != digest
        for name, digest in replay_source_sha256.items()
    ):
        raise ValueError("Phase-B replay source changed during selection")

    output_dir.mkdir(parents=True, exist_ok=False)
    source_dir = output_dir / "source"
    source_dir.mkdir()
    source_copy_path = source_dir / source_path.name
    with source_copy_path.open("xb") as handle:
        handle.write(source_bytes)
    source_record_path = output_dir / "selection-source-code.record.json"
    source_record = {
        "format_version": 1,
        "record_type": "cnbr_selection_synthesizer_source_code",
        "files": {
            "synthesizer": {
                "path": source_copy_path.relative_to(output_dir).as_posix(),
                "sha256": source_sha256,
                "size_bytes": len(source_bytes),
            }
        },
    }
    source_record_path.write_text(
        json.dumps(source_record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "selection.csv"
    manifest_path = output_dir / "selection-manifest.json"
    sidecar_path = output_dir / "selection-manifest.json.sha256"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "format_version": 1,
        "stage": "cnbr_dual_dataset_frozen_family_selection",
        "selection_status": "post_exploratory_two_dataset_family_selection",
        "strict_cross_dataset_holdout_claim": False,
        "candidate_family": list(GRID_NAMES),
        "dual_dataset_all_gate_pass": dual_pass,
        "selected_grid_candidate": SELECTED_GRID_NAME,
        "selected_formal_candidate": FORMAL_NAME,
        "selection_basis": "unique_all_gate_pass_on_both_sift1m_and_glove_200_angular",
        "new_ratio_or_retuning_after_selection_forbidden": True,
        "synthesizer_source_code": {
            "record": {
                "path": source_record_path.name,
                "sha256": sha256_path(source_record_path),
                "size_bytes": source_record_path.stat().st_size,
            },
            "files": source_record["files"],
        },
        "inputs": inputs,
        "rows": rows,
        "outputs": {
            "selection_csv": csv_path.name,
            "selection_manifest": manifest_path.name,
            "selection_manifest_sidecar": sidecar_path.name,
            "source_dir": source_dir.name,
            "source_record": source_record_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256_path(manifest_path)
    sidecar_path.write_text(digest + "\n", encoding="ascii")
    for path in (
        source_copy_path,
        source_record_path,
        csv_path,
        manifest_path,
        sidecar_path,
    ):
        os.chmod(path, 0o444)
    return manifest_path, digest


def main(argv: list[str] | None = None) -> None:
    path, digest = run(parse_args(argv))
    print(f"selection_manifest={path}")
    print(f"selection_manifest_sha256={digest}")


if __name__ == "__main__":
    main()
