#!/usr/bin/env python3
"""Freeze a checksum-only Phase-B binding for a neutral upper projection.

This utility does not rebuild or reinterpret Orion data.  It bridges the
neutral projected upper artifact back to its checksum-bound source build and
binds the exact full L0 vector bytes used by attachment export.  Canonical raw
vectors are accepted directly; transformed cosine rows additionally require a
source-build output, checksums file, and import-manifest lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
RECORD_TYPE = "phase_b_source_build_binding"


def binding_contract() -> dict[str, bool]:
    return {
        "read_only_checksum_binding_only": True,
        "does_not_claim_original_build_log": True,
        "does_not_rebuild_artifact": True,
        "does_not_read_attachments": True,
        "does_not_change_owner_or_multi_assignment": True,
        "neutral_projection_lineage_required": True,
        "source_build_vector_lineage_required": True,
        "transformed_vectors_require_import_and_checksums": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--source-build-manifest", required=True)
    parser.add_argument("--upper-only-projection-manifest", required=True)
    parser.add_argument("--output", required=True)
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
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _file_record(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
        "role": role,
    }


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _normalize_distance(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"l2", "euclidean"}:
        return "euclid"
    return normalized


def _parse_checksums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="ascii").splitlines(), 1
    ):
        if not raw:
            continue
        parts = raw.split("  ", 1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
            or Path(parts[1]).name != parts[1]
        ):
            raise ValueError(f"invalid checksums entry at {path}:{line_number}")
        if parts[1] in records:
            raise ValueError(
                f"duplicate checksums entry at {path}:{line_number}: {parts[1]}"
            )
        records[parts[1]] = parts[0]
    return records


def _manifest_sidecar_sha256(path: Path, label: str) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"{label} checksum sidecar is missing")
    digest = sha256_path(path)
    if sidecar.read_text(encoding="ascii").strip() != digest:
        raise ValueError(f"{label} checksum sidecar drifted")
    return digest


def validate_source_build_vector_lineage(
    *,
    dataset_manifest_path: Path,
    dataset_manifest: dict[str, Any],
    artifact_path: Path,
    artifact: dict[str, Any],
    vectors_path: Path,
    source_build_manifest_path: Path,
    projection_manifest_path: Path,
) -> dict[str, Any]:
    """Replay the projection/build/vector bridge without opening attachments."""

    projection_sha256 = _manifest_sidecar_sha256(
        projection_manifest_path, "upper-only projection manifest"
    )
    projection = _load_object(
        projection_manifest_path, "upper-only projection manifest"
    )
    if (
        projection.get("format_version") != 1
        or projection.get("record_type") != "orion_upper_only_projection"
    ):
        raise ValueError("upper-only projection manifest contract drifted")
    projection_contract = _mapping(
        projection.get("contract"), "upper-only projection contract"
    )
    redaction = _mapping(
        projection.get("redaction_proof"), "upper-only projection redaction proof"
    )
    if (
        projection_contract.get("historical_memberships_removed") is not True
        or projection_contract.get("historical_layout_metadata_removed") is not True
        or redaction.get("every_output_membership_is_exact_neutral_sentinel")
        is not True
    ):
        raise ValueError("upper-only projection is not neutral")
    artifact_sha256 = sha256_path(artifact_path)
    projection_output = _mapping(
        projection.get("output_artifact"), "upper-only projection output"
    )
    if (
        Path(str(projection_output.get("path", ""))).expanduser().resolve()
        != artifact_path
        or projection_output.get("sha256") != artifact_sha256
        or projection_output.get("size_bytes") != artifact_path.stat().st_size
    ):
        raise ValueError("upper-only projection output binding drifted")

    source_build_sha256 = sha256_path(source_build_manifest_path)
    provenance = _mapping(
        projection.get("upper_build_provenance"),
        "upper-only projection source-build provenance",
    )
    if (
        provenance.get("source_manifest_sha256") != source_build_sha256
        or provenance.get("source_manifest_size_bytes")
        != source_build_manifest_path.stat().st_size
    ):
        raise ValueError("projection/source-build checksum lineage drifted")

    source_build = _load_object(source_build_manifest_path, "source build manifest")
    if source_build.get("format_version") != 1:
        raise ValueError("source build manifest format_version must be 1")
    build_dataset = _mapping(source_build.get("dataset"), "source build dataset")
    dataset = _mapping(dataset_manifest.get("dataset"), "dataset manifest dataset")
    source_dataset = _mapping(
        dataset_manifest.get("source_dataset"), "dataset manifest source dataset"
    )
    dataset_sha256 = dataset.get("sha256")
    rows = int(dataset.get("expected_train_rows", 0))
    dimension = int(dataset.get("dimension", 0))
    if (
        not isinstance(dataset_sha256, str)
        or source_dataset.get("sha256") != dataset_sha256
        or build_dataset.get("sha256") != dataset_sha256
        or build_dataset.get("dimension") != dimension
        or build_dataset.get("train_rows_used") != rows
    ):
        raise ValueError("source-build dataset lineage drifted")

    build_outputs = _mapping(source_build.get("outputs"), "source build outputs")
    build_files = _mapping(
        build_outputs.get("files"), "source build output files"
    )
    production_name = build_outputs.get("production_artifact")
    if not isinstance(production_name, str):
        raise ValueError("source build does not name a production artifact")
    production_record = _mapping(
        build_files.get(production_name), "source build production artifact"
    )
    projection_source = _mapping(
        projection.get("source_artifact"), "upper-only projection source artifact"
    )
    if (
        production_record.get("sha256") != projection_source.get("sha256")
        or production_record.get("size_bytes") != projection_source.get("size_bytes")
    ):
        raise ValueError("projection source is not the source-build production artifact")

    vectors_sha256 = sha256_path(vectors_path)
    vector_name = vectors_path.name
    vector_output = _mapping(
        build_files.get(vector_name), "source build vector output"
    )
    expected_vector_size = rows * dimension * 4
    if (
        vector_output.get("sha256") != vectors_sha256
        or vector_output.get("size_bytes") != vectors_path.stat().st_size
        or vectors_path.stat().st_size != expected_vector_size
    ):
        raise ValueError("source-build vector checksum lineage drifted")
    if "path" in vector_output and Path(
        str(vector_output.get("path", ""))
    ).expanduser().resolve() != vectors_path:
        raise ValueError("source-build vector path lineage drifted")

    files = _mapping(dataset_manifest.get("files"), "dataset manifest files")
    canonical_vectors = _mapping(
        files.get("vectors"), "dataset manifest canonical vectors"
    )
    canonical_vectors_path = _absolute_file(
        str(canonical_vectors.get("path", "")), "canonical dataset vectors"
    )
    if (
        canonical_vectors.get("size_bytes") != expected_vector_size
        or canonical_vectors_path.stat().st_size != expected_vector_size
    ):
        raise ValueError("canonical dataset vector shape drifted")
    canonical_exact = (
        canonical_vectors.get("sha256") == vectors_sha256
        and canonical_vectors_path == vectors_path
    )

    artifact_schema = _mapping(artifact.get("vector_schema"), "artifact vector schema")
    dataset_distance = _normalize_distance(
        dataset.get("metric") or dataset.get("hnsw_space")
    )
    artifact_distance = _normalize_distance(artifact_schema.get("distance"))
    parameters = source_build.get("parameters")
    build_distance = (
        _normalize_distance(parameters.get("vector_distance"))
        if isinstance(parameters, dict)
        else ""
    )
    if dataset_distance != artifact_distance:
        raise ValueError("dataset/artifact distance lineage drifted")

    checksums_binding: dict[str, Any] | None = None
    import_binding: dict[str, Any] | None = None
    if canonical_exact:
        preprocessing = "identity_float32_row_order"
        mode = "canonical_dataset_vectors_exact"
    else:
        if dataset_distance != "cosine" or build_distance != "cosine":
            raise ValueError("transformed vectors lack three-way cosine distance lineage")
        checksums_path = source_build_manifest_path.with_name("checksums.sha256")
        if not checksums_path.is_file():
            raise ValueError("transformed vectors lack source-build checksums")
        checksums = _parse_checksums(checksums_path)
        if (
            checksums.get(source_build_manifest_path.name) != source_build_sha256
            or checksums.get(vector_name) != vectors_sha256
        ):
            raise ValueError("transformed-vector checksums lineage drifted")
        import_name = build_outputs.get("import_manifest")
        if not isinstance(import_name, str):
            raise ValueError("transformed vectors lack a source-build import manifest")
        import_record = _mapping(
            build_files.get(import_name), "source build import manifest output"
        )
        import_path = source_build_manifest_path.with_name(import_name)
        if (
            not import_path.is_file()
            or import_record.get("sha256") != sha256_path(import_path)
            or import_record.get("size_bytes") != import_path.stat().st_size
            or checksums.get(import_name) != import_record.get("sha256")
        ):
            raise ValueError("source-build import manifest checksum lineage drifted")
        if "path" in import_record and Path(
            str(import_record.get("path", ""))
        ).expanduser().resolve() != import_path:
            raise ValueError("source-build import manifest path lineage drifted")
        import_manifest = _load_object(import_path, "source build import manifest")
        if (
            import_manifest.get("vectors_file") != vector_name
            or import_manifest.get("vectors_sha256") != vectors_sha256
            or import_manifest.get("orion_artifact_file") != production_name
            or import_manifest.get("orion_artifact_sha256")
            != production_record.get("sha256")
            or import_manifest.get("point_count") != rows
            or import_manifest.get("dimension") != dimension
        ):
            raise ValueError("source-build import/vector lineage drifted")
        preprocessing = "cosine_l2_normalize_nonzero_rows_float32"
        mode = "source_build_bound_cosine_normalized"
        checksums_binding = _file_record(
            checksums_path, role="source_build_checksums"
        )
        import_binding = _file_record(
            import_path, role="source_build_import_manifest"
        )

    return {
        "mode": mode,
        "preprocessing_contract": preprocessing,
        "dataset_manifest": _file_record(
            dataset_manifest_path, role="canonical_dataset_manifest"
        ),
        "dataset_sha256": dataset_sha256,
        "source_build_manifest": _file_record(
            source_build_manifest_path, role="source_build_manifest"
        ),
        "upper_only_projection_manifest": _file_record(
            projection_manifest_path, role="neutral_upper_only_projection_manifest"
        ),
        "upper_only_projection_manifest_sha256": projection_sha256,
        "source_production_artifact": {
            "name": production_name,
            "sha256": production_record.get("sha256"),
            "size_bytes": production_record.get("size_bytes"),
        },
        "projected_artifact": _file_record(
            artifact_path, role="neutral_projected_upper_artifact"
        ),
        "vectors": _file_record(vectors_path, role="ordered_full_l0_vectors"),
        "canonical_dataset_vectors": {
            "path": str(canonical_vectors_path),
            "sha256": canonical_vectors.get("sha256"),
            "size_bytes": canonical_vectors.get("size_bytes"),
            "byte_identical_to_build_vectors": canonical_exact,
        },
        "distance": {
            "dataset": dataset_distance,
            "artifact": artifact_distance,
            "source_build": build_distance or artifact_distance,
        },
        "checksums": checksums_binding,
        "import_manifest": import_binding,
    }


def run(args: argparse.Namespace) -> tuple[Path, str]:
    dataset_manifest_path = _absolute_file(
        args.dataset_manifest, "dataset-manifest"
    )
    artifact_path = _absolute_file(args.artifact, "artifact")
    vectors_path = _absolute_file(args.vectors, "vectors")
    source_build_manifest_path = _absolute_file(
        args.source_build_manifest, "source-build-manifest"
    )
    projection_manifest_path = _absolute_file(
        args.upper_only_projection_manifest, "upper-only-projection-manifest"
    )
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        raise ValueError("output must be an absolute path")
    output_path = output_path.resolve()
    sidecar_path = output_path.with_name(output_path.name + ".sha256")
    if output_path.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite source binding: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output_path.parent}")

    dataset_manifest = _load_object(dataset_manifest_path, "dataset manifest")
    dataset = dataset_manifest.get("dataset")
    source_dataset = dataset_manifest.get("source_dataset")
    files = dataset_manifest.get("files")
    if not isinstance(dataset, dict) or not isinstance(source_dataset, dict):
        raise ValueError("dataset manifest lacks canonical dataset identity")
    if not isinstance(files, dict) or not isinstance(files.get("vectors"), dict):
        raise ValueError("dataset manifest lacks a vector file record")
    dataset_sha256 = dataset.get("sha256")
    if (
        not isinstance(dataset_sha256, str)
        or source_dataset.get("sha256") != dataset_sha256
    ):
        raise ValueError("dataset manifest checksum identity is inconsistent")
    source_dataset_path = _absolute_file(
        str(source_dataset.get("path", "")), "source dataset"
    )
    if sha256_path(source_dataset_path) != dataset_sha256:
        raise ValueError("source dataset checksum drifted")

    artifact = _load_object(artifact_path, "artifact")
    expected_rows = int(dataset.get("expected_train_rows", 0))
    expected_dimension = int(dataset.get("dimension", 0))
    vector_record = files["vectors"]
    if vector_record.get("size_bytes") != expected_rows * expected_dimension * 4:
        raise ValueError("canonical dataset vector shape drifted")
    vector_schema = artifact.get("vector_schema")
    if (
        expected_rows <= 0
        or expected_dimension <= 0
        or artifact.get("logical_point_count") != expected_rows
        or not isinstance(vector_schema, dict)
        or vector_schema.get("dimension") != expected_dimension
    ):
        raise ValueError("artifact shape differs from the canonical dataset")

    artifact_record = _file_record(
        artifact_path, role="neutral_projected_upper_artifact"
    )
    vectors_record = _file_record(vectors_path, role="ordered_full_l0_vectors")
    vector_lineage = validate_source_build_vector_lineage(
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest=dataset_manifest,
        artifact_path=artifact_path,
        artifact=artifact,
        vectors_path=vectors_path,
        source_build_manifest_path=source_build_manifest_path,
        projection_manifest_path=projection_manifest_path,
    )
    record = {
        "format_version": FORMAT_VERSION,
        "record_type": RECORD_TYPE,
        "contract": binding_contract(),
        "dataset": {
            "manifest_path": str(dataset_manifest_path),
            "manifest_sha256": sha256_path(dataset_manifest_path),
            "path": str(source_dataset_path),
            "sha256": dataset_sha256,
            "size_bytes": source_dataset_path.stat().st_size,
            "dimension": expected_dimension,
            "train_rows_used": expected_rows,
        },
        "vector_lineage": vector_lineage,
        "outputs": {
            "files": {
                artifact_path.name: artifact_record,
                vectors_path.name: vectors_record,
            },
            "production_artifact": artifact_path.name,
            "vectors": vectors_path.name,
        },
    }
    encoded = (
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with output_path.open("xb") as handle:
        handle.write(encoded)
    manifest_sha256 = sha256_path(output_path)
    with sidecar_path.open("xb") as handle:
        handle.write((manifest_sha256 + "\n").encode("ascii"))
    os.chmod(output_path, 0o444)
    os.chmod(sidecar_path, 0o444)
    return output_path, manifest_sha256


def main(argv: list[str] | None = None) -> None:
    path, digest = run(parse_args(argv))
    print(f"source_binding={path}")
    print(f"source_binding_sha256={digest}")


if __name__ == "__main__":
    main()
