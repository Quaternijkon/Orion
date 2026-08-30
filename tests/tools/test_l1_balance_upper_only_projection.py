from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.l1_balance import project_upper_only_artifact as projection


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_artifact() -> dict:
    return {
        "format_version": 1,
        "generation": 7,
        "vector_schema": {
            "vector_name": "",
            "dimension": 2,
            "distance": "Cosine",
            "datatype": "float32",
        },
        "shard_count": 32,
        "layout_sha256": "a" * 64,
        "logical_point_count": 64,
        "physical_point_count": 80,
        "upper_k": 2,
        "upper_ef_search": 4,
        "dynamic_ef_base": 50,
        "dynamic_ef_factor": 14,
        "upper_nodes": [
            {"label": 10, "vector": [1.0, 2.0], "shard_membership": [3, 9]},
            {"label": 20, "vector": [4.0, 5.0], "shard_membership": [7]},
        ],
        "upper_graph": {
            "entry_point": 10,
            "max_level": 0,
            "nodes": [
                {"label": 10, "neighbors_by_level": [[20]]},
                {"label": 20, "neighbors_by_level": [[10]]},
            ],
        },
    }


def _source_build_manifest(source_path: Path) -> dict:
    return {
        "mode": "production_bundle",
        "outputs": {
            "files": {
                source_path.name: {
                    "sha256": _sha256(source_path),
                    "size_bytes": source_path.stat().st_size,
                }
            },
            "production_artifact": source_path.name,
        },
        "parameters": {
            "upper_sample_seed": 101,
            "upper_m": 32,
            "upper_ef_construction": 200,
            "upper_graph_seed": 303,
            "balance_mode": "capacity_constrained",
            "enable_topology_refinement": True,
        }
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sift_replay_bridge(
    tmp_path: Path, source_path: Path
) -> tuple[Path, Path]:
    graphless_path = tmp_path / "graphless-orion.json"
    graphless_path.write_bytes(b"graphless fixture\n")
    original_build_path = tmp_path / "graphless-build-manifest.json"
    original_build = {
        "mode": "graphless_only",
        "outputs": {
            "files": {
                graphless_path.name: {
                    "sha256": _sha256(graphless_path),
                    "size_bytes": graphless_path.stat().st_size,
                }
            },
            "graphless_artifact": graphless_path.name,
            "production_artifact": None,
        },
        "parameters": {
            "upper_sample_seed": 101,
            "upper_m": 32,
            "upper_ef_construction": 200,
            "upper_graph_seed": 303,
            "balance_mode": "capacity_constrained",
        },
    }
    _write_json(original_build_path, original_build)

    replay_artifact_path = tmp_path / "generation-replay.json"
    replay_artifact_path.write_bytes(source_path.read_bytes())
    replay_manifest_path = tmp_path / "source-build-replay.json"
    replay_manifest = {
        "mode": projection.REPLAY_ATTESTATION_MODE,
        "outputs": {
            "files": {
                source_path.name: {
                    "path": str(source_path),
                    "sha256": _sha256(source_path),
                    "size_bytes": source_path.stat().st_size,
                },
                replay_artifact_path.name: {
                    "path": str(replay_artifact_path),
                    "sha256": _sha256(replay_artifact_path),
                    "size_bytes": replay_artifact_path.stat().st_size,
                },
                graphless_path.name: {
                    "path": str(graphless_path),
                    "sha256": _sha256(graphless_path),
                    "size_bytes": graphless_path.stat().st_size,
                },
            },
            "production_artifact": source_path.name,
            "replay_artifact": replay_artifact_path.name,
            "source_graphless_artifact": graphless_path.name,
        },
        "provenance": {
            "artifact_byte_identity_reproduced": True,
            "builder_command": [
                "orion_build_artifact",
                str(graphless_path),
                str(replay_artifact_path),
                "--seed",
                "303",
                "--m",
                "32",
                "--ef",
                "200",
            ],
            "source_build_manifest_path": str(original_build_path),
            "source_build_manifest_sha256": _sha256(original_build_path),
        },
    }
    _write_json(replay_manifest_path, replay_manifest)
    return replay_manifest_path, original_build_path


def test_projection_removes_layout_information_and_preserves_upper_identity(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    output_path = tmp_path / "upper-only.json"
    manifest_path = tmp_path / "upper-only.manifest.json"
    source_build_path = tmp_path / "source-build.json"
    source = _source_artifact()
    _write_json(source_path, source)
    _write_json(source_build_path, _source_build_manifest(source_path))

    artifact, artifact_sha, manifest, manifest_sha = projection.run(
        argparse.Namespace(
            source_artifact=str(source_path),
            source_build_manifest=str(source_build_path),
            output_artifact=str(output_path),
            output_manifest=str(manifest_path),
        )
    )

    assert artifact == output_path.resolve()
    assert manifest == manifest_path.resolve()
    assert artifact_sha == _sha256(output_path)
    assert manifest_sha == _sha256(manifest_path)
    assert (
        output_path.with_name(output_path.name + ".sha256").read_text().strip()
        == artifact_sha
    )
    assert (
        manifest_path.with_name(manifest_path.name + ".sha256").read_text().strip()
        == manifest_sha
    )

    projected = json.loads(output_path.read_text(encoding="utf-8"))
    assert projected["upper_graph"] == source["upper_graph"]
    assert [node["label"] for node in projected["upper_nodes"]] == [10, 20]
    assert [node["vector"] for node in projected["upper_nodes"]] == [
        [1.0, 2.0],
        [4.0, 5.0],
    ]
    assert all(node["shard_membership"] == [0] for node in projected["upper_nodes"])
    assert projected["layout_sha256"] == "0" * 64
    assert projected["physical_point_count"] == projected["logical_point_count"]

    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(record["source_artifact"]) == {"sha256", "size_bytes", "role"}
    assert record["source_artifact"]["sha256"] == _sha256(source_path)
    assert record["upper_build_provenance"] == {
        "source_manifest_sha256": _sha256(source_build_path),
        "source_manifest_size_bytes": source_build_path.stat().st_size,
        "parameters": {
            "upper_sample_seed": 101,
            "upper_m": 32,
            "upper_ef_construction": 200,
            "upper_graph_seed": 303,
        },
    }
    assert record["redaction_proof"]["historical_membership_values_emitted"] is False
    assert record["redaction_proof"]["historical_membership_distribution_emitted"] is False
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "[3, 9]" not in manifest_text
    assert str(source_path) not in manifest_text
    assert str(source_build_path) not in manifest_text
    assert "capacity_constrained" not in manifest_text
    assert "enable_topology_refinement" not in manifest_text
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o444


def test_projection_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_build_path = tmp_path / "source-build.json"
    _write_json(source_path, _source_artifact())
    _write_json(source_build_path, _source_build_manifest(source_path))
    outputs = []
    for suffix in ("a", "b"):
        output = tmp_path / f"upper-{suffix}.json"
        manifest = tmp_path / f"upper-{suffix}.manifest.json"
        projection.run(
            argparse.Namespace(
                source_artifact=str(source_path),
                source_build_manifest=str(source_build_path),
                output_artifact=str(output),
                output_manifest=str(manifest),
            )
        )
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        projection.run(
            argparse.Namespace(
                source_artifact=str(source_path),
                source_build_manifest=str(source_build_path),
                output_artifact=str(tmp_path / "upper-a.json"),
                output_manifest=str(tmp_path / "new.manifest.json"),
            )
        )


def test_projection_fails_closed_on_incomplete_source_membership(tmp_path: Path) -> None:
    source = _source_artifact()
    source["upper_nodes"][0]["shard_membership"] = []
    source_path = tmp_path / "bad.json"
    source_build_path = tmp_path / "source-build.json"
    _write_json(source_path, source)
    _write_json(source_build_path, _source_build_manifest(source_path))
    with pytest.raises(ValueError, match="has no source membership"):
        projection.run(
            argparse.Namespace(
                source_artifact=str(source_path),
                source_build_manifest=str(source_build_path),
                output_artifact=str(tmp_path / "upper.json"),
                output_manifest=str(tmp_path / "upper.manifest.json"),
            )
        )


@pytest.mark.parametrize("missing_key", projection.UPPER_BUILD_PARAMETER_KEYS)
def test_projection_fails_closed_when_upper_build_parameter_is_missing(
    tmp_path: Path, missing_key: str
) -> None:
    source_path = tmp_path / "source.json"
    source_build_path = tmp_path / "source-build.json"
    _write_json(source_path, _source_artifact())
    source_build = _source_build_manifest(source_path)
    del source_build["parameters"][missing_key]
    _write_json(source_build_path, source_build)

    with pytest.raises(ValueError, match=missing_key):
        projection.run(
            argparse.Namespace(
                source_artifact=str(source_path),
                source_build_manifest=str(source_build_path),
                output_artifact=str(tmp_path / "upper.json"),
                output_manifest=str(tmp_path / "upper.manifest.json"),
            )
        )


def test_projection_rejects_unrelated_parameter_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_build_path = tmp_path / "unrelated-build.json"
    _write_json(source_path, _source_artifact())
    source_build = _source_build_manifest(source_path)
    source_build["outputs"]["files"][source_path.name]["sha256"] = "f" * 64
    _write_json(source_build_path, source_build)

    with pytest.raises(ValueError, match="does not bind source artifact"):
        projection.run(
            argparse.Namespace(
                source_artifact=str(source_path),
                source_build_manifest=str(source_build_path),
                output_artifact=str(tmp_path / "upper.json"),
                output_manifest=str(tmp_path / "upper.manifest.json"),
            )
        )


def test_projection_accepts_sift_replay_bridge_but_not_graphless_manifest_alone(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    _write_json(source_path, _source_artifact())
    replay_manifest_path, original_build_path = _sift_replay_bridge(
        tmp_path, source_path
    )
    output_path = tmp_path / "upper.json"
    manifest_path = tmp_path / "upper.manifest.json"

    projection.run(
        argparse.Namespace(
            source_artifact=str(source_path),
            source_build_manifest=str(replay_manifest_path),
            output_artifact=str(output_path),
            output_manifest=str(manifest_path),
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["upper_build_provenance"] == {
        "source_manifest_sha256": _sha256(replay_manifest_path),
        "source_manifest_size_bytes": replay_manifest_path.stat().st_size,
        "parameters": {
            "upper_sample_seed": 101,
            "upper_m": 32,
            "upper_ef_construction": 200,
            "upper_graph_seed": 303,
        },
    }
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert str(replay_manifest_path) not in manifest_text
    assert str(original_build_path) not in manifest_text

    with pytest.raises(ValueError, match="lacks production_artifact"):
        projection.run(
            argparse.Namespace(
                source_artifact=str(source_path),
                source_build_manifest=str(original_build_path),
                output_artifact=str(tmp_path / "upper-direct.json"),
                output_manifest=str(tmp_path / "upper-direct.manifest.json"),
            )
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("source_manifest_sha", "source manifest checksum drifted"),
        ("builder_seed", "--seed differs from source manifest"),
        ("replay_artifact_sha", "does not bind source artifact"),
    ],
)
def test_projection_rejects_tampered_sift_replay_bridge(
    tmp_path: Path, tamper: str, message: str
) -> None:
    source_path = tmp_path / "source.json"
    _write_json(source_path, _source_artifact())
    replay_manifest_path, _original_build_path = _sift_replay_bridge(
        tmp_path, source_path
    )
    replay = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    if tamper == "source_manifest_sha":
        replay["provenance"]["source_build_manifest_sha256"] = "0" * 64
    elif tamper == "builder_seed":
        seed_index = replay["provenance"]["builder_command"].index("--seed")
        replay["provenance"]["builder_command"][seed_index + 1] = "999"
    else:
        replay_name = replay["outputs"]["replay_artifact"]
        replay["outputs"]["files"][replay_name]["sha256"] = "0" * 64
    _write_json(replay_manifest_path, replay)

    with pytest.raises(ValueError, match=message):
        projection.run(
            argparse.Namespace(
                source_artifact=str(source_path),
                source_build_manifest=str(replay_manifest_path),
                output_artifact=str(tmp_path / "upper.json"),
                output_manifest=str(tmp_path / "upper.manifest.json"),
            )
        )
