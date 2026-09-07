from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "experiments/c6/scripts/c6_prepare.py"


def load_module():
    spec = importlib.util.spec_from_file_location("c6_prepare", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bundle(tmp_path: Path) -> Path:
    assignments = tmp_path / "orion_numeric_import.assignments.jsonl"
    assignments.write_text(
        '{"id":0,"shards":[0]}\n'
        '{"id":1,"shards":[1,0]}\n'
        '{"id":2,"shards":[1]}\n'
    )
    vectors = tmp_path / "orion_numeric_import.f32le"
    vectors.write_bytes(np.asarray([[1, 0], [0, 1], [1, 1]], dtype="<f4").tobytes())
    artifact = tmp_path / "generation-7.json"
    artifact.write_text(
        json.dumps(
            {
                "format_version": 2,
                "generation": 7,
                "vector_schema": {
                    "vector_name": "",
                    "dimension": 2,
                    "distance": "Cosine",
                    "datatype": "float32",
                },
                "shard_count": 2,
                "layout_sha256": sha(assignments),
                "logical_point_count": 3,
                "physical_point_count": 4,
                "upper_k": 1,
                "upper_ef_search": 2,
                "dynamic_ef_base": 8,
                "dynamic_ef_factor": 4,
                "upper_nodes": [
                    {"label": 0, "vector": [1, 0], "owner_shard": 0},
                    {"label": 1, "vector": [0, 1], "owner_shard": 1},
                ],
                "upper_graph": {
                    "entry_point": 0,
                    "max_level": 0,
                    "nodes": [],
                },
            }
        )
    )
    manifest = tmp_path / "orion_numeric_import.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "dimension": 2,
                "point_count": 3,
                "shard_count": 2,
                "vector_name": "",
                "orion_generation": 7,
                "orion_artifact_file": artifact.name,
                "orion_artifact_sha256": sha(artifact),
                "vectors_file": vectors.name,
                "vectors_sha256": sha(vectors),
                "assignments_file": assignments.name,
                "assignments_sha256": sha(assignments),
                "total_point_copies": 4,
            }
        )
    )
    return tmp_path


def test_frozen_bundle_validates_checksums_counts_and_numeric_upper_labels(tmp_path):
    module = load_module()
    bundle = module.load_frozen_bundle(write_bundle(tmp_path))
    assert bundle.point_to_shards == [[0], [1, 0], [1]]
    assert bundle.upper_indices.tolist() == [0, 1]
    assert bundle.vectors.shape == (3, 2)
    assert module.qdrant_distance(bundle.artifact) == "Cosine"


def test_frozen_bundle_fails_closed_on_checksum_mismatch(tmp_path):
    module = load_module()
    root = write_bundle(tmp_path)
    (root / "orion_numeric_import.f32le").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.load_frozen_bundle(root)


def test_frozen_bundle_accepts_only_checksum_matching_vector_override(tmp_path):
    module = load_module()
    root = write_bundle(tmp_path)
    original = root / "orion_numeric_import.f32le"
    override = tmp_path / "canonical-vectors.f32le"
    original.replace(override)
    bundle = module.load_frozen_bundle(root, override)
    assert bundle.vectors_path == override.resolve()
    override.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.load_frozen_bundle(root, override)


def test_build_identity_binds_layout_vectors_and_hnsw(tmp_path):
    module = load_module()
    bundle = module.load_frozen_bundle(write_bundle(tmp_path))
    args = Namespace(hnsw_m=32, ef_construct=200)
    identity = module.build_identity(bundle, args)
    assert identity["layout_sha256"] == bundle.artifact["layout_sha256"]
    assert identity["vectors_sha256"] == bundle.manifest["vectors_sha256"]
    assert identity["hnsw_m"] == 32
    assert identity["hnsw_ef_construct"] == 200


def test_prepare_argument_validation_requires_explicit_round_robin_peers():
    module = load_module()
    args = Namespace(
        hnsw_m=32,
        ef_construct=100,
        upload_batch_size=512,
        shard_placement="round_robin",
        peer_ids=[],
    )
    with pytest.raises(ValueError, match="peer-ids"):
        module.validate_args(args)
