from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "experiments/l1_balance/formal_ab_placement.py"
BASE_URL = "http://10.10.1.1:6333"
COLLECTION = "formal_orion_arm"
GENERATION = 701
ARTIFACT_LAYOUT_SHA256 = "2" * 64
PEERS = [10, 20, 30, 40]


def load_module():
    name = "l1_balance_formal_ab_placement_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def placement_for(peer_order):
    return {shard_id: peer_order[shard_id % len(peer_order)] for shard_id in range(32)}


def make_live_info(
    module,
    *,
    generation=GENERATION,
    layout_sha=ARTIFACT_LAYOUT_SHA256,
    artifact_sha="1" * 64,
):
    provenance = {
        "method": "orion",
        "shard_count": 32,
        "logical_point_count": 100,
        "physical_point_count": 120,
        "routing": {
            "generation": generation,
            "artifact_sha256": artifact_sha,
            "layout_sha256": layout_sha,
            "attachment_search_ef": 100,
        },
    }
    return {
        "status": "green",
        "optimizer_status": "ok",
        "points_count": 120,
        "config": {
            "params": {
                "sharding_method": "auto",
                "shard_number": 32,
                "replication_factor": 1,
                "write_consistency_factor": 1,
            },
            "auto_shard_policy": {
                "type": "orion",
                "generation": generation,
                "artifact_sha256": artifact_sha,
            },
            "metadata": {
                "native_auto_shard_prepare": {
                    "schema_version": 2,
                    "provenance": provenance,
                    "provenance_sha256": module.canonical_json_sha256(provenance),
                }
            },
        },
    }


def make_cluster(placement, controller_peer=10):
    local = []
    remote = []
    for shard_id, peer_id in sorted(placement.items()):
        row = {"shard_id": shard_id, "state": "Active"}
        if peer_id == controller_peer:
            local.append(row)
        else:
            remote.append({**row, "peer_id": peer_id})
    return {
        "peer_id": controller_peer,
        "shard_count": 32,
        "local_shards": local,
        "remote_shards": remote,
        "shard_transfers": [],
    }


def write_prepare(module, tmp_path, target_peer_order=None):
    target_peer_order = target_peer_order or [20, 30, 40, 10]
    layout_dir = tmp_path / "layout"
    layout_dir.mkdir()
    artifact_path = layout_dir / f"generation-{GENERATION}.json"
    artifact = {
        "generation": GENERATION,
        "shard_count": 32,
        "layout_sha256": ARTIFACT_LAYOUT_SHA256,
    }
    write_json(artifact_path, artifact)
    artifact_sha = module.sha256_path(artifact_path)

    build_manifest_path = layout_dir / "build-manifest.json"
    build_manifest = {
        "tool": "synthetic-test",
        "mode": "production_bundle",
        "outputs": {"production_artifact": artifact_path.name},
    }
    write_json(build_manifest_path, build_manifest)
    build_manifest_sha = module.sha256_path(build_manifest_path)

    provenance = {
        "method": "orion",
        "shard_count": 32,
        "logical_point_count": 100,
        "physical_point_count": 120,
        "routing": {
            "generation": GENERATION,
            "artifact_sha256": artifact_sha,
            "layout_sha256": ARTIFACT_LAYOUT_SHA256,
            "attachment_search_ef": 100,
        },
    }
    target = placement_for(target_peer_order)
    prepare = {
        "method": "orion",
        "base_url": BASE_URL,
        "collection": COLLECTION,
        "shard_count": 32,
        "replication_factor": 1,
        "checksums": {
            "routing_artifact_sha256": artifact_sha,
            "layout_build_manifest_sha256": build_manifest_sha,
        },
        "layout": {
            "layout_dir": str(layout_dir),
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
            "build_manifest_path": str(build_manifest_path),
            "build_manifest_sha256": build_manifest_sha,
            "generation": GENERATION,
            "shard_count": 32,
            "artifact": {
                "generation": GENERATION,
                "shard_count": 32,
                "layout_sha256": ARTIFACT_LAYOUT_SHA256,
            },
        },
        "provenance_metadata": {
            "native_auto_shard_prepare": {
                "schema_version": 2,
                "provenance": provenance,
                "provenance_sha256": module.canonical_json_sha256(provenance),
            }
        },
        "final_collection_proof": {
            "policy": {
                "type": "orion",
                "generation": GENERATION,
                "artifact_sha256": artifact_sha,
            }
        },
        "placement_plan": {
            "strategy": "round_robin",
            "target_placement": {str(key): value for key, value in target.items()},
        },
        "placement_peers": {
            "mode": "all_peers",
            "peer_ids": target_peer_order,
            "includes_controller": True,
        },
        "final_placement_proof": {
            "expected_placement": {str(key): value for key, value in target.items()},
            "replication_factor": 1,
            "shard_count": 32,
            "valid": True,
        },
    }
    prepare_path = tmp_path / "preparation_manifest.json"
    write_json(prepare_path, prepare)
    return prepare_path, target, artifact_sha


@pytest.fixture
def guarded_collection(monkeypatch):
    module = load_module()
    original = placement_for(PEERS)
    state = {
        "placement": dict(original),
        "info": make_live_info(module),
        "collections_read": [],
    }

    def collection_info(_base_url, collection):
        state["collections_read"].append(collection)
        assert collection == COLLECTION
        return copy.deepcopy(state["info"])

    def cluster_info(_base_url, collection):
        state["collections_read"].append(collection)
        assert collection == COLLECTION
        return make_cluster(state["placement"])

    monkeypatch.setattr(module.experiment, "collection_info", collection_info)
    monkeypatch.setattr(module.experiment, "collection_cluster_info", cluster_info)
    return module, state, original


def freeze_snapshot(module, tmp_path):
    path = tmp_path / "placement-snapshot.json"
    snapshot = module.snapshot_collection(
        base_url=BASE_URL,
        collection=COLLECTION,
        output=path,
    )
    return path, snapshot


def test_snapshot_is_self_checksummed_and_reads_only_explicit_collection(
    guarded_collection, tmp_path
):
    module, state, original = guarded_collection
    path, snapshot = freeze_snapshot(module, tmp_path)

    assert snapshot["placement"] == {str(key): value for key, value in original.items()}
    assert snapshot["identity"]["generation"] == GENERATION
    unsigned = dict(snapshot)
    unsigned.pop("snapshot_sha256")
    assert snapshot["snapshot_sha256"] == module.canonical_json_sha256(unsigned)
    assert path.is_file()
    assert set(state["collections_read"]) == {COLLECTION}


def test_apply_dry_run_validates_every_binding_and_never_moves(
    guarded_collection, tmp_path, monkeypatch
):
    module, state, original = guarded_collection
    prepare_path, target, artifact_sha = write_prepare(module, tmp_path)
    state["info"] = make_live_info(module, artifact_sha=artifact_sha)
    snapshot_path, snapshot = freeze_snapshot(module, tmp_path)

    def forbidden_move(*_args, **_kwargs):
        raise AssertionError("dry-run must not move shards")

    monkeypatch.setattr(module.experiment, "move_numeric_shards_explicit", forbidden_move)
    proof = module.apply_prepare_placement(
        base_url=BASE_URL,
        collection=COLLECTION,
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot["snapshot_sha256"],
        prepare_manifest=prepare_path,
        proof_output=tmp_path / "apply-dry-run.json",
        dry_run=True,
        transfer_method="snapshot",
        timeout_sec=10.0,
        poll_interval_sec=0.0,
    )

    assert proof["status"] == "DRY_RUN"
    assert proof["move_proof"] is None
    assert proof["exact_target_verified"] is False
    assert len(proof["planned_moves"]) == 32
    assert state["placement"] == original
    assert proof["target_placement"] == {
        str(key): value for key, value in target.items()
    }


def test_apply_calls_only_explicit_move_then_verifies_exact_map(
    guarded_collection, tmp_path, monkeypatch
):
    module, state, _original = guarded_collection
    prepare_path, target, artifact_sha = write_prepare(module, tmp_path)
    state["info"] = make_live_info(module, artifact_sha=artifact_sha)
    snapshot_path, snapshot = freeze_snapshot(module, tmp_path)
    calls = []

    def move(base_url, collection, peer_ids, placement, **kwargs):
        calls.append((base_url, collection, peer_ids, dict(placement), kwargs))
        state["placement"] = dict(placement)
        return {"valid": True, "moves": [{"shard_id": 0}]}

    monkeypatch.setattr(module.experiment, "move_numeric_shards_explicit", move)
    proof = module.apply_prepare_placement(
        base_url=BASE_URL,
        collection=COLLECTION,
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot["snapshot_sha256"],
        prepare_manifest=prepare_path,
        proof_output=tmp_path / "apply-proof.json",
        dry_run=False,
        transfer_method="snapshot",
        timeout_sec=10.0,
        poll_interval_sec=0.0,
    )

    assert len(calls) == 1
    assert calls[0][0:2] == (BASE_URL, COLLECTION)
    assert calls[0][2] == [20, 30, 40, 10]
    assert calls[0][3] == target
    assert calls[0][4]["expected_shard_count"] == 32
    assert calls[0][4]["include_controller"] is True
    assert proof["status"] == "PASS"
    assert proof["exact_target_verified"] is True
    assert state["placement"] == target


def test_restore_requires_original_sha_and_verifies_exact_original_map(
    guarded_collection, tmp_path, monkeypatch
):
    module, state, original = guarded_collection
    snapshot_path, snapshot = freeze_snapshot(module, tmp_path)
    state["placement"] = placement_for([20, 30, 40, 10])
    calls = []

    def move(_base_url, _collection, peer_ids, placement, **kwargs):
        calls.append((peer_ids, dict(placement), kwargs))
        state["placement"] = dict(placement)
        return {"valid": True}

    monkeypatch.setattr(module.experiment, "move_numeric_shards_explicit", move)
    proof = module.restore_snapshot_placement(
        base_url=BASE_URL,
        collection=COLLECTION,
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot["snapshot_sha256"],
        proof_output=tmp_path / "restore-proof.json",
        dry_run=False,
        transfer_method="snapshot",
        timeout_sec=10.0,
        poll_interval_sec=0.0,
    )

    assert calls[0][0] == PEERS
    assert calls[0][1] == original
    assert calls[0][2]["include_controller"] is True
    assert state["placement"] == original
    assert proof["exact_snapshot_verified"] is True

    with pytest.raises(RuntimeError, match="authorized original SHA"):
        module.restore_snapshot_placement(
            base_url=BASE_URL,
            collection=COLLECTION,
            snapshot_path=snapshot_path,
            snapshot_sha256="f" * 64,
            proof_output=tmp_path / "must-not-exist.json",
            dry_run=True,
            transfer_method="snapshot",
            timeout_sec=10.0,
            poll_interval_sec=0.0,
        )


def test_restore_dry_run_never_moves(guarded_collection, tmp_path, monkeypatch):
    module, state, original = guarded_collection
    snapshot_path, snapshot = freeze_snapshot(module, tmp_path)
    changed = placement_for([20, 30, 40, 10])
    state["placement"] = dict(changed)

    def forbidden_move(*_args, **_kwargs):
        raise AssertionError("restore dry-run must not move shards")

    monkeypatch.setattr(module.experiment, "move_numeric_shards_explicit", forbidden_move)
    proof = module.restore_snapshot_placement(
        base_url=BASE_URL,
        collection=COLLECTION,
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot["snapshot_sha256"],
        proof_output=tmp_path / "restore-dry-run.json",
        dry_run=True,
        transfer_method="snapshot",
        timeout_sec=10.0,
        poll_interval_sec=0.0,
    )

    assert proof["status"] == "DRY_RUN"
    assert proof["exact_snapshot_verified"] is False
    assert proof["move_proof"] is None
    assert proof["target_placement"] == {
        str(key): value for key, value in original.items()
    }
    assert state["placement"] == changed


def test_tampered_snapshot_fails_closed(guarded_collection, tmp_path):
    module, _state, _original = guarded_collection
    snapshot_path, snapshot = freeze_snapshot(module, tmp_path)
    tampered = copy.deepcopy(snapshot)
    tampered["placement"]["0"] = 999
    tampered_path = tmp_path / "tampered.json"
    write_json(tampered_path, tampered)

    with pytest.raises(RuntimeError, match="content checksum mismatch"):
        module.validate_snapshot_record(
            module.load_json_object(tampered_path, "snapshot"),
            snapshot["snapshot_sha256"],
            base_url=BASE_URL,
            collection=COLLECTION,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda prepare: prepare["placement_plan"].__setitem__(
                "strategy", "layout_size_balanced"
            ),
            "must be round_robin",
        ),
        (
            lambda prepare: prepare["layout"].__setitem__("generation", GENERATION + 1),
            "generation mismatch",
        ),
        (
            lambda prepare: prepare["layout"]["artifact"].__setitem__(
                "layout_sha256", "3" * 64
            ),
            "layout checksum mismatch",
        ),
        (
            lambda prepare: prepare.__setitem__("replication_factor", 2),
            "RF=1",
        ),
    ],
)
def test_prepare_validation_rejects_contract_drift(
    guarded_collection, tmp_path, mutation, message
):
    module, _state, _original = guarded_collection
    prepare_path, _target, _artifact_sha = write_prepare(module, tmp_path)
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    mutation(prepare)
    write_json(prepare_path, prepare)

    with pytest.raises(RuntimeError, match=message):
        module.validate_prepare_manifest(
            prepare_path,
            base_url=BASE_URL,
            collection=COLLECTION,
        )


def test_apply_rejects_live_identity_or_original_placement_drift(
    guarded_collection, tmp_path
):
    module, state, _original = guarded_collection
    prepare_path, _target, artifact_sha = write_prepare(module, tmp_path)
    state["info"] = make_live_info(module, artifact_sha=artifact_sha)
    snapshot_path, snapshot = freeze_snapshot(module, tmp_path)
    state["placement"][0] = 20

    with pytest.raises(RuntimeError, match="exact original snapshot placement"):
        module.apply_prepare_placement(
            base_url=BASE_URL,
            collection=COLLECTION,
            snapshot_path=snapshot_path,
            snapshot_sha256=snapshot["snapshot_sha256"],
            prepare_manifest=prepare_path,
            proof_output=tmp_path / "unused.json",
            dry_run=True,
            transfer_method="snapshot",
            timeout_sec=10.0,
            poll_interval_sec=0.0,
        )
