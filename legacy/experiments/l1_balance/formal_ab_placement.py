#!/usr/bin/env python3
"""Checksum-bound numeric-shard placement guard for the formal Orion A/B.

The guard has exactly three operations:

* ``snapshot`` freezes one explicitly named collection's live P=32/RF=1 map;
* ``apply`` installs the round-robin target recorded by one prepare manifest;
* ``restore`` reinstalls the exact map in the original snapshot.

It never creates, deletes, or updates a collection, installs an artifact, or
changes container resources.  The only mutating function reachable here is
``move_numeric_shards_explicit``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import qdrant_two_level_routing_experiment as experiment


SCHEMA_VERSION = 1
SNAPSHOT_KIND = "orion-formal-ab-placement-snapshot"
PROOF_KIND = "orion-formal-ab-placement-proof"
FORMAL_SHARD_COUNT = 32
PROVENANCE_KEY = "native_auto_shard_prepare"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a valid SHA-256 string")
    return normalized


def normalize_base_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base URL must be a non-empty string")
    return value.strip().rstrip("/")


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def parse_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def parse_placement(value: Any, label: str) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    placement: dict[int, int] = {}
    for raw_shard, raw_peer in value.items():
        try:
            shard_id = int(raw_shard)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} contains invalid shard ID {raw_shard!r}") from error
        if str(shard_id) != str(raw_shard) and raw_shard != shard_id:
            raise ValueError(f"{label} contains non-canonical shard ID {raw_shard!r}")
        if (
            shard_id < 0
            or isinstance(raw_peer, bool)
            or not isinstance(raw_peer, int)
            or raw_peer < 0
        ):
            raise ValueError(f"{label} contains invalid mapping {raw_shard!r}: {raw_peer!r}")
        if shard_id in placement:
            raise ValueError(f"{label} repeats shard ID {shard_id}")
        placement[shard_id] = int(raw_peer)
    expected = set(range(FORMAL_SHARD_COUNT))
    if set(placement) != expected:
        raise ValueError(
            f"{label} must cover exactly numeric shards 0..{FORMAL_SHARD_COUNT - 1}"
        )
    return {shard_id: placement[shard_id] for shard_id in sorted(placement)}


def serialized_placement(placement: Mapping[int, int]) -> dict[str, int]:
    return {str(shard_id): int(placement[shard_id]) for shard_id in sorted(placement)}


def _live_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    metadata = config.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("live collection lacks checksum-bound prepare provenance")
    wrapper = metadata.get(PROVENANCE_KEY)
    if not isinstance(wrapper, Mapping):
        raise RuntimeError("live collection lacks native_auto_shard_prepare provenance")
    provenance = wrapper.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("live collection prepare provenance is malformed")
    declared = normalize_sha256(
        wrapper.get("provenance_sha256"),
        "live provenance_sha256",
    )
    actual = canonical_json_sha256(provenance)
    if declared != actual:
        raise RuntimeError("live collection prepare provenance checksum mismatch")
    return dict(provenance)


def collection_identity(info: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the immutable routing/layout identity from a live collection."""
    config = info.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("live collection config is unavailable")
    params = config.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("live collection params are unavailable")
    if str(params.get("sharding_method") or "auto").lower() != "auto":
        raise RuntimeError("formal placement guard requires numeric auto-sharding")
    if params.get("shard_number") != FORMAL_SHARD_COUNT:
        raise RuntimeError(
            f"formal placement guard requires P={FORMAL_SHARD_COUNT}, "
            f"found {params.get('shard_number')!r}"
        )
    if params.get("replication_factor") != 1:
        raise RuntimeError(
            "formal placement guard requires RF=1, "
            f"found {params.get('replication_factor')!r}"
        )

    policy = config.get("auto_shard_policy")
    if not isinstance(policy, Mapping) or str(policy.get("type") or "").lower() != "orion":
        raise RuntimeError("formal placement guard requires an Orion auto_shard_policy")
    generation = parse_positive_int(policy.get("generation"), "live Orion generation")
    artifact_sha256 = normalize_sha256(
        policy.get("artifact_sha256"),
        "live Orion artifact_sha256",
    )

    provenance = _live_provenance(config)
    if provenance.get("method") != "orion":
        raise RuntimeError("live prepare provenance method is not Orion")
    if provenance.get("shard_count") != FORMAL_SHARD_COUNT:
        raise RuntimeError("live prepare provenance does not declare P=32")
    routing = provenance.get("routing")
    if not isinstance(routing, Mapping):
        raise RuntimeError("live prepare provenance lacks routing identity")
    if routing.get("generation") != generation:
        raise RuntimeError("live policy/provenance generation mismatch")
    if normalize_sha256(
        routing.get("artifact_sha256"),
        "live provenance artifact_sha256",
    ) != artifact_sha256:
        raise RuntimeError("live policy/provenance artifact checksum mismatch")
    layout_sha256 = normalize_sha256(
        routing.get("layout_sha256"),
        "live provenance layout_sha256",
    )
    return {
        "generation": generation,
        "artifact_sha256": artifact_sha256,
        "layout_sha256": layout_sha256,
        "shard_count": FORMAL_SHARD_COUNT,
        "replication_factor": 1,
    }


def inspect_collection(base_url: str, collection: str) -> dict[str, Any]:
    """Read exactly one named collection; unrelated collections are never enumerated."""
    info = experiment.collection_info(base_url, collection)
    cluster = experiment.collection_cluster_info(base_url, collection)
    if cluster is None:
        raise RuntimeError(f"collection {collection!r} has no cluster placement response")
    placement = experiment.numeric_shard_placement_from_cluster(
        cluster,
        expected_shard_count=FORMAL_SHARD_COUNT,
    )
    identity = collection_identity(info)
    return {
        "identity": identity,
        "placement": placement,
        "config_sha256": canonical_json_sha256(info.get("config")),
        "points_count": info.get("points_count"),
        "controller_peer_id": cluster.get("peer_id"),
    }


def create_snapshot_record(
    base_url: str,
    collection: str,
    inspection: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "created_at": utc_now(),
        "base_url": normalize_base_url(base_url),
        "collection": collection,
        "identity": dict(inspection["identity"]),
        "config_sha256": inspection["config_sha256"],
        "points_count": inspection.get("points_count"),
        "controller_peer_id": inspection.get("controller_peer_id"),
        "placement": serialized_placement(inspection["placement"]),
    }
    return {**unsigned, "snapshot_sha256": canonical_json_sha256(unsigned)}


def validate_snapshot_record(
    record: Mapping[str, Any],
    expected_sha256: str,
    *,
    base_url: str,
    collection: str,
) -> dict[str, Any]:
    if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != SNAPSHOT_KIND:
        raise RuntimeError("unsupported formal placement snapshot format")
    declared = normalize_sha256(record.get("snapshot_sha256"), "snapshot_sha256")
    expected = normalize_sha256(expected_sha256, "expected snapshot SHA-256")
    unsigned = dict(record)
    unsigned.pop("snapshot_sha256", None)
    actual = canonical_json_sha256(unsigned)
    if declared != actual:
        raise RuntimeError("placement snapshot content checksum mismatch")
    if expected != declared:
        raise RuntimeError("placement snapshot is not the explicitly authorized original SHA")
    if normalize_base_url(record.get("base_url")) != normalize_base_url(base_url):
        raise RuntimeError("placement snapshot base URL mismatch")
    if record.get("collection") != collection:
        raise RuntimeError("placement snapshot collection mismatch")
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("placement snapshot identity is malformed")
    normalized_identity = {
        "generation": parse_positive_int(identity.get("generation"), "snapshot generation"),
        "artifact_sha256": normalize_sha256(
            identity.get("artifact_sha256"), "snapshot artifact_sha256"
        ),
        "layout_sha256": normalize_sha256(
            identity.get("layout_sha256"), "snapshot layout_sha256"
        ),
        "shard_count": identity.get("shard_count"),
        "replication_factor": identity.get("replication_factor"),
    }
    if normalized_identity["shard_count"] != FORMAL_SHARD_COUNT:
        raise RuntimeError("placement snapshot is not P=32")
    if normalized_identity["replication_factor"] != 1:
        raise RuntimeError("placement snapshot is not RF=1")
    normalize_sha256(record.get("config_sha256"), "snapshot config_sha256")
    return {
        **dict(record),
        "identity": normalized_identity,
        "placement_int": parse_placement(record.get("placement"), "snapshot placement"),
    }


def _require_equal(values: list[tuple[str, Any]], expected: Any, label: str) -> None:
    mismatches = {name: value for name, value in values if value != expected}
    if mismatches:
        raise RuntimeError(f"prepare manifest {label} mismatch: {mismatches}")


def validate_prepare_manifest(
    path: Path,
    *,
    base_url: str,
    collection: str,
) -> dict[str, Any]:
    """Validate the prepare, layout bundle, and exact round-robin target."""
    path = path.expanduser().resolve(strict=True)
    prepare = load_json_object(path, "prepare manifest")
    if prepare.get("method") != "orion":
        raise RuntimeError("formal placement apply requires an Orion prepare manifest")
    if prepare.get("collection") != collection:
        raise RuntimeError("prepare manifest collection mismatch")
    if normalize_base_url(prepare.get("base_url")) != normalize_base_url(base_url):
        raise RuntimeError("prepare manifest base URL mismatch")
    if prepare.get("shard_count") != FORMAL_SHARD_COUNT:
        raise RuntimeError("prepare manifest must declare P=32")
    if prepare.get("replication_factor") != 1:
        raise RuntimeError("prepare manifest must declare RF=1")

    layout = prepare.get("layout")
    if not isinstance(layout, Mapping):
        raise RuntimeError("prepare manifest lacks a routed layout binding")
    layout_dir = Path(str(layout.get("layout_dir") or "")).expanduser().resolve(strict=True)
    artifact_path = Path(str(layout.get("artifact_path") or "")).expanduser().resolve(strict=True)
    build_manifest_path = Path(
        str(layout.get("build_manifest_path") or "")
    ).expanduser().resolve(strict=True)
    if not layout_dir.is_dir() or artifact_path.parent != layout_dir:
        raise RuntimeError("prepare artifact is not inside its declared layout directory")
    if build_manifest_path.parent != layout_dir:
        raise RuntimeError("prepare build manifest is not inside its declared layout directory")

    artifact_sha256 = normalize_sha256(
        layout.get("artifact_sha256"), "prepare layout artifact_sha256"
    )
    build_manifest_sha256 = normalize_sha256(
        layout.get("build_manifest_sha256"), "prepare build_manifest_sha256"
    )
    checksums = prepare.get("checksums")
    if not isinstance(checksums, Mapping):
        raise RuntimeError("prepare manifest lacks checksums")
    _require_equal(
        [
            ("layout.artifact_sha256", artifact_sha256),
            (
                "checksums.routing_artifact_sha256",
                normalize_sha256(
                    checksums.get("routing_artifact_sha256"),
                    "prepare routing_artifact_sha256",
                ),
            ),
            ("artifact file", sha256_path(artifact_path)),
        ],
        artifact_sha256,
        "artifact checksum",
    )
    _require_equal(
        [
            ("layout.build_manifest_sha256", build_manifest_sha256),
            (
                "checksums.layout_build_manifest_sha256",
                normalize_sha256(
                    checksums.get("layout_build_manifest_sha256"),
                    "prepare layout_build_manifest_sha256",
                ),
            ),
            ("build manifest file", sha256_path(build_manifest_path)),
        ],
        build_manifest_sha256,
        "build manifest checksum",
    )

    artifact = load_json_object(artifact_path, "Orion artifact")
    build_manifest = load_json_object(build_manifest_path, "layout build manifest")
    generation = parse_positive_int(layout.get("generation"), "prepare generation")
    layout_sha256 = normalize_sha256(
        artifact.get("layout_sha256"), "artifact layout_sha256"
    )
    artifact_record = layout.get("artifact")
    if not isinstance(artifact_record, Mapping):
        raise RuntimeError("prepare layout lacks artifact validation record")
    _require_equal(
        [
            ("layout.generation", generation),
            ("layout.artifact.generation", artifact_record.get("generation")),
            ("artifact.generation", artifact.get("generation")),
        ],
        generation,
        "generation",
    )
    _require_equal(
        [
            ("prepare.shard_count", prepare.get("shard_count")),
            ("layout.shard_count", layout.get("shard_count")),
            ("layout.artifact.shard_count", artifact_record.get("shard_count")),
            ("artifact.shard_count", artifact.get("shard_count")),
        ],
        FORMAL_SHARD_COUNT,
        "P=32 contract",
    )
    _require_equal(
        [
            ("layout.artifact.layout_sha256", artifact_record.get("layout_sha256")),
            ("artifact.layout_sha256", layout_sha256),
        ],
        layout_sha256,
        "layout checksum",
    )
    outputs = build_manifest.get("outputs")
    if not isinstance(outputs, Mapping) or outputs.get("production_artifact") != artifact_path.name:
        raise RuntimeError("layout build manifest selects a different production artifact")

    provenance_wrapper = (prepare.get("provenance_metadata") or {}).get(PROVENANCE_KEY)
    if not isinstance(provenance_wrapper, Mapping):
        raise RuntimeError("prepare manifest lacks checksum-bound provenance")
    provenance = provenance_wrapper.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("prepare provenance is malformed")
    provenance_sha256 = normalize_sha256(
        provenance_wrapper.get("provenance_sha256"),
        "prepare provenance_sha256",
    )
    if canonical_json_sha256(provenance) != provenance_sha256:
        raise RuntimeError("prepare provenance checksum mismatch")
    routing = provenance.get("routing")
    if provenance.get("method") != "orion" or not isinstance(routing, Mapping):
        raise RuntimeError("prepare provenance is not Orion routing provenance")
    if provenance.get("shard_count") != FORMAL_SHARD_COUNT:
        raise RuntimeError("prepare provenance does not declare P=32")
    if routing.get("generation") != generation:
        raise RuntimeError("prepare provenance generation mismatch")
    if normalize_sha256(
        routing.get("artifact_sha256"), "prepare provenance artifact_sha256"
    ) != artifact_sha256:
        raise RuntimeError("prepare provenance artifact checksum mismatch")
    if normalize_sha256(
        routing.get("layout_sha256"), "prepare provenance layout_sha256"
    ) != layout_sha256:
        raise RuntimeError("prepare provenance layout checksum mismatch")

    final_collection = prepare.get("final_collection_proof")
    if not isinstance(final_collection, Mapping):
        raise RuntimeError("prepare manifest lacks final collection proof")
    final_policy = final_collection.get("policy")
    if not isinstance(final_policy, Mapping):
        raise RuntimeError("prepare final collection proof lacks Orion policy")
    if (
        final_policy.get("type") != "orion"
        or final_policy.get("generation") != generation
        or normalize_sha256(
            final_policy.get("artifact_sha256"),
            "prepare final policy artifact_sha256",
        )
        != artifact_sha256
    ):
        raise RuntimeError("prepare final collection policy identity mismatch")

    placement_plan = prepare.get("placement_plan")
    if not isinstance(placement_plan, Mapping):
        raise RuntimeError("prepare manifest lacks placement_plan")
    if placement_plan.get("strategy") != "round_robin":
        raise RuntimeError("formal A/B placement target must be round_robin")
    target = parse_placement(
        placement_plan.get("target_placement"),
        "prepare round-robin target",
    )
    placement_peers = prepare.get("placement_peers")
    if not isinstance(placement_peers, Mapping):
        raise RuntimeError("prepare manifest lacks placement_peers")
    raw_peer_ids = placement_peers.get("peer_ids")
    if not isinstance(raw_peer_ids, list) or not raw_peer_ids:
        raise RuntimeError("prepare placement_peers.peer_ids must be a non-empty list")
    if any(
        isinstance(peer_id, bool) or not isinstance(peer_id, int) or peer_id < 0
        for peer_id in raw_peer_ids
    ):
        raise RuntimeError("prepare placement peer IDs must be non-negative integers")
    peer_ids = [int(peer_id) for peer_id in raw_peer_ids]
    expected_target = experiment.round_robin_numeric_shard_targets(
        list(range(FORMAL_SHARD_COUNT)),
        peer_ids,
    )
    if target != expected_target:
        raise RuntimeError("prepare placement target is not its declared round-robin map")
    include_controller = placement_peers.get("includes_controller")
    if not isinstance(include_controller, bool):
        raise RuntimeError("prepare placement_peers.includes_controller must be boolean")
    final_placement = prepare.get("final_placement_proof")
    if not isinstance(final_placement, Mapping):
        raise RuntimeError("prepare manifest lacks final placement proof")
    if parse_placement(
        final_placement.get("expected_placement"),
        "prepare final expected placement",
    ) != target:
        raise RuntimeError("prepare placement plan/final proof target mismatch")
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "identity": {
            "generation": generation,
            "artifact_sha256": artifact_sha256,
            "layout_sha256": layout_sha256,
            "shard_count": FORMAL_SHARD_COUNT,
            "replication_factor": 1,
        },
        "target_placement": target,
        "peer_ids": peer_ids,
        "include_controller": include_controller,
        "artifact_path": str(artifact_path),
        "build_manifest_path": str(build_manifest_path),
    }


def _assert_snapshot_live_contract(
    snapshot: Mapping[str, Any],
    live: Mapping[str, Any],
    *,
    require_original_placement: bool,
) -> None:
    if live["identity"] != snapshot["identity"]:
        raise RuntimeError("live generation/artifact/layout identity drifted from snapshot")
    if live["config_sha256"] != snapshot.get("config_sha256"):
        raise RuntimeError("live collection config drifted from snapshot")
    if live.get("points_count") != snapshot.get("points_count"):
        raise RuntimeError("live collection point count drifted from snapshot")
    if require_original_placement and live["placement"] != snapshot["placement_int"]:
        raise RuntimeError("apply requires the exact original snapshot placement")


def planned_moves(
    source: Mapping[int, int], target: Mapping[int, int]
) -> list[dict[str, int]]:
    return [
        {
            "shard_id": shard_id,
            "from_peer_id": int(source[shard_id]),
            "to_peer_id": int(target[shard_id]),
        }
        for shard_id in sorted(target)
        if source[shard_id] != target[shard_id]
    ]


def _write_proof(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "kind": PROOF_KIND,
        "created_at": utc_now(),
        **payload,
    }
    proof = {**unsigned, "proof_sha256": canonical_json_sha256(unsigned)}
    write_json_new(path, proof)
    return proof


def snapshot_collection(
    *,
    base_url: str,
    collection: str,
    output: Path,
) -> dict[str, Any]:
    live = inspect_collection(base_url, collection)
    snapshot = create_snapshot_record(base_url, collection, live)
    write_json_new(output, snapshot)
    return snapshot


def apply_prepare_placement(
    *,
    base_url: str,
    collection: str,
    snapshot_path: Path,
    snapshot_sha256: str,
    prepare_manifest: Path,
    proof_output: Path,
    dry_run: bool,
    transfer_method: str,
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    snapshot_record = load_json_object(snapshot_path, "placement snapshot")
    snapshot = validate_snapshot_record(
        snapshot_record,
        snapshot_sha256,
        base_url=base_url,
        collection=collection,
    )
    prepare = validate_prepare_manifest(
        prepare_manifest,
        base_url=base_url,
        collection=collection,
    )
    if snapshot["identity"] != prepare["identity"]:
        raise RuntimeError("prepare generation/artifact/layout differs from original snapshot")
    before = inspect_collection(base_url, collection)
    _assert_snapshot_live_contract(snapshot, before, require_original_placement=True)
    controller_peer_id = before.get("controller_peer_id")
    if not isinstance(controller_peer_id, int):
        raise RuntimeError("live collection cluster lacks a numeric controller peer ID")
    if prepare["include_controller"] != (controller_peer_id in prepare["peer_ids"]):
        raise RuntimeError(
            "prepare includes_controller disagrees with the live controller peer ID"
        )
    target = prepare["target_placement"]
    moves = planned_moves(before["placement"], target)
    payload: dict[str, Any] = {
        "operation": "apply",
        "status": "DRY_RUN" if dry_run else "PASS",
        "dry_run": dry_run,
        "base_url": normalize_base_url(base_url),
        "collection": collection,
        "snapshot_path": str(snapshot_path.expanduser().resolve()),
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "prepare_manifest_path": prepare["path"],
        "prepare_manifest_sha256": prepare["sha256"],
        "identity": prepare["identity"],
        "before_placement": serialized_placement(before["placement"]),
        "target_placement": serialized_placement(target),
        "planned_moves": moves,
        "mutation_scope": "numeric_shard_move_only",
        "forbidden_mutations": [
            "collection_create_delete_or_update",
            "artifact_install_or_activation",
            "container_resource_change",
        ],
    }
    if dry_run:
        payload["move_proof"] = None
        payload["after_placement"] = None
        payload["exact_target_verified"] = before["placement"] == target
        return _write_proof(proof_output, payload)

    move_proof = experiment.move_numeric_shards_explicit(
        base_url,
        collection,
        prepare["peer_ids"],
        target,
        expected_shard_count=FORMAL_SHARD_COUNT,
        transfer_method=transfer_method,
        include_controller=prepare["include_controller"],
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
    )
    after = inspect_collection(base_url, collection)
    _assert_snapshot_live_contract(snapshot, after, require_original_placement=False)
    if after["placement"] != target:
        raise RuntimeError("apply completed without the exact prepare target placement")
    payload["move_proof"] = move_proof
    payload["after_placement"] = serialized_placement(after["placement"])
    payload["exact_target_verified"] = True
    return _write_proof(proof_output, payload)


def restore_snapshot_placement(
    *,
    base_url: str,
    collection: str,
    snapshot_path: Path,
    snapshot_sha256: str,
    proof_output: Path,
    dry_run: bool,
    transfer_method: str,
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    snapshot_record = load_json_object(snapshot_path, "placement snapshot")
    snapshot = validate_snapshot_record(
        snapshot_record,
        snapshot_sha256,
        base_url=base_url,
        collection=collection,
    )
    before = inspect_collection(base_url, collection)
    _assert_snapshot_live_contract(snapshot, before, require_original_placement=False)
    target = snapshot["placement_int"]
    moves = planned_moves(before["placement"], target)
    peer_ids = sorted(set(target.values()))
    controller_peer_id = before.get("controller_peer_id")
    include_controller = controller_peer_id in peer_ids
    payload: dict[str, Any] = {
        "operation": "restore",
        "status": "DRY_RUN" if dry_run else "PASS",
        "dry_run": dry_run,
        "base_url": normalize_base_url(base_url),
        "collection": collection,
        "snapshot_path": str(snapshot_path.expanduser().resolve()),
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "identity": snapshot["identity"],
        "before_placement": serialized_placement(before["placement"]),
        "target_placement": serialized_placement(target),
        "planned_moves": moves,
        "mutation_scope": "numeric_shard_move_only",
        "forbidden_mutations": [
            "collection_create_delete_or_update",
            "artifact_install_or_activation",
            "container_resource_change",
        ],
    }
    if dry_run:
        payload["move_proof"] = None
        payload["after_placement"] = None
        payload["exact_snapshot_verified"] = before["placement"] == target
        return _write_proof(proof_output, payload)

    move_proof = experiment.move_numeric_shards_explicit(
        base_url,
        collection,
        peer_ids,
        target,
        expected_shard_count=FORMAL_SHARD_COUNT,
        transfer_method=transfer_method,
        include_controller=include_controller,
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
    )
    after = inspect_collection(base_url, collection)
    _assert_snapshot_live_contract(snapshot, after, require_original_placement=True)
    payload["move_proof"] = move_proof
    payload["after_placement"] = serialized_placement(after["placement"])
    payload["exact_snapshot_verified"] = True
    return _write_proof(proof_output, payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    snapshot = subparsers.add_parser("snapshot", help="freeze the current placement")
    snapshot.add_argument("--base-url", required=True)
    snapshot.add_argument("--collection", required=True)
    snapshot.add_argument("--output", type=Path, required=True)

    for operation in ("apply", "restore"):
        command = subparsers.add_parser(operation)
        command.add_argument("--base-url", required=True)
        command.add_argument("--collection", required=True)
        command.add_argument("--snapshot", type=Path, required=True)
        command.add_argument("--snapshot-sha256", required=True)
        command.add_argument("--proof-output", type=Path, required=True)
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--transfer-method", default="snapshot")
        command.add_argument("--timeout-sec", type=float, default=3600.0)
        command.add_argument("--poll-interval-sec", type=float, default=1.0)
    apply = subparsers.choices["apply"]
    apply.add_argument("--prepare-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.collection.strip():
        raise ValueError("--collection must be explicitly named")
    if args.operation == "snapshot":
        record = snapshot_collection(
            base_url=args.base_url,
            collection=args.collection,
            output=args.output,
        )
        result = {
            "snapshot_path": str(args.output.expanduser().resolve()),
            "snapshot_sha256": record["snapshot_sha256"],
            "snapshot_file_sha256": sha256_path(args.output.expanduser().resolve()),
        }
    elif args.operation == "apply":
        record = apply_prepare_placement(
            base_url=args.base_url,
            collection=args.collection,
            snapshot_path=args.snapshot,
            snapshot_sha256=args.snapshot_sha256,
            prepare_manifest=args.prepare_manifest,
            proof_output=args.proof_output,
            dry_run=args.dry_run,
            transfer_method=args.transfer_method,
            timeout_sec=args.timeout_sec,
            poll_interval_sec=args.poll_interval_sec,
        )
        result = {
            "proof_path": str(args.proof_output.expanduser().resolve()),
            "proof_sha256": record["proof_sha256"],
            "status": record["status"],
        }
    else:
        record = restore_snapshot_placement(
            base_url=args.base_url,
            collection=args.collection,
            snapshot_path=args.snapshot,
            snapshot_sha256=args.snapshot_sha256,
            proof_output=args.proof_output,
            dry_run=args.dry_run,
            transfer_method=args.transfer_method,
            timeout_sec=args.timeout_sec,
            poll_interval_sec=args.poll_interval_sec,
        )
        result = {
            "proof_path": str(args.proof_output.expanduser().resolve()),
            "proof_sha256": record["proof_sha256"],
            "status": record["status"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
