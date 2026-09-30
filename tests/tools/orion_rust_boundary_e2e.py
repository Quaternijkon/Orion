#!/usr/bin/env python3
"""Isolated real-node acceptance for the Rust boundary. Never uses existing storage.

This is external API/process test orchestration; all authority stays in Rust.
The output directory retains HTTP evidence and node logs, including failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant", required=True)
    parser.add_argument("--layout-cli", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--streaming", action="store_true", help="Exercise concurrent build-install and idempotent retry.")
    parser.add_argument("--uint8", action="store_true", help="Exercise lossless u8bin input, uint8 storage and byte metrics.")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    key = "orion-isolated-acceptance"
    ports = [[free_port() for _ in range(3)] for _ in range(4)]
    processes: dict[int, subprocess.Popen] = {}
    logs = []
    events = []
    report = {"scope": "four local peers on one host; no performance claim", "checks": {}, "ports": ports}
    for label, binary in [("qdrant", args.qdrant), ("layout_cli", args.layout_cli)]:
        digest = hashlib.sha256()
        with Path(binary).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        report[f"{label}_sha256"] = digest.hexdigest()

    def api(node, method, path, body=None, timeout=90):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{ports[node][0]}{path}", data=data,
                                         headers={"Content-Type": "application/json", "api-key": key}, method=method)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            events.append({"node": node, "method": method, "path": path, "elapsed": time.monotonic()-started,
                           "ok": True})
            return result.get("result", result)
        except Exception as error:
            detail = error.read().decode(errors="replace") if isinstance(error, urllib.error.HTTPError) else str(error)
            events.append({"node": node, "method": method, "path": path, "elapsed": time.monotonic()-started,
                           "ok": False, "error": detail})
            raise RuntimeError(detail) from error

    def wait(predicate, seconds=120):
        deadline = time.monotonic()+seconds
        error = None
        while time.monotonic() < deadline:
            try:
                value = predicate()
                if value:
                    return value
            except Exception as exc:
                error = exc
            time.sleep(0.1)
        raise RuntimeError(f"wait timed out: {error}")

    def start(node):
        directory = output / f"node-{node}"
        directory.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update({
            "QDRANT__SERVICE__HOST": "127.0.0.1", "QDRANT__SERVICE__HTTP_PORT": str(ports[node][0]),
            "QDRANT__SERVICE__GRPC_PORT": str(ports[node][1]), "QDRANT__CLUSTER__ENABLED": "true",
            "QDRANT__CLUSTER__P2P__PORT": str(ports[node][2]), "QDRANT__STORAGE__STORAGE_PATH": str(directory / "storage"),
            "QDRANT__STORAGE__SNAPSHOTS_PATH": str(directory / "snapshots"),
            "QDRANT__STORAGE__TEMP_PATH": str(directory / "temp"), "QDRANT__SERVICE__API_KEY": key,
            "QDRANT__SERVICE__MAX_WORKERS": "2", "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS": "2",
            "QDRANT__LOG_LEVEL": "warn", "QDRANT__TELEMETRY_DISABLED": "true",
        })
        command = [str(Path(args.qdrant).resolve()), "--uri", f"http://127.0.0.1:{ports[node][2]}"]
        if node:
            command += ["--bootstrap", f"http://127.0.0.1:{ports[0][2]}"]
        log = (directory / "server.log").open("ab")
        logs.append(log)
        processes[node] = subprocess.Popen(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
        wait(lambda: api(node, "GET", "/collections"), 90)

    def stop(node, hard=False):
        process = processes.pop(node, None)
        if process and process.poll() is None:
            process.kill() if hard else process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    try:
        raw = output / ("raw.u8bin" if args.uint8 else "raw.f32le")
        vectors = [[(i % 16) * 16, (i // 16) * 64] for i in range(64)] if args.uint8 else [[i % 16 / 8 - 1, i // 16 - 1.5] for i in range(64)]
        raw.write_bytes(struct.pack("<II", 64, 2) + bytes(x for row in vectors for x in row) if args.uint8 else b"".join(struct.pack("<ff", *row) for row in vectors))
        request = {"input": {"format_version": 1, "path": str(raw), "rows": 64, "dimension": 2,
                             "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()},
                   "options": {"generation": 1, "shard_count": 2, "distance": "Cosine", "vector_name": "",
                               "sample_denominator": 4, "sample_seed": 100, "upper_m": 4, "upper_ef_construct": 32,
                               "upper_seed": 100, "efs": 3, "upper_k": 4, "upper_ef_search": 4,
                               "dynamic_ef_base": 20, "dynamic_ef_factor": 4, "kmeans_iterations": 10,
                               "kmeans_seed": 1, "topology_iterations": 100, "bundle_prefix": "import"}}
        if args.uint8:
            request["input"]["encoding"] = "u8bin"
            request["options"]["distance"] = "Euclid"
        request_path = output / "request.json"
        request_path.write_text(json.dumps(request))
        bundle = output / "bundle"
        reference = output / "reference" if args.streaming else bundle
        with (output / "builder.log").open("wb") as log:
            subprocess.run([args.layout_cli, "build", "--request", str(request_path), "--output", str(reference)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((reference / "import.manifest.json").read_text())
        artifact = json.loads((reference / manifest["orion_artifact_file"]).read_text())
        for node in range(3):
            start(node)
        wait(lambda: len(api(0, "GET", "/cluster")["peers"]) == 3)
        name = "boundary"
        api(0, "PUT", f"/collections/{name}?timeout=60", {
            "vectors": {"size": 2, "distance": "Euclid" if args.uint8 else "Cosine", "datatype": "uint8" if args.uint8 else "float32"}, "shard_number": 2,
            "replication_factor": 2, "write_consistency_factor": 1,
            "optimizers_config": {"default_segment_number": 1, "max_optimization_threads": 1},
            "hnsw_config": {"m": 4, "ef_construct": 32},
            "auto_shard_policy": {"type": "orion", "generation": 1,
                                  "artifact_sha256": "0" * 64 if args.streaming else manifest["orion_artifact_sha256"]},
        })
        install_command = ([args.layout_cli, "build-install", "--request", str(request_path), "--output", str(bundle)]
                           if args.streaming else [args.layout_cli, "install", "--bundle", str(bundle)])
        install_command += ["--uri", f"http://127.0.0.1:{ports[0][0]}", "--collection", name,
                            "--timeout-secs", "90", "--api-key", key]
        with (output / "install.log").open("wb") as log:
            subprocess.run(install_command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        if args.streaming:
            for path in reference.iterdir():
                assert path.read_bytes() == (bundle / path.name).read_bytes(), path.name
            with (output / "install-retry.log").open("wb") as log:
                subprocess.run(install_command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
            report["checks"]["streamed_bundle_parity_and_retry"] = True
        state = api(0, "GET", f"/collections/{name}/orion")
        assert all(peer["phase"] == "Ready" and peer["active"]["generation"] == 1 for peer in state)
        report["checks"]["consensus_install"] = state
        if args.uint8:
            assert artifact["vector_schema"]["datatype"] == "uint8"
            assert (bundle / manifest["vectors_file"]).stat().st_size == 64 * 2
            assert not list(bundle.glob("*.f32le"))
            for start_id in range(0, 64, 16):
                records = api(0, "POST", f"/collections/{name}/points", {"ids": list(range(start_id,start_id+16)), "with_vector": True})
                assert len(records) == 16
                for row in records:
                    assert row["vector"] == vectors[row["id"]]
            # Identical lower datatype and metric, independent ordinary hash collection.
            api(0, "PUT", "/collections/uint8_control?timeout=60", {
                "vectors": {"size": 2, "distance": "Euclid", "datatype": "uint8"},
                "shard_number": 2, "hnsw_config": {"m": 4, "ef_construct": 32}})
            api(0, "PUT", "/collections/uint8_control/points?wait=true", {
                "points": [{"id": i, "vector": v} for i,v in enumerate(vectors)]})
            for query in [[0,0], [128,128], [255,255], [127.9,192.0]]:
                body = {"vector": query, "limit": 5, "params": {"hnsw_ef": 100}}
                actual = api(0, "POST", f"/collections/{name}/points/search", body)
                control = api(0, "POST", "/collections/uint8_control/points/search", body)
                assert [r["score"] for r in actual] == [r["score"] for r in control], (query, actual, control)
            report["checks"]["uint8_compact_bundle_exact_retrieval_and_hashall_scores"] = True
        inserted = [123.0, 15.0] if args.uint8 else [123.0, 0.15]
        updated = [250.0, 15.0] if args.uint8 else [-123.0, 0.15]
        point_id = str(uuid.uuid4())
        api(0, "PUT", f"/collections/{name}/points?wait=true&ordering=strong&timeout=60", {
            "points": [{"id": point_id, "vector": inserted, "payload": {"marker": "uuid", "value": 1}}]})
        for node in range(3):
            record = api(node, "POST", f"/collections/{name}/points", {"ids": [point_id], "with_payload": True, "with_vector": True})
            assert len(record) == 1 and record[0]["payload"]["value"] == 1
        api(1, "POST", f"/collections/{name}/points/payload?wait=true&timeout=60", {"points": [point_id], "payload": {"value": 2}})
        result = wait(lambda: api(2, "POST", f"/collections/{name}/points/search", {"vector": inserted, "limit": 5}))
        assert any(row["id"] == point_id for row in result)
        api(0, "PUT", f"/collections/{name}/points/vectors?wait=true&timeout=60", {"points": [{"id": point_id, "vector": updated}]})
        record = api(1, "POST", f"/collections/{name}/points", {"ids": [point_id], "with_payload": True, "with_vector": True})[0]
        assert record["payload"]["value"] == 2 and (record["vector"][0] == updated[0] if args.uint8 else record["vector"][0] < 0)
        api(0, "POST", f"/collections/{name}/points/vectors/delete?wait=true&timeout=60", {"points": [point_id], "vector": [""]})
        record = api(2, "POST", f"/collections/{name}/points", {"ids": [point_id], "with_payload": True, "with_vector": True})[0]
        assert record.get("vector") in (None, {}) and record["payload"]["value"] == 2
        api(1, "PUT", f"/collections/{name}/points/vectors?wait=true&timeout=60", {"points": [{"id": point_id, "vector": updated}]})
        api(0, "POST", f"/collections/{name}/points/payload?wait=true&timeout=60", {"payload": {"filtered": True}, "filter": {"must": [{"key": "marker", "match": {"value": "uuid"}}]}})
        assert api(2, "POST", f"/collections/{name}/points", {"ids": [point_id], "with_payload": True})[0]["payload"]["filtered"] is True
        report["checks"]["vector_update_delete_restore_and_filtered_payload"] = True
        anchors = [node["label"] for node in artifact["upper_nodes"]]
        api(0, "POST", f"/collections/{name}/points/delete?wait=true&timeout=60", {"points": anchors})
        result = wait(lambda: api(1, "POST", f"/collections/{name}/points/search", {"vector": [1.0, 1.0], "limit": 5}))
        assert result and all(row["id"] not in anchors for row in result)
        assert api(0, "POST", f"/collections/{name}/points/count", {"exact": True})["count"] == 49
        report["checks"]["ordinary_uuid_payload_delete_and_search"] = True
        # A fourth, initially remote-only peer must obtain artifacts/catalog through Rust RPC.
        start(3)
        wait(lambda: len(api(0, "GET", "/cluster")["peers"]) == 4)
        wait(lambda: len(api(3, "POST", f"/collections/{name}/points", {"ids": [point_id], "with_payload": True})) == 1)
        report["checks"]["new_peer_bootstrap"] = True
        stop(2, hard=True)
        try:
            api(0, "POST", f"/collections/{name}/orion/repartition?timeout=3", {"generation": 2, "shard_count": 3}, timeout=20)
        except RuntimeError:
            report["checks"]["activation_refuses_missing_peer"] = True
        else:
            raise AssertionError("activated a generation without the missing peer")
        start(2)
        wait(lambda: len(api(2, "POST", f"/collections/{name}/points", {"ids": [point_id], "with_payload": True})) == 1)
        report["checks"]["restart_preserves_catalog"] = True
        api(0, "POST", f"/collections/{name}/orion/repartition?timeout=90", {"generation": 2, "shard_count": 3}, timeout=150)
        state = api(0, "GET", f"/collections/{name}/orion")
        assert all(peer["phase"] == "Ready" and peer["active"]["generation"] == 2 for peer in state)
        api(1, "POST", f"/collections/{name}/orion/repartition?timeout=90", {"generation": 3, "shard_count": 2}, timeout=150)
        for node in range(4):
            record = api(node, "POST", f"/collections/{name}/points", {"ids": [point_id], "with_payload": True})
            assert record[0]["payload"]["value"] == 2
            assert api(node, "POST", f"/collections/{name}/points/count", {"exact": True})["count"] == 49
        report["checks"]["distributed_grow_shrink"] = api(0, "GET", f"/collections/{name}/orion")
        # Public callers cannot inject a shard plan through the administrative control endpoint.
        try:
            api(0, "POST", f"/collections/{name}/orion", {"operation": "write", "plan": {
                "token": "untrusted", "generation": 3, "expected_revision": 0, "writes": [], "changes": [], "obsolete": []}})
        except RuntimeError:
            report["checks"]["reject_client_membership_authority"] = True
        else:
            raise AssertionError("public write-plan injection accepted")
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        for node in list(processes):
            stop(node)
        for log in logs:
            log.close()
        (output / "http-events.json").write_text(json.dumps(events, indent=2))
        (output / "verdict.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"status": report.get("status"), "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
