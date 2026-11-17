#!/usr/bin/env python3
"""Reuse native Orion/KMeans indexes, with strict matched-build preflight.

All archive reads are immutable; only independently extracted segment copies may
be opened by the native loader. Default mode inspects small archive metadata.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import time
import subprocess
import shlex
import uuid
from contextlib import contextmanager, nullcontext

import numpy as np

if __package__:
    from . import shard_locality as core
else:
    import shard_locality as core


def member_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return str(path)


def checked_read(tar, member, manifest):
    name = member_name(member.name)
    data = tar.extractfile(member).read()
    expected = manifest[name]
    if len(data) != expected["size"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError(f"archive metadata checksum mismatch: {name}")
    return data


def inspect_archives(folder, method, collection="sift100m"):
    """Use seekable uncompressed tar headers; do not read vector/index contents."""
    records = []
    collection_config = None
    artifact = None
    seen_shards = set()
    for path in sorted(folder.glob("*.tar")):
        manifest_path = path.with_suffix(".manifest.json")
        manifest = core.read_json(manifest_path)
        prefixes = sorted({str(PurePosixPath(k).parent) for k in manifest if k.endswith("/segment.json")})
        if not prefixes:
            raise ValueError(f"no segments in {path}")
        shard_ids = {int(PurePosixPath(prefix).parent.parent.name) for prefix in prefixes}
        if len(shard_ids) != 1 or seen_shards & shard_ids:
            raise ValueError("archives must each contain one distinct full shard")
        shard = next(iter(shard_ids))
        seen_shards.add(shard)
        with tarfile.open(path, "r:") as tar:
            members = {member_name(m.name): m for m in tar.getmembers()}
            config_name = next(k for k in members if k.endswith(f"collections/{collection}/config.json"))
            config = json.loads(checked_read(tar, members[config_name], manifest))
            policy = config["auto_shard_policy"]
            expected_type = "orion" if method == "orion" else "simple_kmeans"
            if policy["type"] != expected_type:
                raise ValueError("wrong archived routing policy")
            if collection_config is None:
                collection_config = config
            elif config != collection_config:
                raise ValueError("archive collection configs disagree")
            if method == "kmeans" and artifact is None:
                name = str(PurePosixPath(config_name).parent / "simple_kmeans_router" / f'generation-{policy["generation"]}.json')
                data = checked_read(tar, members[name], manifest)
                if hashlib.sha256(data).hexdigest() != policy["artifact_sha256"]:
                    raise ValueError("KMeans artifact/config checksum mismatch")
                artifact = json.loads(data)
            graph_configs = []
            for prefix in prefixes:
                state = json.loads(checked_read(tar, members[prefix + "/segment.json"], manifest))
                vector = state["config"]["vector_data"][""]
                if vector["index"]["type"] == "hnsw":
                    graph_configs.append(json.loads(checked_read(tar, members[prefix + "/vector_index/hnsw_config.json"], manifest)))
            records.append({"archive": str(path.resolve()), "manifest": str(manifest_path.resolve()),
                            "manifest_sha256": core.sha256(manifest_path), "shard": shard,
                            "segments": prefixes, "graph_configs": graph_configs})
    if collection_config is None or seen_shards != set(range(collection_config["params"]["shard_number"])):
        raise ValueError("missing archives: need every numeric shard, starting at 0")
    return records, collection_config, artifact


def extract_segments(record, destination):
    manifest = core.read_json(record["manifest"])
    if core.sha256(record["manifest"]) != record["manifest_sha256"]:
        raise ValueError("archive manifest changed since preflight")
    prefixes = tuple(prefix + "/" for prefix in record["segments"])
    extracted = set()
    with tarfile.open(record["archive"], "r:") as tar:
        for member in tar:
            name = member_name(member.name)
            if not name.startswith(prefixes) or member.isdir():
                continue
            if not (member.isfile() or member.islnk()):
                raise ValueError(f"unsupported segment member {name}")
            expected = manifest[name]
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            total = 0
            with tar.extractfile(member) as source, target.open("xb") as sink:
                for chunk in iter(lambda: source.read(8 << 20), b""):
                    sink.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
            if digest.hexdigest() != expected["sha256"] or total != expected["size"]:
                raise ValueError(f"archived segment checksum mismatch: {name}")
            extracted.add(name)
    if extracted != {name for name in manifest if name.startswith(prefixes)}:
        raise ValueError("missing archived segment files")
    return [str(destination / prefix) for prefix in record["segments"]]


def save_status(output, stage, **details):
    status = {"stage": stage, "time": time.time(), "pid": os.getpid(), **details}
    temp = output / "status.tmp.json"
    temp.write_text(json.dumps(status, indent=2, allow_nan=False))
    temp.replace(output / "status.json")
    print(json.dumps(status, allow_nan=False), flush=True)


def require_matched_indexes(config, groups):
    """No warning-only acceptance of different construction parameters."""
    for method, records, collection_config in groups:
        for key in ("m", "ef_construct"):
            expected = config["lower"][key]
            if collection_config["hnsw_config"][key] != expected:
                raise ValueError(f"{method} archived {key}={collection_config['hnsw_config'][key]} != required {expected}; prepare a matching index before formal search")
        for record in records:
            for graph in record["graph_configs"]:
                if (graph["m"] != config["lower"]["m"] or
                        graph["m0"] != 2 * config["lower"]["m"] or
                        graph["ef_construct"] != config["lower"]["ef_construct"]):
                    raise ValueError(f"{method} shard {record['shard']} contains an unmatched physical HNSW graph")
    reference = groups[0][2]
    for method, _, current in groups[1:]:
        if (current["params"]["vectors"] != reference["params"]["vectors"] or
                current.get("quantization_config") != reference.get("quantization_config") or
                current["hnsw_config"]["full_scan_threshold"] != reference["hnsw_config"]["full_scan_threshold"]):
            raise ValueError(f"{method} vector/quantization/full-scan settings differ")


def require_upper_build(config):
    """Check the native build metadata without reading its millions of sample IDs."""
    path = Path(config["orion_manifest"]).parent / "build-manifest.json"
    with path.open() as source:
        prefix = source.read(65536)
    import re
    match = re.search(r'"options"\s*:\s*', prefix)
    if not match:
        raise ValueError("missing native upper build options")
    options, _ = json.JSONDecoder().raw_decode(prefix[match.end():])
    if (options["upper_m"] != config["lower"]["m"] or
            options["upper_ef_construct"] != config["lower"]["ef_construct"]):
        raise ValueError("Orion upper graph construction parameters do not match the formal experiment")


SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def remote_python(host, code):
    return subprocess.run(SSH + [host, "python3 -c " + shlex.quote(code)],
                          text=True, capture_output=True, check=True, timeout=60).stdout


@contextmanager
def host_leases(hosts):
    """Hold the shared experiment locks without changing host service settings."""
    processes = []
    hostnames = set()
    code = ("import fcntl,socket,sys; "
            "f=open('/tmp/orion-eight-host-benchmark.lock','a+'); "
            "fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); "
            "print(socket.gethostname(),flush=True); sys.stdin.read()")
    try:
        for host in hosts:
            process = subprocess.Popen(SSH + [host, "python3 -c " + shlex.quote(code)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(process)
            import select
            if not select.select([process.stdout], [], [], 20)[0]:
                raise RuntimeError(f"{host}: experiment lease timed out")
            name = process.stdout.readline().strip()
            if not name:
                raise RuntimeError(f"{host}: experiment lock is occupied or unreachable")
            if name in hostnames:
                raise RuntimeError("eight-host mapping resolves to duplicate physical hostnames")
            hostnames.add(name)
        yield
    finally:
        for process in processes:
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)


def remote_segments(binary, request, stem, host, working):
    """One physical shard per configured host; only query-result files return."""
    token = uuid.uuid4().hex
    remote = Path("/dev/shm/orion-shard-locality") / token
    local = Path(working)
    needed = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
    prepare = ("import pathlib,shutil,json; "
        f"p=pathlib.Path({str(remote)!r}); p.parent.mkdir(parents=True,exist_ok=True); "
        f"assert shutil.disk_usage(p.parent).free>{needed + (1 << 30)},'insufficient staging space'; "
        f"p.mkdir(); (p/'.owner').write_text({token!r}); print('ready')")
    remote_python(host, prepare)
    try:
        import shutil
        for key in ("queries", "tasks", "truth_ids"):
            shutil.copyfile(request[key], local / (key + ".input"))
        shutil.copyfile(binary, local / "shard_locality")
        (local / "shard_locality").chmod(0o700)
        remote_request = dict(request)
        for key in ("queries", "tasks", "truth_ids"):
            remote_request[key] = str(remote / (key + ".input"))
        remote_request["segments"] = [str(remote / Path(p).relative_to(local)) for p in request["segments"]]
        core.write_json(local / "request.json", remote_request)
        with stem.with_suffix(".log").open("x") as log:
            subprocess.run(["rsync", "-a", "--protect-args", str(local) + "/", f"{host}:{remote}/"],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(SSH + [host, shlex.join([str(remote / "shard_locality"), "segments",
                str(remote / "request.json"), str(remote / "results.jsonl")])],
                check=True, stdout=log, stderr=subprocess.STDOUT)
            result = stem.with_suffix(".jsonl")
            if result.exists():
                raise FileExistsError(result)
            subprocess.run(["rsync", "-a", "--protect-args", f"{host}:{remote}/results.jsonl", str(result)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        core.write_json(stem.with_suffix(".request.json"), {"host": host, "request": remote_request})
        return result
    finally:
        remote_python(host, "import pathlib,shutil; "
            f"p=pathlib.Path({str(remote)!r}); "
            f"assert p.parent==pathlib.Path('/dev/shm/orion-shard-locality') and (p/'.owner').read_text()=={token!r}; "
            "shutil.rmtree(p)")


def search_method(config, manifest, files, binary, output, method, truth, records, artifact_path):
    folder = output / method
    folder.mkdir()
    shards, top, query_count = manifest["shard_count"], truth.shape[1], len(truth)
    if method == "orion":
        # One frozen configuration for the conditional Real curve. No pooling
        # repeated queries across different parameter settings.
        selected = config.get("real_profile", config["orion_profiles"][0]["name"])
        profile = next(dict(p) for p in config["orion_profiles"] if p["name"] == selected)
        profile["whitebox"] = dict(profile["whitebox"], distance=config["distance"])
        profiles = [profile]
        layout_sha = manifest["assignments_sha256"]
        physical_count = manifest["total_point_copies"]
    else:
        artifact = core.read_json(artifact_path)
        profiles = [{"name": f"nprobe-{n}", "nprobe": n, "lower_hnsw_ef": config["lower"]["kmeans_ef"]}
                    for n in range(1, shards + 1)]
        layout_sha = artifact["layout_sha256"]
        physical_count = artifact["physical_point_count"]
    base = {"method": method, "artifact": str(artifact_path), "queries": str(output / "queries.f32le"),
            "query_count": query_count, "dimension": manifest["dimension"], "shard_count": shards,
            "point_count": manifest["point_count"], "physical_point_count": physical_count,
            "distance": config["distance"], "datatype": config.get("datatype", "float32"), "layout_sha256": layout_sha}
    collect_ann = config.get('collect_ann_diagnostic', False)
    routes, tasks = {}, defaultdict(list)
    for profile in profiles:
        save_status(output, "routing", method=method, profile=profile["name"])
        request = dict(base, **{k: v for k, v in profile.items() if k != "name"})
        result = core.worker(binary, "route", request, folder / ("route-" + profile["name"]))
        rows = core.load_routes(result, query_count, shards, method, profile.get("nprobe"))
        routes[profile["name"]] = rows
        if collect_ann:
            for q, targets in enumerate(rows):
                for target in targets:
                    tasks[target["shard_id"]].append({"profile": profile["name"], "query_index": q, "target": target})
    merged = {p["name"]: [[] for _ in truth] for p in profiles} if collect_ann else None
    memberships = {int(i): [] for i in np.unique(truth)}
    counts = [0] * shards
    segment_metadata = []
    def search_one(record):
        shard = record["shard"]
        task_path = folder / f"tasks-{shard}.jsonl"
        with task_path.open("x") as sink:
            for task in tasks[shard]:
                sink.write(json.dumps(task) + "\n")
        # TempDirectory owns only newly extracted files, never source archives.
        with tempfile.TemporaryDirectory(prefix=f"{method}-{shard}-", dir=output) as working:
            segments = extract_segments(record, Path(working))
            request = {"segments": segments, "queries": str(output / "queries.f32le"),
                       "query_count": query_count, "dimension": manifest["dimension"],
                       "distance": config["distance"], "vector_name": manifest["vector_name"],
                       "tasks": str(task_path), "truth_ids": str(output / "truth-ids.u64le"), "top_k": top}
            execution = config.get("execution", {"mode": "single_machine"})
            if execution["mode"] == "eight_hosts":
                result = remote_segments(binary, request, folder / f"existing-{shard}",
                                         execution["hosts"][shard], working)
            else:
                result = core.worker(binary, "segments", request, folder / f"existing-{shard}")
            return shard, result, len(segments)

    parallel = config.get("execution", {}).get("mode") == "eight_hosts"
    save_status(output, "searching_existing_shards" if collect_ann else "reading_gt_memberships", method=method, shard_count=shards,
                worker_hosts=shards if parallel else 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=shards if parallel else 1) as pool:
        # Reduce in stable shard order; scheduling cannot change merge tie breaks.
        for shard, result, segment_count in pool.map(search_one, sorted(records, key=lambda r: r["shard"])):
            expected_rows = 0
            actual_rows = 0
            segment_ids = set()
            with result.open() as source:
                for line in source:
                    row = json.loads(line)
                    if row["kind"] == "membership":
                        if row["segment_index"] in segment_ids:
                            raise ValueError("duplicate membership response")
                        segment_ids.add(row["segment_index"])
                        counts[shard] += row["point_count"]
                        if row["point_count"]:
                            expected_rows += len(tasks[shard])
                        segment_metadata.append({"shard": shard, **{k:v for k,v in row.items() if k != "gt_ids"}})
                        for point in row["gt_ids"]:
                            if point not in memberships or shard in memberships[point]:
                                raise ValueError("unexpected/duplicate GT point across live segments")
                            memberships[point].append(shard)
                    else:
                        if merged is None:
                            raise ValueError('unexpected ANN result in GT-only evaluation')
                        expected = tasks[shard][actual_rows % len(tasks[shard])]
                        if (row["profile"], row["query_index"]) != (expected["profile"], expected["query_index"]):
                            raise ValueError("existing-index result order mismatch")
                        old = merged[row["profile"]][row["query_index"]]
                        merged[row["profile"]][row["query_index"]] = core.merge_topk(old, row["points"], top, config["distance"])
                        actual_rows += 1
            if expected_rows != actual_rows or len(segment_ids) != segment_count:
                raise ValueError("incomplete existing-index responses")
    if sum(counts) != physical_count or any(not owners for owners in memberships.values()):
        raise ValueError("archive point count or GT coverage mismatch")
    if method == "kmeans" and any(len(owners) != 1 for owners in memberships.values()):
        raise ValueError("KMeans archives have overlapping GT memberships")
    core.write_json(folder / "segments.json", segment_metadata)
    core.write_json(folder / "gt-memberships.json", memberships)
    oracle, masks = core.oracle_matrix(truth, memberships, shards)
    np.save(folder / "oracle.npy", oracle, allow_pickle=False)
    return core.summarize_method(config, manifest, folder, method, profiles, routes, merged,
                                 truth, counts, oracle, masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--orion-archives", type=Path, required=True)
    parser.add_argument("--kmeans-archives", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config, manifest, files = core.load_plan(args.config)
    orion, orion_config, _ = inspect_archives(args.orion_archives, "orion")
    kmeans, kmeans_config, artifact = inspect_archives(args.kmeans_archives, "kmeans")
    for config_archived in (orion_config, kmeans_config):
        if (config_archived["params"]["shard_number"] != manifest["shard_count"] or
                config_archived["params"]["vectors"]["size"] != manifest["dimension"] or
                config_archived["params"]["vectors"]["distance"] != config["distance"]):
            raise ValueError("archive/config shard count or vector schema mismatch")
    if artifact["logical_point_count"] != manifest["point_count"]:
        raise ValueError("KMeans/Orion logical dataset size mismatch")
    if orion_config["auto_shard_policy"]["artifact_sha256"] != manifest["orion_artifact_sha256"]:
        raise ValueError("Orion archives are bound to a different layout")
    require_matched_indexes(config, [("orion", orion, orion_config), ("kmeans", kmeans, kmeans_config)])
    if config.get("execution", {}).get("mode") == "eight_hosts":
        require_upper_build(config)
    report = {"shards": manifest["shard_count"], "logical_points": manifest["point_count"],
              "orion": orion, "kmeans": kmeans,
              "orion_hnsw": orion_config["hnsw_config"], "kmeans_hnsw": kmeans_config["hnsw_config"],
              "rebuilt": False, "real": "conditional mean per integer fanout; empty bins N/A",
              "execution": config.get("execution", {"mode":"single_machine"}),
              "matched_index_parameters_required": True}
    if not args.execute:
        print(json.dumps(report, indent=2))
        return
    if args.output is None or args.native_binary is None or not os.access(args.native_binary, os.X_OK):
        parser.error("--execute requires --output and executable --native-binary")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    core.write_json(output / "config.json", config)
    core.write_json(output / "archive-preflight.json", report)
    core.write_json(output / "kmeans-artifact.json", artifact)
    _, truth = core.load_queries(config, manifest, output)
    np.unique(truth).astype("<u8").tofile(output / "truth-ids.u64le")
    save_status(output, "verifying_artifact")
    if core.sha256(files["orion_artifact"]) != manifest["orion_artifact_sha256"]:
        raise ValueError("Orion artifact checksum mismatch")
    core.write_json(output / "provenance.json", {
        "binary_sha256": core.sha256(args.native_binary), "queries_sha256": core.sha256(output / "queries.f32le"),
        "truth_sha256": core.sha256(output / "truth.npy"), "source_script_sha256": core.sha256(Path(__file__)),
        "orion_artifact_sha256": manifest["orion_artifact_sha256"], "qps_measured": False,
        "reuses_existing_indexes": True, "rebuilds_or_retrains": False, "metric_schema": core.METRIC_SCHEMA})
    curves, details, groups = [], [], []
    try:
        execution = config.get("execution", {"mode":"single_machine"})
        lease = host_leases(execution["hosts"]) if execution["mode"] == "eight_hosts" else nullcontext()
        with lease:
            for method, records, artifact_path in [("orion", orion, files["orion_artifact"]),
                                                    ("kmeans", kmeans, output / "kmeans-artifact.json")]:
                result = search_method(config, manifest, files, args.native_binary.resolve(), output,
                                       method, truth, records, artifact_path)
                for name, rows in zip(("curves", "per_query", "real_by_fanout"), result):
                    core.write_csv(output / method / f"{name}.csv", rows)
                curves.extend(result[0]); details.extend(result[1]); groups.extend(result[2])
        for name, rows in (("curves", curves), ("per_query", details), ("real_by_fanout", groups)):
            core.write_csv(output / f"{name}.csv", rows)
        core.plot(curves, output, truth.shape[1], manifest["shard_count"])
        core.write_json(output / "COMPLETED.json", {"complete": True, "qps_measured": False, "metric_schema": core.METRIC_SCHEMA})
        save_status(output, "complete")
    except KeyboardInterrupt:
        save_status(output, "interrupted")
        raise
    except Exception as error:
        save_status(output, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
