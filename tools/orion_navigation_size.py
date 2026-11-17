#!/usr/bin/env python3
"""Plan or explicitly execute the SIFT1M 8-host navigation-size experiment.

Default: JSON plan only, no SSH, dataset scan, compilation or experiment writes.
Runtime dependencies are imported only inside explicit execution paths.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import select
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "config/orion_navigation_size_sift1m.json"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cpus(value):
    result = []
    for part in value.split(","):
        lo, sep, hi = part.partition("-")
        result.extend(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    require(len(set(result)) == len(result) and min(result) >= 0, "invalid CPU list")
    return sorted(result)


def candidates(config):
    c = config["calibration"]
    return [{"upper_k": k, "upper_ef_search": k * c["upper_ef_multiplier"],
             "dynamic_ef_base": b, "dynamic_ef_factor": c["dynamic_ef_factor"]}
            for k, b in itertools.product(c["upper_k_values"], c["dynamic_ef_base_values"])]


def validate(config):
    require(config["schema_version"] == 1 and config["execution_mode"] == "manual_only", "manual_only required")
    require(len(config["hosts"]) == len(set(config["hosts"])) == 8 and config["shards_per_host"] == 1,
            "exactly eight distinct hosts, one shard each required")
    require(config["denominators"] == [32, 64, 128, 256], "required ratios are 1/32, 1/64, 1/128, 1/256")
    d, b, c, load = (config[k] for k in ("dataset", "build", "calibration", "load"))
    require((d["rows"], d["dimension"], d["queries"], d["distance"], d["top_k"]) ==
            (1000000, 128, 10000, "Euclid", 10), "full SIFT1M / Recall@10 required")
    require(0 < d["calibration_queries"] < d["queries"], "invalid split")
    require((b["m"], b["ef_construct"]) == (32, 100), "user-fixed M=32, ef_construct=100 required")
    require(len(cpus(config["server_cpus"])) == config["search_threads"] == 48, "48 service cores required")
    require(not set(cpus(config["server_cpus"])) & set(cpus(config["client_cpus"])), "CPU lists overlap")
    require(c["target_recall"] == .95 and .95 <= c["selection_floor"] <= c["preferred_ceiling"] <= 1,
            "invalid recall thresholds")
    require(1024 < config["port"] < 65533 and load["api"] == "grpc_SearchBatch", "invalid port/API")
    for key in ("run_root", "remote_root"):
        p = Path(config[key])
        require(p.is_absolute() and len(p.parts) >= 4 and p.name.startswith("orion-navigation-size-"),
                f"{key} must be a dedicated absolute experiment directory")
    require(config["run_root"] != config["remote_root"], "controller and worker directories must differ")
    for key in ("upper_k_values", "dynamic_ef_base_values"):
        require(len(set(c[key])) == len(c[key]) > 0 and all(type(x) is int and x > 0 for x in c[key]), key)
    require(type(c["upper_ef_multiplier"]) is int and c["upper_ef_multiplier"] >= 1, "upper EF multiplier")
    w = config["whitebox"]
    require(w["distance"] == "Euclid" and 0 <= w["max_prefix_entries"] < w["prefix_k"] and
            math.isfinite(w["max_distance_gap"]) and w["max_distance_gap"] >= 0, "invalid whitebox")
    require(w["prefix_k"] <= min(c["upper_k_values"]) and w["prefix_k"] <= b["upper_k"], "whitebox prefix exceeds upper k")
    require(max(c["upper_k_values"]) <= d["rows"] // max(config["denominators"]), "upper k exceeds graph")
    for key in ("batch_size", "workers_per_host", "seconds", "warmup_seconds", "repeats"):
        require(type(load[key]) is int and load[key] > 0, key)
    require(load["repeats"] >= 3 and all(type(x) is int and x > 0 for x in load["lanes_values"]), "invalid load grid")
    require(load["busiest_service_cpu_fraction_minimum"] == .95 and
            .95 < load["screen_cpu_fraction_target"] <= 1, "strict CPU >95% acceptance required")
    require(load["maximum_lanes"] >= max(load["lanes_values"]), "maximum_lanes is below initial grid")
    return config


def make_plan(config):
    validate(config)
    ratios = config["denominators"]
    return {"state": "planned_not_executed", "config": config,
            "cases": [{"denominator": d, "ratio": f"1/{d}", "upper_nodes": 1000000 // d,
                       "actual_ratio": (1000000 // d) / 1000000} for d in ratios],
            "search_candidates_per_ratio": candidates(config),
            "formal_order": [ratios[i % 4:] + ratios[:i % 4] for i in range(config["load"]["repeats"])],
            "stages": ["freeze source and compile", "export and split SIFT1M", "build four native v5 layouts",
                       "deploy and build eight lower indexes per ratio", "calibrate on 2000 queries",
                       "pilot eligible candidates; freeze budgets; screen load on calibration queries",
                       "audit 8000 held-out queries; reject any recall below 0.95",
                       "three interleaved formal 60-second windows per ratio", "archive and summarize"],
            "qps_definition": "successful global Top10 queries completed in fixed window / window seconds; batch counts as 64 queries",
            "no_automatic_wait_or_start": True}


def choose_candidate(rows, config):
    c = config["calibration"]
    good = [r for r in rows if r["recall"] >= c["selection_floor"] and r.get("qps", 0) > 0]
    require(good, "no recall-qualified candidate; retain evidence and stop this ratio")
    band = [r for r in good if r["recall"] <= c["preferred_ceiling"]]
    return max(band or good, key=lambda r: (r["qps"], -r["recall"]))


def saturation_evidence(cpu_cores, config):
    require(set(cpu_cores) == set(config["hosts"]), "CPU evidence must cover all eight hosts")
    require(all(math.isfinite(v) and 0 <= v <= 48 * 1.05 for v in cpu_cores.values()), "invalid service CPU measurement")
    fractions = {host: cores / 48 for host, cores in cpu_cores.items()}
    busiest = max(fractions, key=fractions.get)
    return {"server_cpu_fractions": fractions, "busiest_host": busiest,
            "busiest_service_cpu_fraction": fractions[busiest],
            "cpu_saturated": fractions[busiest] > config["load"]["busiest_service_cpu_fraction_minimum"] + 1e-9}


def qualifies_formal(row, config):
    if row.get("errors") != 0 or row.get("recall_uniform_queries", 0) < .95:
        return False
    try:
        return saturation_evidence(row.get("server_cpu_cores", {}), config)["cpu_saturated"]
    except ValueError:
        return False


def cluster_stable(states):
    if len(states) != 8:
        return False
    rows = list(states.values())
    ids = {str(v.get("peer_id")) for v in rows}
    if len(ids) != 8 or "None" in ids:
        return False
    raft = [v["raft_info"] for v in rows]
    return (all(set(v["peers"]) == ids for v in rows)
            and len({(r["term"], r["leader"]) for r in raft}) == 1
            and str(raft[0]["leader"]) in ids
            and all(r["is_voter"] and r["pending_operations"] == 0 and r["role"] in ("Leader", "Follower") for r in raft)
            and sum(r["role"] == "Leader" for r in raft) == 1)


class Experiment:
    def __init__(self, config):
        self.c = validate(config)
        self.root = Path(config["run_root"])
        self.remote_root = Path(config["remote_root"])
        self.leases = []
        self.servers = {}
        self.clients = {}
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("ORION_", "QDRANT_"))}
        self.env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NO_PROXY="*", no_proxy="*",
                        PYTHONPATH=config["python_path"])

    def each(self, function):
        with ThreadPoolExecutor(8) as pool:
            return dict(zip(self.c["hosts"], pool.map(function, self.c["hosts"])))

    def remote(self, host, code, timeout=120):
        result = subprocess.run(SSH + [host, "python3 -c " + shlex.quote(code)],
                                text=True, capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"{host}: remote command failed ({result.returncode}): {result.stderr[-2500:]}")
        return json.loads(result.stdout)

    def copy(self, host, source, destination):
        subprocess.run(["scp", "-q", str(source), f"{host}:{destination}"], check=True, timeout=600)

    def check(self):
        require(all(p.poll() is None for p in self.leases), "lost an experiment lease")

    def status(self, stage, **details):
        value = {"stage": stage, "epoch": time.time(), **details}
        save(self.root / "status.json", value)
        print(json.dumps(value), flush=True)

    def clock_offsets(self):
        """Estimate monotonic offsets over an established SSH stream, not SSH startup RTT."""
        code = "import sys,time,json\nfor line in sys.stdin: print(json.dumps({'monotonic':time.monotonic(),'epoch':time.time()}),flush=True)"
        def one(host):
            p = subprocess.Popen(SSH + [host, "python3 -u -c " + shlex.quote(code)], text=True,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            samples = []
            try:
                for _ in range(6):
                    before = time.monotonic()
                    p.stdin.write("probe\n")
                    p.stdin.flush()
                    require(select.select([p.stdout], [], [], 15)[0], f"clock probe timeout: {host}")
                    remote = json.loads(p.stdout.readline())
                    after = time.monotonic()
                    samples.append({"offset": remote["monotonic"] - (before + after) / 2,
                                    "rtt": after - before, "remote_epoch": remote["epoch"]})
                selected = min(samples[1:], key=lambda row: row["rtt"])
                require(selected["rtt"] < .1, f"clock probe uncertainty too high: {host}")
                return {**selected, "samples": samples}
            finally:
                p.stdin.close()
                p.wait(timeout=10)
        return self.each(one)

    @contextmanager
    def lease(self):
        code = """import fcntl,pathlib,sys
f=open('/tmp/orion-eight-host-benchmark.lock','a+')
fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
print(pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),flush=True)
sys.stdin.read()
"""
        names = set()
        try:
            for host in self.c["hosts"]:
                p = subprocess.Popen(SSH + [host, "python3 -c " + shlex.quote(code)], text=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.leases.append(p)
                require(select.select([p.stdout], [], [], 20)[0], f"lease timeout: {host}")
                name = p.stdout.readline().strip()
                require(name and name not in names, f"occupied lease or duplicate machine boot ID: {host}")
                names.add(name)
            yield
        finally:
            try:
                self.stop_group(self.clients)
                self.stop_group(self.servers)
            finally:
                for p in self.leases:
                    p.stdin.close()
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        p.terminate()
                self.leases.clear()

    def command(self, args, log, cwd=None, env=None):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("xb") as stream:
            p = subprocess.Popen(list(map(str, args)), cwd=cwd, env=env or self.env,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while p.poll() is None:
                    self.check()
                    time.sleep(2)
                require(p.returncode == 0, f"command failed; inspect {log}")
            finally:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=30)

    def preflight(self, require_direct_network=True):
        """Executed only under leases. No system settings or routes are changed."""
        for module in ("h5py", "numpy", "pymetis", "grpc", "qdrant_client"):
            __import__(module)
        require(shutil.disk_usage(self.root.parent).free >= self.c["minimum_controller_free_gib"] * 2**30,
                "insufficient controller disk")
        marker = json.dumps(self.c, sort_keys=True)
        fingerprint = hashlib.sha256(marker.encode()).hexdigest()
        if self.root.exists():
            require(read(self.root / "config.json") == self.c, "run root belongs to another configuration")
        else:
            self.root.mkdir()
            save(self.root / "config.json", self.c)
        server, client = cpus(self.c["server_cpus"]), cpus(self.c["client_cpus"])
        def one(host):
            before = time.time()
            result = self.remote(host, f'''import os,pathlib,json,socket,subprocess,shutil,time
r=pathlib.Path({str(self.remote_root)!r})
if r.exists():
 assert json.loads((r/'owner.json').read_text())['config_sha256']=={fingerprint!r},'unowned remote root'
else:
 r.mkdir();(r/'owner.json').write_text(json.dumps({{'config_sha256':{fingerprint!r}}}))
assert shutil.disk_usage(r).free>={self.c['minimum_remote_free_gib']}*2**30,'insufficient worker storage'
topology=[list(map(int,x.split(','))) for x in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True).splitlines() if not x.startswith('#')]
service={{(x[1],x[2]) for x in topology if x[0] in {server!r}}}
load={{(x[1],x[2]) for x in topology if x[0] in {client!r}}}
assert len(service)==48 and len(load)==48 and not service&load,'physical CPU overlap or unexpected topology'
for port in range({self.c['port']},{self.c['port']+3}):
 with socket.socket() as s:s.bind(('0.0.0.0',port))
for p in pathlib.Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   if (p/'comm').read_text().strip()=='qdrant':assert (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z','another qdrant active'
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
routes={{h:json.loads(subprocess.check_output(['ip','-j','route','get',h],text=True)) for h in {self.c['hosts']!r} if h!={host!r}}}
if {require_direct_network!r}:assert all(x[0]['dev'] not in ('Meta','tun0') for x in routes.values()),'benchmark traffic crosses proxy'
env=dict(os.environ);env['PYTHONPATH']={self.c['remote_python_path']!r}
subprocess.run([{self.c['remote_python']!r},'-c','import numpy,grpc;from qdrant_client.grpc import SearchBatchResponse'],env=env,check=True)
print(json.dumps({{'host':socket.gethostname(),'routes':routes,'topology':topology,'epoch':time.time(),'free':shutil.disk_usage(r).free}}))''')
            result["wall_clock_offset_estimate"] = result["epoch"] - (before + time.time()) / 2
            return result
        save(self.root / f"preflight-{time.time_ns()}.json", self.each(one))
        save(self.root / "clock-check.json", self.clock_offsets())

    def freeze_and_compile(self):
        if (self.root / "binaries.json").exists():
            for name, digest in read(self.root / "binaries.json").items():
                require(sha(self.root / "bin" / name) == digest, "binary changed")
            for name in ("orion_navigation_size.py", "orion_navigation_size_client.py"):
                require(sha(REPO / "tools" / name) == sha(self.root / "source/tools" / name),
                        "experiment scripts changed after preparation; use the saved source or a fresh run root")
            return
        source = self.root / "source"
        require(not source.exists(), "incomplete source/build exists; inspect logs and use a fresh run root")
        paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=REPO).decode().split("\0")
        manifest = {}
        for rel in sorted(set(paths)):
            if not rel or rel.startswith(("datasets/", "docs/")):
                continue
            p = REPO / rel
            if not p.is_file():
                continue
            target = source / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            manifest[rel] = sha(target)
        save(self.root / "source-files.json", manifest)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        save(self.root / "source-identity.json", {"commit": commit, "dirty_files_included": True,
              "status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)})
        script = '''set -euo pipefail
source "$1"
export CARGO_TARGET_DIR="$2"
export GIT_COMMIT_ID="$3"
cargo build --release --locked --offline -p qdrant --bin qdrant -p collection --bin orion_build_layout -j "$4"
cargo build --release --locked --offline -p collection --example native_whitebox_diagnostic -j "$4"
'''
        self.command(["bash", "-c", script, "build", self.c["toolchain_env"], self.root / "target", commit,
                      self.c["build_jobs"]], self.root / "compile.log", cwd=source)
        (self.root / "bin").mkdir()
        for name, sub in (("qdrant", ""), ("orion_build_layout", ""), ("native_whitebox_diagnostic", "examples")):
            shutil.copy2(self.root / "target/release" / sub / name, self.root / "bin" / name)
        save(self.root / "binaries.json", {p.name: sha(p) for p in (self.root / "bin").iterdir()})

    def export_data(self):
        import h5py
        import numpy as np
        dest = self.root / "data"
        if (dest / "identity.json").exists():
            for name, digest in read(dest / "identity.json").items():
                require(sha(dest / name) == digest, "exported data changed")
            return
        dest.mkdir(exist_ok=False)
        d = self.c["dataset"]
        require(sha(d["path"]) == d["sha256"], "SIFT1M HDF5 checksum mismatch")
        with h5py.File(d["path"]) as data:
            require(data["train"].shape == (d["rows"], d["dimension"]) and
                    data["test"].shape == (d["queries"], d["dimension"]), "SIFT1M shape mismatch")
            with (dest / "train.f32le").open("xb") as out:
                for i in range(0, d["rows"], 10000):
                    chunk = np.asarray(data["train"][i:i+10000], dtype="<f4")
                    require(np.isfinite(chunk).all(), "non-finite train vectors")
                    out.write(chunk.tobytes())
            queries = data["test"][:].astype("<f4")
            truth = data["neighbors"][:, :10].astype(np.int64)
            require(truth.shape == (10000, 10) and (truth >= 0).all() and (truth < 1000000).all(), "invalid GT")
            require(np.isfinite(queries).all() and (np.diff(np.sort(truth, axis=1), axis=1) != 0).all(), "invalid queries/GT")
            np.savez(dest / "queries.npz", queries=queries, truth=truth)
            order = np.random.default_rng(d["split_seed"]).permutation(d["queries"])
            n = d["calibration_queries"]
            save(dest / "split.json", {"calibration": order[:n].tolist(), "test": order[n:].tolist(), "seed": d["split_seed"]})
        save(dest / "identity.json", {name: sha(dest / name) for name in ("train.f32le", "queries.npz", "split.json")})

    def deploy_files(self):
        files = [(self.root / "bin/qdrant", "qdrant"), (self.root / "data/queries.npz", "queries.npz"),
                 (self.root / "source/tools/orion_navigation_size_client.py", "client.py")]
        save(self.root / "whitebox.json", self.c["whitebox"])
        files.append((self.root / "whitebox.json", "whitebox.json"))
        def one(host):
            for src, name in files:
                self.copy(host, src, self.remote_root / name)
            return self.remote(host, f'''import pathlib,hashlib,json
r=pathlib.Path({str(self.remote_root)!r})
expected={ {name:sha(src) for src,name in files}!r}
for name,digest in expected.items():
 with (r/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
print(json.dumps(expected))''')
        save(self.root / "deployment.json", self.each(one))

    def build_layout(self, denominator):
        case = self.root / f"d{denominator}"
        case.mkdir(exist_ok=True)
        if (case / "layout-ready.json").exists():
            return
        b = self.c["build"]
        options = {"generation": 1, "shard_count": 8, "distance": "Euclid", "vector_name": "",
                   "sample_denominator": denominator, "sample_seed": b["sample_seed"], "upper_seed": b["upper_seed"],
                   "upper_m": b["m"], "upper_ef_construct": b["ef_construct"], "efs": b["attachment_efs"],
                   "kmeans_iterations": 0, "kmeans_seed": 0, "topology_iterations": 0, "bundle_prefix": "import",
                   **{k: b[k] for k in ("upper_k", "upper_ef_search", "dynamic_ef_base", "dynamic_ef_factor")}}
        request = {"input": {"format_version": 1, "path": str(self.root / "data/train.f32le"),
                   "rows": 1000000, "dimension": 128, "sha256": read(self.root / "data/identity.json")["train.f32le"]},
                   "options": options}
        if (case / "request.json").exists():
            require(read(case / "request.json") == request, "retained layout request differs")
        else:
            save(case / "request.json", request)
        if (case / "layout").exists():
            m = read(case / "layout/build-manifest.json")
            require(m["options"] == options and m["raw_vectors_sha256"] == request["input"]["sha256"],
                    "retained layout does not match requested input/options")
            self.status("verifying_retained_layout", denominator=denominator)
            verification_log = case / f"verify-resume-{time.time_ns()}.log"
        else:
            self.status("building_layout", denominator=denominator)
            self.command([self.c["python"], self.root / "source/tools/orion_capacity_partition.py", "--request", case / "request.json",
                          "--output", case / "layout", "--rust-layout-binary", self.root / "bin/orion_build_layout",
                          "--whitebox-config", self.root / "whitebox.json"], case / "build.log")
            verification_log = case / "verify.log"
        self.command([self.root / "bin/orion_build_layout", "verify", "--bundle", case / "layout", "--replay"], verification_log)
        self.write_layout_summary(denominator, options)

    def write_layout_summary(self, denominator, options):
        case = self.root / f"d{denominator}"
        m = read(case / "layout/build-manifest.json")
        require(m["algorithm_version"] == "orion-layout-v5-access-probability-capacity", "wrong builder")
        require(len(m["upper_indices"]) == 1000000 // denominator, "wrong upper sample size")
        require(m["options"] == options, "builder changed requested options")
        artifact = read(case / "layout" / m["source_artifact_file"])
        save(case / "layout-ready.json", {"upper_nodes": len(m["upper_indices"]), "shard_counts": m["shard_counts"],
             "replication_factor": sum(m["shard_counts"]) / 1000000, "upper_graph_sha256": m["upper_graph_sha256"],
             "upper_artifact_bytes": (case / "layout" / m["source_artifact_file"]).stat().st_size,
             "upper_graph_json_bytes": len(json.dumps(artifact["upper_graph"], separators=(",", ":")).encode()),
             "build_manifest_sha256": sha(case / "layout/build-manifest.json")})

    def api(self, path, method="GET", body=None, host=None):
        self.check()
        req = urllib.request.Request(f'http://{host or self.c["hosts"][0]}:{self.c["port"]}{path}',
                                     data=None if body is None else json.dumps(body).encode(), method=method,
                                     headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=600) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            save(self.root / f"api-errors/{time.time_ns()}.json",
                 {"method": method, "path": path, "host": host, "status": error.code, "response": detail})
            error.add_note(detail)
            raise
        return result.get("result", result)

    def launch(self, host, args, directory, log, env):
        identity = self.remote(host, f'''import os,pathlib,subprocess,json
d=pathlib.Path({str(directory)!r});d.mkdir(parents=True,exist_ok=True)
env={{k:v for k,v in os.environ.items() if not k.startswith(('ORION_','QDRANT_'))}};env.update({env!r})
with (d/{log!r}).open('ab') as stream:
 p=subprocess.Popen({list(map(str,args))!r},env=env,cwd=d,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
f=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().rsplit(')',1)[1].split()
v={{'pid':p.pid,'start_ticks':int(f[19]),'directory':str(d)}}
(d/'process.json').write_text(json.dumps(v));print(json.dumps(v))''')
        return identity

    def stop_group(self, group):
        for host, ident in list(group.items()):
            self.remote(host, f'''import os,pathlib,signal,time,json
p=pathlib.Path('/proc/{ident['pid']}')
def alive():
 try:
  f=(p/'stat').read_text().rsplit(')',1)[1].split()
  return int(f[19])=={ident['start_ticks']} and f[0]!='Z'
 except FileNotFoundError:return False
if alive():
 assert str((p/'cwd').resolve())=={ident['directory']!r},'PID ownership changed'
 os.killpg({ident['pid']},signal.SIGTERM)
 for _ in range(100):
  if not alive():break
  time.sleep(.5)
 assert not alive(),'owned process did not exit; manual inspection required'
print('{{"stopped":true}}')''')
            group.pop(host)

    def start(self, denominator):
        self.stop_group(self.servers)
        self.status("starting_eight_host_cluster", denominator=denominator)
        directory = self.remote_root / f"d{denominator}" / "cluster"
        cfg = {"log_level": "WARN", "telemetry_disabled": True,
               "storage": {"storage_path": str(directory / "storage"), "snapshots_path": str(directory / "snapshots"),
                           "performance": {"max_search_threads": 48, "optimizer_cpu_budget": 8,
                                           "orion_navigation": {"enabled": self.c["navigation_balancing"]}}},
               "service": {"host": "0.0.0.0", "http_port": self.c["port"], "grpc_port": self.c["port"] + 1,
                           "max_workers": self.c["http_workers"], "hardware_reporting": False},
               "cluster": {"enabled": True, "p2p": {"port": self.c["port"] + 2, "connection_pool_size": 16}}}
        env = {"NO_PROXY": "*", "no_proxy": "*", "ORION_WHITEBOX_PRUNER": str(self.remote_root / "whitebox.json"),
               "QDRANT_NUM_CPUS": "48"}
        def one(host):
            self.remote(host, f"import pathlib,json;p=pathlib.Path({str(directory)!r});p.mkdir(parents=True,exist_ok=True);(p/'config.yaml').write_text({json.dumps(cfg)!r});print('{{}}')")
            args = ["taskset", "-c", self.c["server_cpus"], self.remote_root / "qdrant", "--config-path", directory / "config.yaml",
                    "--uri", f'http://{host}:{self.c["port"] + 2}']
            if host != self.c["hosts"][0]:
                args += ["--bootstrap", f'http://{self.c["hosts"][0]}:{self.c["port"] + 2}']
            self.servers[host] = self.launch(host, args, directory, "server.log", env)
        one(self.c["hosts"][0])
        time.sleep(2)
        for host in self.c["hosts"][1:]:
            one(host)
        deadline = time.monotonic() + 300
        stable_checks = 0
        while time.monotonic() < deadline:
            self.check()
            try:
                states = self.each(lambda h: self.api("/cluster", host=h))
                stable_checks = stable_checks + 1 if cluster_stable(states) else 0
                if stable_checks >= 3:
                    save(self.root / f"d{denominator}/startup-consensus-{time.time_ns()}.json", states)
                    expected = read(self.root / "binaries.json")["qdrant"]
                    for host, ident in self.servers.items():
                        proof = self.remote(host, f'''import pathlib,hashlib,os,json
p=pathlib.Path('/proc/{ident['pid']}');f=(p/'stat').read_text().rsplit(')',1)[1].split();assert int(f[19])=={ident['start_ticks']}
with (p/'exe').open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()=={expected!r}
affinity={{t.name:sorted(os.sched_getaffinity(int(t.name))) for t in (p/'task').iterdir()}}
assert all(v=={cpus(self.c['server_cpus'])!r} for v in affinity.values())
print(json.dumps({{'binary_sha256':{expected!r},'threads':affinity}}))''')
                        save(self.root / f"d{denominator}/process-{host}-{time.time_ns()}.json", proof)
                    return
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise RuntimeError("cluster did not converge")

    def graphs(self, denominator):
        directory = self.remote_root / f"d{denominator}/cluster/storage/collections/sift1m"
        return self.each(lambda host: self.remote(host, f'''import pathlib,json,hashlib
r=pathlib.Path({str(directory)!r});out={{}};configs=[]
for p in sorted(r.glob('*/segments/*/vector_index/*')):
 if p.is_file():
  with p.open('rb') as stream:out[str(p.relative_to(r))]=hashlib.file_digest(stream,'sha256').hexdigest()
  if p.name=='hnsw_config.json':configs.append(json.loads(p.read_text()))
assert len(configs)==1,'need one HNSW per host'
assert all(x['m']==32 and x['m0']==64 and x['ef_construct']==100 for x in configs),'wrong physical HNSW configuration'
print(json.dumps({{'files':out,'configs':configs}}))'''))

    def ready(self, denominator, generation):
        counts = read(self.root / f"d{denominator}/layout/build-manifest.json")["shard_counts"]
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            info = self.api("/collections/sift1m")
            places = self.each(lambda h: self.api("/collections/sift1m/cluster", host=h))
            good = info["status"] == "green" and info["optimizer_status"] == "ok"
            good &= info["points_count"] == sum(counts) and info["indexed_vectors_count"] == sum(counts)
            ids = []
            for p in places.values():
                local = p["local_shards"]
                good &= len(local) == 1 and not p["shard_transfers"]
                if len(local) == 1:
                    s = local[0]
                    ids.append(s["shard_id"])
                    good &= s["state"] == "Active" and s["points_count"] == counts[s["shard_id"]]
            good &= sorted(ids) == list(range(8))
            states = self.api("/collections/sift1m/orion")
            good &= len(states) == 8 and all(s and s["phase"] == "Ready" and
                       (s.get("active") or {}).get("generation") == generation for s in states)
            if good:
                config = info["config"]
                require(config["params"]["replication_factor"] == 1 and config["params"]["shard_number"] == 8,
                        "collection topology mismatch")
                save(self.root / f"d{denominator}/readiness-g{generation}-{time.time_ns()}.json",
                     {"info": info, "placement": places, "native_states": states})
                return
            time.sleep(3)
        raise RuntimeError("index/native generation readiness timeout")

    def control(self, operation, generation, **kwargs):
        # Reconcile a timeout/commit error by reading state; never blindly replay mutations.
        try:
            return self.api("/collections/sift1m/orion?timeout=600", "POST",
                            {"operation": operation, "generation": generation, **kwargs})
        except OSError:
            states = self.api("/collections/sift1m/orion")
            phases = {"begin": {"Collecting"}, "prepare": {"BuildingLayout", "Indexing"},
                      "import": {"Indexing"}, "validate": {"Finalizing"},
                      "activate": {"Finalizing", "Ready"}, "cleanup": {"Ready"}}
            applied = len(states) == 8 and all(s and s["phase"] in phases[operation] and
                      (s.get("pending_generation") == generation or (s.get("active") or {}).get("generation") == generation)
                      for s in states)
            if operation == "begin":
                applied &= all(s and s.get("files") == kwargs["files"] for s in states)
            if not applied:
                raise
            save(self.root / f"reconciled-{operation}-g{generation}-{time.time_ns()}.json", states)

    def install(self, denominator, bundle, initial=False):
        m = read(bundle / "build-manifest.json")
        im = read(bundle / m["import_manifest_file"])
        self.status("installing_bundle", denominator=denominator, generation=im["orion_generation"])
        if initial:
            body = {"vectors": {"size": 128, "distance": "Euclid", "on_disk": False},
                    "shard_number": 8, "replication_factor": 1, "write_consistency_factor": 1,
                    "hnsw_config": {"m": 32, "ef_construct": 100, "full_scan_threshold": 10,
                                    "max_indexing_threads": self.c["build"]["lower_indexing_threads"], "on_disk": False},
                    "optimizers_config": {"default_segment_number": 1, "max_segment_size": 2000000,
                                          "indexing_threshold": 0, "max_optimization_threads": 1},
                    "auto_shard_policy": {"type": "orion", "generation": im["orion_generation"],
                                          "artifact_sha256": im["orion_artifact_sha256"]}}
            self.api("/collections/sift1m?timeout=600", "PUT", body)
        gen = im["orion_generation"]
        names = ["build-manifest.json", m["source_artifact_file"], m["attachments_file"], m["import_manifest_file"],
                 im["orion_artifact_file"], im["vectors_file"], im["assignments_file"]]
        bindings = {name: {"size": (bundle / name).stat().st_size, "sha256": sha(bundle / name)} for name in names}
        self.control("begin", gen, files=bindings)
        destination = self.remote_root / f"d{denominator}/cluster/storage/collections/sift1m/orion_control/generation-{gen}"
        immutable = [im["vectors_file"], im["assignments_file"], m["attachments_file"]]
        def distribute(host):
            needed = names
            if gen > 1:
                needed = self.remote(host, f'''import pathlib,hashlib,json,os
d=pathlib.Path({str(destination)!r});source=d.parent/'generation-1';needed=[]
for name,binding in {bindings!r}.items():
 p=source/name
 if name in {immutable!r} and p.is_file() and p.stat().st_size==binding['size']:
  with p.open('rb') as f:matches=hashlib.file_digest(f,'sha256').hexdigest()==binding['sha256']
  if matches:
   p.chmod(0o444);os.link(p,d/name);continue
 needed.append(name)
print(json.dumps(needed))''')
            for name in needed:
                self.copy(host, bundle / name, destination / name)
            return self.remote(host, f'''import pathlib,hashlib,json
d=pathlib.Path({str(destination)!r})
for name,binding in {bindings!r}.items():
 p=d/name;assert p.stat().st_size==binding['size']
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==binding['sha256']
print('{{"verified":true}}')''')
        save(bundle.parent / f"transport-g{gen}.json", self.each(distribute))
        self.control("prepare", gen)
        self.control("import", gen)
        if initial:
            self.api("/collections/sift1m", "PATCH", {"optimizers_config": {"indexing_threshold": 1}})
        self.status("indexing_lower_shards", denominator=denominator, generation=gen)
        deadline = time.monotonic() + 3600
        while True:
            info = self.api("/collections/sift1m")
            if (info["status"] == "green" and info["optimizer_status"] == "ok" and
                    info["points_count"] == info["indexed_vectors_count"] == sum(m["shard_counts"])):
                break
            require(time.monotonic() < deadline, "lower indexing timeout")
            time.sleep(3)
        self.control("validate", gen)
        self.control("activate", gen)
        self.control("cleanup", gen)
        self.ready(denominator, gen)
        save(self.root / f"d{denominator}/active.json", {"generation": im["orion_generation"], "bundle": str(bundle),
             "search": {k: m["options"][k] for k in candidates(self.c)[0]}})

    def profile(self, denominator, params):
        case = self.root / f"d{denominator}"
        active = read(case / "active.json")
        if active["search"] == params:
            return
        generation = active["generation"] + 1
        bundle = case / "profiles" / f"g{generation}"
        args = [self.root / "bin/orion_build_layout", "profile", "--source", case / "layout",
                "--output", bundle, "--generation", generation]
        for key, value in params.items():
            args += ["--" + key.replace("_", "-"), value]
        self.command(args, case / f"profile-g{generation}.log")
        base, derived = read(case / "layout/build-manifest.json"), read(bundle / "build-manifest.json")
        require(all(base[k] == derived[k] for k in ("upper_graph_sha256", "upper_identity_sha256", "attachments_sha256")),
                "runtime profile changed graph or assignments")
        imported = read(bundle / derived["import_manifest_file"])
        for name in (imported["vectors_file"], imported["assignments_file"], derived["attachments_file"]):
            source, copy = case / "layout" / name, bundle / name
            require(sha(source) == sha(copy), "profile changed immutable data")
            source.chmod(0o444)
            copy.unlink()
            copy.hardlink_to(source)
        self.install(denominator, bundle)
        require(self.graphs(denominator) == read(case / "graphs.json"), "profile rebuilt lower HNSW")

    def spec(self, denominator, subset):
        return {"collection": "sift1m", "host": self.c["hosts"][0], "port": self.c["port"], "rows": 1000000,
                "queries_path": str(self.root / "data/queries.npz"), "batch_size": self.c["load"]["batch_size"],
                "query_ids": read(self.root / "data/split.json")[subset]}

    def audit(self, denominator, subset, name):
        import orion_navigation_size_client as client
        dest = self.root / f"d{denominator}" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        require(not dest.with_suffix(".json").exists(), "audit output already exists")
        return asyncio.run(client.audit(self.spec(denominator, subset), dest))

    def measure(self, denominator, name, subset, workers, lanes, warmup, seconds):
        import numpy as np
        out = self.root / f"d{denominator}/measurements" / name
        out.mkdir(parents=True, exist_ok=False)
        remote = self.remote_root / f"d{denominator}/measurements" / name
        clusters = self.each(lambda h: self.api("/cluster", host=h))
        spec = {**self.spec(denominator, subset), "workers_per_host": workers, "lanes": lanes,
                "warmup_seconds": warmup, "seconds": seconds, "seed": self.c["load"]["seed"],
                "client_cpus": cpus(self.c["client_cpus"])}
        save(out / "spec.json", spec)
        self.status("measuring", denominator=denominator, window=name, workers=workers, lanes=lanes)
        try:
            for i, host in enumerate(self.c["hosts"]):
                s = {**spec, "host": host, "host_index": i, "directory": str(remote),
                     "queries_path": str(self.remote_root / "queries.npz")}
                self.remote(host, f"import pathlib,json;p=pathlib.Path({str(remote)!r});p.mkdir(parents=True,exist_ok=False);(p/'spec.json').write_text({json.dumps(s)!r});print('{{}}')")
                self.clients[host] = self.launch(host, [self.c["remote_python"], self.remote_root / "client.py", "--spec", remote / "spec.json"],
                    remote, "client.log", {"PYTHONPATH": self.c["remote_python_path"], "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
            deadline = time.monotonic() + 180
            while True:
                self.check()
                ready = self.each(lambda h: self.remote(h, f"import pathlib,json;print(json.dumps(len(list(pathlib.Path({str(remote)!r}).glob('w*.ready')))))"))
                if all(n == workers for n in ready.values()):
                    break
                require(time.monotonic() < deadline, "client readiness timeout")
                time.sleep(1)
            clocks = self.clock_offsets()
            origin = time.monotonic() + 15
            save(out / "start.json", {"controller_start_monotonic": origin, "clocks": clocks})
            starts = {h: origin + clocks[h]["offset"] for h in self.c["hosts"]}
            def release(host):
                value = {"start_monotonic": starts[host]}
                return self.remote(host, f"import pathlib,json;p=pathlib.Path({str(remote)!r});(p/'start.json').write_text({json.dumps(value)!r});print('{{}}')")
            self.each(release)
            require(time.monotonic() < origin - 2, "barrier release took too long")
            def samples(host):
                ident = self.servers[host]
                return self.remote(host, f'''import pathlib,time,os,json
p=pathlib.Path('/proc/{ident['pid']}');rows=[]
for i in range({seconds}+1):
 at={starts[host]+warmup!r}+i
 time.sleep(max(0,at-time.monotonic()))
 f=(p/'stat').read_text().rsplit(')',1)[1].split();assert int(f[19])=={ident['start_ticks']}
 a=sorted(os.sched_getaffinity({ident['pid']}));assert a=={cpus(self.c['server_cpus'])!r}
 status=(p/'status').read_text();rss=next(int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:'))
 rows.append({{'epoch':time.time(),'monotonic':time.monotonic(),'cpu':(int(f[11])+int(f[12]))/os.sysconf('SC_CLK_TCK'),'rss':rss,'major_faults':int(f[9]),'io':(p/'io').read_text()}})
print(json.dumps(rows))''', timeout=warmup + seconds + 90)
            with ThreadPoolExecutor(8) as pool:
                futures = {host: pool.submit(samples, host) for host in self.c["hosts"]}
                while not all(f.done() for f in futures.values()):
                    self.check()
                    time.sleep(1)
                cpu = {host: f.result() for host, f in futures.items()}
            save(out / "server-samples.json", cpu)
            require(all(len(rows) == seconds + 1 and
                        max(abs(r["monotonic"] - starts[h] - warmup - i) for i, r in enumerate(rows)) < .5
                        for h, rows in cpu.items()), "server sampling missed the measurement boundary")
            deadline = time.monotonic() + 90
            while True:
                done = self.each(lambda h: self.remote(h, f"import pathlib,json;print(json.dumps(len(list(pathlib.Path({str(remote)!r}).glob('w*.json')))))"))
                if all(n == workers for n in done.values()):
                    break
                self.check()
                require(time.monotonic() < deadline, "load worker did not finish")
                time.sleep(1)
            for host in self.c["hosts"]:
                # Only this window's worker data; no broad wildcard outside its directory.
                subprocess.run(["scp", "-q", f"{host}:{remote}/w*.npz", f"{host}:{remote}/w*.json", str(out)], check=True, timeout=180)
            after = self.each(lambda h: self.api("/cluster", host=h))
            require(all(clusters[h]["raft_info"]["term"] == after[h]["raft_info"]["term"] and
                        after[h]["raft_info"]["pending_operations"] == 0 for h in after), "consensus changed")
            save(out / "consensus.json", {"before": clusters, "after": after})
            result = summarize_window(out, spec)
            result["server_cpu_cores"] = {h: (rows[-1]["cpu"] - rows[0]["cpu"]) /
                                         (rows[-1]["monotonic"] - rows[0]["monotonic"]) for h, rows in cpu.items()}
            result["server_rss_bytes"] = {h: max(r["rss"] for r in rows) for h, rows in cpu.items()}
            result.update(saturation_evidence(result["server_cpu_cores"], self.c))
            result["server_cpu_fraction_per_second"] = {
                h: [(b["cpu"]-a["cpu"])/(b["monotonic"]-a["monotonic"])/48 for a,b in zip(rows,rows[1:])]
                for h, rows in cpu.items()}
            result["server_cpu_ms_per_query"] = 1000 * sum(result["server_cpu_cores"].values()) / result["qps"]
            save(out / "summary.json", result)
            return result
        finally:
            self.stop_group(self.clients)

    def archive(self, denominator, label):
        self.stop_group(self.servers)
        folder = self.root / f"d{denominator}/archives" / label
        folder.mkdir(parents=True, exist_ok=False)
        source = self.remote_root / f"d{denominator}/cluster"
        for i, host in enumerate(self.c["hosts"]):
            target = folder / f"host{i}.tar"
            with target.open("xb") as stream:
                subprocess.run(SSH + [host, shlex.join(["tar", "-C", str(source), "-cf", "-", "."])], stdout=stream, check=True, timeout=600)
            # Verify all archived graph bytes against the fingerprints taken while serving.
            import tarfile
            expected = read(self.root / f"d{denominator}/graphs.json")[host]["files"]
            found = {}
            prefix = "storage/collections/sift1m/"
            with tarfile.open(target, "r:") as archive:
                for member in archive:
                    name = member.name.removeprefix("./")
                    rel = name.removeprefix(prefix)
                    if name.startswith(prefix) and rel in expected:
                        with archive.extractfile(member) as stream:
                            found[rel] = hashlib.file_digest(stream, "sha256").hexdigest()
            require(found == expected, "archive graph verification failed")
        save(folder / "sha256.json", {p.name: sha(p) for p in folder.glob("*.tar")})

    def prepare(self):
        self.freeze_and_compile()
        self.export_data()
        self.deploy_files()
        for d in self.c["denominators"]:
            self.build_layout(d)
            case = self.root / f"d{d}"
            if (case / "prepared.json").exists():
                continue
            self.start(d)
            self.install(d, case / "layout", initial=True)
            save(case / "graphs.json", self.graphs(d))
            self.archive(d, "prepared")
            save(case / "prepared.json", {"complete": True})
        # Same sampler seed produces nested prefixes, but verify the actual native output.
        samples = [read(self.root / f"d{d}/layout/build-manifest.json")["upper_indices"] for d in self.c["denominators"]]
        require(all(small == large[:len(small)] for large, small in zip(samples, samples[1:])), "samples are not nested")
        save(self.root / "PREPARED.json", {"ratios": self.c["denominators"], "nested_samples": True})

    def screen_saturation(self, d, label, minimum_lanes=1):
        """Increase only real ANN traffic; a QPS plateau alone cannot qualify a load."""
        load = self.c["load"]
        screens = []
        grid = sorted({v for v in load["lanes_values"] if v >= minimum_lanes} | {minimum_lanes})
        while grid:
            lanes = grid.pop(0)
            if lanes > load["maximum_lanes"]:
                break
            result = self.measure(d, f"{label}-{lanes}", "calibration", load["workers_per_host"], lanes,
                                  load["screen_warmup_seconds"], load["screen_seconds"])
            screens.append({"lanes": lanes, **result})
            if not grid and max(r["busiest_service_cpu_fraction"] for r in screens) < load["screen_cpu_fraction_target"]:
                grid.append(lanes * 2)
        eligible = [r for r in screens if r["cpu_saturated"]]
        save(self.root / f"d{d}/{label}-evidence.json", screens)
        require(eligible, "load screen never exceeded 95% service CPU; no formal QPS may be accepted")
        target = [r for r in eligible if r["busiest_service_cpu_fraction"] >= load["screen_cpu_fraction_target"]]
        eligible = target or eligible
        best = max(r["qps"] for r in eligible)
        chosen = min((r for r in eligible if r["qps"] >= .98 * best), key=lambda r: r["lanes"])
        return chosen, screens

    def calibrate(self, d):
        case = self.root / f"d{d}"
        if (case / "frozen.json").exists():
            return
        self.start(d)
        self.ready(d, read(case / "active.json")["generation"])
        require(self.graphs(d) == read(case / "graphs.json"), "prepared indexes changed")
        c, load = self.c["calibration"], self.c["load"]
        rows = []
        for i, params in enumerate(candidates(self.c)):
            file = case / f"calibration/candidate-{i}.json"
            if file.exists():
                rows.append(read(file))
                continue
            self.profile(d, params)
            row = {"index": i, "search": params, **self.audit(d, "calibration", f"calibration/audit-{i}")}
            if row["recall"] >= c["selection_floor"]:
                pilot = self.measure(d, f"pilot-{i}", "calibration", c["pilot_workers_per_host"], c["pilot_lanes"],
                                     c["pilot_warmup_seconds"], c["pilot_seconds"])
                row["qps"] = pilot["qps"]
            save(file, row)
            rows.append(row)
        if not any(r.get("qps", 0) > 0 and r["recall"] >= c["selection_floor"] for r in rows):
            save(case / "FAILED_CALIBRATION.json", {"selection_floor": c["selection_floor"],
                 "best_observed_recall": max(r["recall"] for r in rows), "grid_exhausted": True,
                 "does_not_prove_target_is_unattainable": True})
            self.stop_group(self.servers)
            return
        selected = choose_candidate(rows, self.c)
        self.profile(d, selected["search"])
        chosen, screens = self.screen_saturation(d, "load")
        frozen = {"search": selected["search"], "calibration_recall": selected["recall"], "lanes": chosen["lanes"],
                  "workers_per_host": load["workers_per_host"], "load_screen": screens, "state": "frozen_before_test"}
        save(case / "selection.json", frozen)
        audit = self.audit(d, "test", "test-audit")
        save(case / "qualification.json", {**audit, "qualified": audit["recall"] >= .95})
        if audit["recall"] < .95:
            save(case / "FAILED_RECALL.json", {"actual_recall": audit["recall"], "target": .95,
                                              "search": selected["search"], "no_test_driven_retuning": True})
            self.stop_group(self.servers)
            return
        frozen["test_recall"] = audit["recall"]
        save(case / "frozen.json", frozen)
        self.stop_group(self.servers)

    def diagnose(self, d):
        import numpy as np
        case = self.root / f"d{d}"
        if (case / "route-summary.json").exists():
            return
        active = read(case / "active.json")
        bundle = Path(active["bundle"])
        m = read(bundle / "build-manifest.json")
        im = read(bundle / m["import_manifest_file"])
        with np.load(self.root / "data/queries.npz") as data:
            np.asarray(data["queries"], dtype="<f4").tofile(case / "queries.f32le")
        env = {**self.env, "ORION_WHITEBOX_PRUNER": str(self.root / "whitebox.json")}
        self.command([self.root / "bin/native_whitebox_diagnostic", bundle / im["orion_artifact_file"], case / "queries.f32le", 128,
                      case / "routes.jsonl"], case / "routes.log", env=env)
        ids = set(read(self.root / "data/split.json")["test"])
        fanout, entries, efs = [], [], []
        shard_visits = [0] * 8
        with (case / "routes.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["query_index"] in ids:
                    fanout.append(len(row["after"]))
                    entries.append(sum(len(t["entries"]) for t in row["after"]))
                    efs.append(sum(t["ef"] for t in row["after"]))
                    for target in row["after"]:
                        shard_visits[target["shard"]] += 1
        require(len(fanout) == len(ids), "missing native routes")
        save(case / "route-summary.json", {"mean_fanout": statistics.mean(fanout), "mean_entries": statistics.mean(entries),
             "mean_ef_sum": statistics.mean(efs), "shard_visits": shard_visits, "offline_diagnostic_only": True})

    def benchmark(self):
        require((self.root / "PREPARED.json").exists(), "prepare must complete first")
        self.freeze_and_compile()
        self.export_data()
        deployment = read(self.root / "deployment.json")
        for host in self.c["hosts"]:
            self.remote(host, f'''import pathlib,hashlib,json
r=pathlib.Path({str(self.remote_root)!r})
for name,digest in {deployment[host]!r}.items():
 with (r/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,'deployment changed'
print('{{"verified":true}}')''')
        for d in self.c["denominators"]:
            case = self.root / f"d{d}"
            require(sha(case / "layout/build-manifest.json") == read(case / "layout-ready.json")["build_manifest_sha256"],
                    "prepared layout manifest changed")
        for d in self.c["denominators"]:
            if not any((self.root / f"d{d}" / name).exists() for name in ("FAILED_RECALL.json", "FAILED_CALIBRATION.json")):
                self.calibrate(d)
        for repeat, order in enumerate(make_plan(self.c)["formal_order"]):
            for d in order:
                case = self.root / f"d{d}"
                if not (case / "frozen.json").exists():
                    continue
                if (case / f"formal-{repeat}.json").exists():
                    require(qualifies_formal(read(case / f"formal-{repeat}.json"), self.c), "saved formal window violates acceptance")
                    continue
                frozen = read(case / "frozen.json")
                self.start(d)
                self.ready(d, read(case / "active.json")["generation"])
                require(read(case / "active.json")["search"] == frozen["search"], "frozen search parameters changed")
                require(self.graphs(d) == read(case / "graphs.json"), "graphs changed before measurement")
                audit = self.audit(d, "test", f"formal-{repeat}-audit")
                require(audit["recall"] >= .95, "formal recall below target")
                load = self.c["load"]
                lanes = frozen["lanes"]
                attempt = 0
                while True:
                    result = self.measure(d, f"formal-{repeat}-attempt{attempt}", "test", frozen["workers_per_host"], lanes,
                                          load["warmup_seconds"], load["seconds"])
                    if result["cpu_saturated"]:
                        break
                    save(case / f"rejected-formal-{repeat}-attempt{attempt}.json",
                         {**result, "reason": "busiest service CPU did not strictly exceed 95%"})
                    chosen, _ = self.screen_saturation(d, f"rescreen-{repeat}-{attempt}", lanes * 2)
                    lanes = chosen["lanes"]
                    attempt += 1
                require(qualifies_formal(result, self.c), "formal acceptance requires recall >=0.95 and busiest service CPU >95%")
                require(result["recall_uniform_queries"] >= .95, "timed recall below target")
                require(self.graphs(d) == read(case / "graphs.json"), "graphs changed during measurement")
                result["audit_recall"] = audit["recall"]
                result["lanes"] = lanes
                result["workers_per_host"] = frozen["workers_per_host"]
                save(case / f"formal-{repeat}.json", result)
                self.stop_group(self.servers)
        for d in self.c["denominators"]:
            case = self.root / f"d{d}"
            if (case / "frozen.json").exists():
                self.diagnose(d)
                if not (case / "archives/final/sha256.json").exists():
                    self.archive(d, "final")
        summarize_experiment(self.c)


def summarize_window(folder, spec):
    import numpy as np
    counts = np.zeros(10000, dtype=np.int64)
    hits = np.zeros_like(counts)
    latency, client_cpu, per_second = [], [], np.zeros(spec["seconds"], dtype=np.int64)
    pending_start = pending_end = 0
    drain = 0.0
    end = spec["warmup_seconds"] + spec["seconds"]
    for w in range(8 * spec["workers_per_host"]):
        meta = read(folder / f"w{w}.json")
        require(not meta["errors"] and meta["affinity"] == spec["client_cpus"], "invalid worker errors/affinity")
        a, b = meta["cpu_marks"]
        client_cpu.append((b[1] - a[1]) / (b[0] - a[0]))
        with np.load(folder / f"w{w}.npz") as data:
            batches = data["batches"]
            require(len(batches) > 0 and batches[:, 0].min() < .5, "client missed start")
            mask = (batches[:, 1] >= spec["warmup_seconds"]) & (batches[:, 1] < end)
            selected = batches[mask]
            require((batches[:, 1] >= batches[:, 0]).all(), "negative latency")
            latency.extend((selected[:, 1] - selected[:, 0]) * 1000)
            np.add.at(per_second, (selected[:, 1] - spec["warmup_seconds"]).astype(int), selected[:, 2].astype(int))
            qids = data["query_ids"]
            require(sorted(qids.tolist()) == sorted(spec["query_ids"]), "wrong query subset")
            require(data["counts"].sum() == selected[:, 2].sum(), "completion accounting mismatch")
            require((data["counts"] >= 0).all() and (data["hits"] >= 0).all() and
                    (data["hits"] <= 10 * data["counts"]).all(), "invalid recall accounting")
            counts[qids] += data["counts"]
            hits[qids] += data["hits"]
            pending_start += int(((batches[:, 0] < spec["warmup_seconds"]) & (batches[:, 1] >= spec["warmup_seconds"])).sum())
            pending_end += int(((batches[:, 0] < end) & (batches[:, 1] >= end)).sum())
            drain = max(drain, float(batches[:, 1].max()) - end)
    require((counts[spec["query_ids"]] > 0).all(), "not every query was measured")
    require(counts.sum() > 0 and counts.sum() == per_second.sum(), "empty or inconsistent window")
    np.savez_compressed(folder / "query-counts.npz", counts=counts, hits=hits, per_second=per_second)
    return {"qps": int(counts.sum()) / spec["seconds"], "successful_queries": int(counts.sum()),
            "recall_uniform_queries": float(np.mean(hits[spec["query_ids"]] / (10 * counts[spec["query_ids"]]))),
            "recall_request_weighted": float(hits.sum() / (10 * counts.sum())), "errors": 0,
            "batch_latency_p50_ms": float(np.percentile(latency, 50)), "batch_latency_p95_ms": float(np.percentile(latency, 95)),
            "batch_latency_p99_ms": float(np.percentile(latency, 99)), "client_cpu_cores": sum(client_cpu),
            "pending_batches_start": pending_start, "pending_batches_end": pending_end, "drain_seconds": drain}


def summarize_experiment(config):
    root = Path(config["run_root"])
    rows = []
    for d in config["denominators"]:
        case = root / f"d{d}"
        runs = [read(p) for p in sorted(case.glob("formal-*.json")) if "audit" not in p.name]
        evidence = all((case / name).is_file() for name in
                       ("frozen.json", "graphs.json", "route-summary.json", "archives/final/sha256.json"))
        valid = (evidence and len(runs) == config["load"]["repeats"] and
                 all(qualifies_formal(r, config) for r in runs))
        row = {"ratio": f"1/{d}", "upper_nodes": 1000000 // d, "state": "complete" if valid else "not_qualified_or_incomplete",
               "repeats": len(runs), "qps_median": statistics.median(r["qps"] for r in runs) if valid else None,
               "qps_min": min((r["qps"] for r in runs), default=None), "qps_max": max((r["qps"] for r in runs), default=None),
               "recall_min": min((r["recall_uniform_queries"] for r in runs), default=None),
               "recall_max": max((r["recall_uniform_queries"] for r in runs), default=None),
               "all_windows_cpu_saturated": all(qualifies_formal(r, config) for r in runs) if runs else None,
               "busiest_cpu_fraction_min": min((r["busiest_service_cpu_fraction"] for r in runs), default=None),
               "formal_lanes": [r.get("lanes") for r in runs],
               "cpu_ms_per_query_median": statistics.median(r["server_cpu_ms_per_query"] for r in runs) if runs else None,
               "batch_p99_ms_median": statistics.median(r["batch_latency_p99_ms"] for r in runs) if runs else None}
        if (case / "layout-ready.json").exists():
            row.update(read(case / "layout-ready.json"))
        if (case / "frozen.json").exists():
            f = read(case / "frozen.json")
            row.update(f["search"], lanes=f["lanes"], workers_per_host=f["workers_per_host"])
        if (case / "route-summary.json").exists():
            row.update(read(case / "route-summary.json"))
        rows.append(row)
    save(root / "summary.json", rows)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (root / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = all(row["state"] == "complete" for row in rows)
    save(root / ("COMPLETED.json" if complete else "INCOMPLETE.json"), {"all_ratios_qualified": complete})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("plan", "build", "prepare", "benchmark", "all", "summarize"), default="plan")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true", help="explicitly perform the selected stage; never implicit")
    args = parser.parse_args(argv)
    config = validate(read(args.config))
    if args.action == "plan" or not args.execute:
        print(json.dumps(make_plan(config), indent=2, ensure_ascii=False))
        return
    if args.action == "summarize":
        summarize_experiment(config)
        return
    experiment = Experiment(config)
    # Imports for executed stages use the configured dependency directory, never during plan.
    sys.path.insert(0, config["python_path"])
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    with experiment.lease():
        experiment.preflight(require_direct_network=args.action != "build")
        if args.action == "build":
            experiment.freeze_and_compile()
            experiment.export_data()
            save(experiment.root / "whitebox.json", config["whitebox"])
            for d in config["denominators"]:
                experiment.build_layout(d)
        if args.action in ("prepare", "all"):
            experiment.prepare()
        if args.action in ("benchmark", "all"):
            experiment.benchmark()


if __name__ == "__main__":
    main()
