#!/usr/bin/env python3
"""Single-machine Orion/KMeans Oracle and Real recall experiment.

Without --execute this only validates small metadata and prints the plan.
See shard_locality.md for the metric definitions and execution contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from contextlib import ExitStack
from functools import reduce

# Keep preparation and plotting from silently occupying every core.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np


REPO = Path(__file__).resolve().parents[1]
METRIC_SCHEMA = "gt-coverage-paired-queries-v2"


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_path(base, name):
    path = Path(name).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_plan(path):
    """Read only small configuration/manifest files; no graph or data scans."""
    path = Path(path).resolve()
    config = read_json(path)
    if config.get("format_version") != 1:
        raise ValueError("config format_version must be 1")
    allowed = {"format_version", "orion_manifest", "queries_npz", "query_key", "truth_key",
               "query_limit", "top_k", "distance", "datatype", "lower", "kmeans", "orion_profiles", "real_profile", "execution", "collect_ann_diagnostic"}
    if set(config) - allowed:
        raise ValueError(f"unknown configuration keys: {set(config) - allowed}")
    for key in ("orion_manifest", "queries_npz"):
        config[key] = str(resolve_path(path.parent, config[key]))
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    manifest = read_json(config["orion_manifest"])
    if manifest.get("format_version") != 1 or "orion_artifact_file" not in manifest:
        raise ValueError("expected a native Orion import.manifest.json (v1)")
    for key in ("dimension", "point_count", "shard_count", "total_point_copies"):
        positive(manifest[key], key)
    base = Path(config["orion_manifest"]).parent
    files = {key: resolve_path(base, manifest[key + "_file"])
             for key in ("vectors", "assignments", "orion_artifact")}
    for file in files.values():
        if not file.is_file():
            raise FileNotFoundError(file)
    encoding = manifest.get("vector_encoding", "f32le")
    if encoding not in ("f32le", "u8"):
        raise ValueError("only f32le and u8 vectors are supported")
    expected_size = manifest["point_count"] * manifest["dimension"] * (4 if encoding == "f32le" else 1)
    if files["vectors"].stat().st_size != expected_size:
        raise ValueError("base vector size does not match manifest")
    if config["distance"] not in ("Euclid", "Cosine"):
        raise ValueError("this KMeans comparison supports Euclid and Cosine")
    if config.get("datatype", "float32") not in ("float32", "uint8"):
        raise ValueError("only float32 and uint8 local indexes are supported")
    if config.get("datatype", "float32") == "uint8" and config["distance"] != "Euclid":
        raise ValueError("uint8 comparison is supported only for Euclid")
    top = positive(config.get("top_k", 10), "top_k")
    if top > 20:
        raise ValueError("exact overlap-aware Oracle requires top_k <= 20 (exponential in k)")
    if config.get("query_limit") is not None:
        positive(config["query_limit"], "query_limit")
    if type(config.get("collect_ann_diagnostic", False)) is not bool:
        raise ValueError("collect_ann_diagnostic must be boolean")
    lower = config["lower"]
    if (not {"m", "ef_construct", "kmeans_ef"} <= set(lower) or
            set(lower) - {"m", "ef_construct", "seed", "kmeans_ef"}):
        raise ValueError("lower must specify m, ef_construct, kmeans_ef; optional diagnostic build seed")
    for key, value in lower.items():
        positive(value, f"lower.{key}")
    km = config["kmeans"]
    if km != {"reuse_existing_index": True}:
        if set(km) != {"train_size", "iterations", "seed", "n_init"}:
            raise ValueError("kmeans must request existing indexes or specify diagnostic training parameters")
        for key, value in km.items():
            positive(value, f"kmeans.{key}")
        if not manifest["shard_count"] <= km["train_size"] <= manifest["point_count"]:
            raise ValueError("KMeans train_size must be between shard_count and point_count")
    profiles = config["orion_profiles"]
    if not profiles:
        raise ValueError("at least one Orion profile is required")
    seen = set()
    for p in profiles:
        if set(p) != {"name", "upper_k", "upper_ef_search", "dynamic_ef_base", "dynamic_ef_factor", "whitebox"}:
            raise ValueError("invalid Orion profile fields")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", p["name"]) or p["name"] in seen:
            raise ValueError("profile names must be unique filename-safe identifiers")
        seen.add(p["name"])
        for key in ("upper_k", "upper_ef_search", "dynamic_ef_base", "dynamic_ef_factor"):
            positive(p[key], key)
        w = p["whitebox"]
        if set(w) != {"prefix_k", "max_prefix_entries", "max_distance_gap"}:
            raise ValueError("whitebox must specify prefix_k, max_prefix_entries, max_distance_gap")
        positive(w["prefix_k"], "prefix_k")
        support = w["max_prefix_entries"]
        if isinstance(support, bool) or not isinstance(support, int) or not 0 <= support < w["prefix_k"]:
            raise ValueError("invalid whitebox support")
        if not math.isfinite(w["max_distance_gap"]) or w["max_distance_gap"] < 0:
            raise ValueError("invalid whitebox distance gap")
        if not w["prefix_k"] <= p["upper_k"] <= p["upper_ef_search"]:
            raise ValueError("requires prefix_k <= upper_k <= upper_ef_search")
    if config.get("real_profile", profiles[0]["name"]) not in seen:
        raise ValueError("real_profile must name one Orion profile")
    execution = config.get("execution", {"mode": "single_machine"})
    if execution["mode"] == "eight_hosts":
        hosts = execution["hosts"]
        if (len(hosts) != 8 or len(set(hosts)) != 8 or manifest["shard_count"] != 8 or
                execution.get("shards_per_host") != 1 or
                any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]*", h) for h in hosts)):
            raise ValueError("eight_hosts requires 8 distinct hosts and exactly one shard per host")
        if len(profiles) != 1:
            raise ValueError("formal comparison requires one fixed Orion configuration")
    elif execution["mode"] != "single_machine":
        raise ValueError("unsupported execution mode")
    return config, manifest, files


def exact_oracle(shard_masks, top_k):
    """Max distinct GT IDs covered by <= n shards, with exact set-union DP.

    A mask has one bit per GT neighbor. Dominated/equal masks can be discarded:
    replacing a shard by a superset cannot hurt an at-most-n optimum.
    min_cost[covered_mask] stores the fewest shards realizing that union.
    """
    shards = len(shard_masks)
    masks = sorted(set(map(int, shard_masks)) - {0}, key=int.bit_count, reverse=True)
    maximal = []
    for mask in masks:
        if not any(mask | other == other for other in maximal):
            maximal.append(mask)
    costs = {0: 0}
    for mask in maximal:
        for covered, count in list(costs.items()):
            union = covered | mask
            costs[union] = min(costs.get(union, shards + 1), count + 1)
    best = np.zeros(shards + 1, dtype=np.float64)
    for mask, count in costs.items():
        best[count] = max(best[count], mask.bit_count() / top_k)
    return np.maximum.accumulate(best)


def oracle_matrix(truth, memberships, shards):
    oracle = []
    masks_by_query = []
    for row in truth:
        masks = [0] * shards
        for bit, point in enumerate(row):
            for shard in memberships[int(point)]:
                masks[shard] |= 1 << bit
        if (sum(1 << bit for bit in range(len(row))) !=
                reduce(int.__or__, masks, 0)):
            raise ValueError("ground-truth neighbor is absent from all shards")
        masks_by_query.append(masks)
        oracle.append(exact_oracle(masks, len(row)))
    return np.asarray(oracle), masks_by_query


def coverage(masks, targets, top):
    covered = 0
    for target in targets:
        covered |= masks[target["shard_id"]]
    return covered.bit_count() / top


def merge_topk(previous, points, top, distance="Euclid"):
    """Larger-is-better scores; native global ID tie-break; source-ID dedup."""
    scores = dict(previous)
    for point in points:
        point_id, score = int(point["id"]), float(point["score"])
        if not math.isfinite(score):
            raise ValueError("non-finite search score")
        scores[point_id] = max(scores.get(point_id, -math.inf), score)
    # ScoredPointTies: SmallBetter (Euclid) sorts ascending IDs on ties;
    # LargeBetter (Cosine) sorts descending IDs on ties.
    return sorted(scores.items(), key=lambda pair: (-pair[1],
                  pair[0] if distance == "Euclid" else -pair[0]))[:top]


def load_queries(config, manifest, output):
    with np.load(config["queries_npz"], allow_pickle=False) as source:
        query = source[config.get("query_key", "queries")]
        truth = source[config.get("truth_key", "truth")]
    top = config.get("top_k", 10)
    if query.ndim != 2 or truth.ndim != 2 or len(query) != len(truth) or not len(query):
        raise ValueError("queries/truth must be nonempty aligned matrices")
    if query.shape[1] != manifest["dimension"] or truth.shape[1] < top:
        raise ValueError("query dimension or truth width mismatch")
    if truth.dtype.kind not in "iu":
        raise ValueError("truth IDs must be integers")
    limit = config.get("query_limit") or len(query)
    if limit > len(query):
        raise ValueError("query_limit exceeds available query rows")
    query = np.asarray(query[:limit], dtype="<f4")
    truth = np.asarray(truth[:limit, :top], dtype=np.int64)
    if not np.isfinite(query).all() or (truth < 0).any() or (truth >= manifest["point_count"]).any():
        raise ValueError("non-finite query or truth ID outside full dataset")
    if (np.diff(np.sort(truth, axis=1), axis=1) == 0).any():
        raise ValueError("duplicate ground-truth IDs within a query")
    query.tofile(output / "queries.f32le")
    np.save(output / "truth.npy", truth, allow_pickle=False)
    return query, truth


def split_orion(manifest, assignments, truth, output):
    """One sequential assignment scan. Retain only GT memberships in Python RAM."""
    wanted = set(map(int, truth.ravel()))
    membership = {}
    shards = manifest["shard_count"]
    counts = [0] * shards
    digest = hashlib.sha256()
    with ExitStack() as stack:
        sinks = [stack.enter_context((output / f"ids-{s}.u64le").open("xb")) for s in range(shards)]
        with assignments.open("rb") as source:
            rows = 0
            for index, line in enumerate(source):
                digest.update(line)
                row = json.loads(line)
                point, owners = row["id"], row["shards"]
                if isinstance(point, bool) or not isinstance(point, int) or point != index:
                    raise ValueError("requires canonical contiguous source IDs equal to input row numbers")
                if not owners or len(set(owners)) != len(owners) or any(
                        isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < shards for s in owners):
                    raise ValueError(f"invalid memberships for {point}")
                for s in owners:
                    sinks[s].write(point.to_bytes(8, "little"))
                    counts[s] += 1
                if point in wanted:
                    membership[point] = owners
                rows += 1
    if rows != manifest["point_count"] or sum(counts) != manifest["total_point_copies"]:
        raise ValueError("assignment row/copy counts do not match manifest")
    if digest.hexdigest() != manifest["assignments_sha256"]:
        raise ValueError("assignment checksum mismatch")
    if set(membership) != wanted:
        raise ValueError("missing ground truth IDs")
    return membership, counts


def normalize_vectors(vectors, distance):
    result = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("non-finite base vector")
    if distance == "Cosine":
        norms = np.sqrt(np.sum(result * result, axis=1, dtype=np.float32))
        result /= np.where(norms >= 1e-12, norms, 1)[:, None]
    return result


def build_kmeans(config, manifest, vectors, truth, output):
    """Standard sampled Lloyd KMeans; never trains on test queries or truth."""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    shards, count, dimension = (manifest[k] for k in ("shard_count", "point_count", "dimension"))
    dtype = "<f4" if manifest.get("vector_encoding", "f32le") == "f32le" else "u1"
    source = np.memmap(vectors, dtype=dtype, mode="r", shape=(count, dimension))
    opts = config["kmeans"]
    rng = np.random.default_rng(opts["seed"])
    sample_ids = np.sort(rng.choice(count, size=opts["train_size"], replace=False))
    np.save(output / "training-ids.npy", sample_ids, allow_pickle=False)
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=shards, init="k-means++", n_init=opts["n_init"],
                       max_iter=opts["iterations"], random_state=opts["seed"], algorithm="lloyd")
        model.fit(normalize_vectors(source[sample_ids], config["distance"]))
    # Preserve arithmetic means: cosine centroids are intentionally not renormalized.
    centroids = model.cluster_centers_
    np.save(output / "centroids.npy", centroids, allow_pickle=False)
    wanted = set(map(int, truth.ravel()))
    membership, counts = {}, [0] * shards
    digest = hashlib.sha256()
    with ExitStack() as stack:
        sinks = [stack.enter_context((output / f"ids-{s}.u64le").open("xb")) for s in range(shards)]
        sink = stack.enter_context((output / "assignments.jsonl").open("xb"))
        # Direct f32 squared-L2 accumulation, dimension order, shard-ID tie-break:
        # the same arithmetic ordering as SimpleKmeansRouter::route_query.
        for start in range(0, count, 16_384):
            chunk = normalize_vectors(source[start:start + 16_384], config["distance"])
            distances = np.zeros((len(chunk), shards), dtype=np.float32)
            for d in range(dimension):
                delta = chunk[:, d, None] - centroids[None, :, d]
                distances += delta * delta
            if not np.isfinite(distances).all():
                raise ValueError("KMeans assignment distance overflow")
            owners = distances.argmin(axis=1)
            ids = np.arange(start, start + len(chunk), dtype="<u8")
            for s in range(shards):
                selected = ids[owners == s]
                sinks[s].write(selected.tobytes())
                counts[s] += len(selected)
            lines = []
            for point, owner in zip(ids, owners):
                point, owner = int(point), int(owner)
                lines.append(f'{{"id":{point},"shards":[{owner}]}}\n')
                if point in wanted:
                    membership[point] = [owner]
            block = "".join(lines).encode()
            digest.update(block)
            sink.write(block)
    artifact = {"format_version": 1, "generation": 1,
                "vector_schema": {"vector_name": manifest["vector_name"], "dimension": dimension,
                                  "distance": config["distance"], "datatype": config.get("datatype", "float32")},
                "shard_count": shards, "layout_sha256": digest.hexdigest(),
                "logical_point_count": count, "physical_point_count": count,
                "routing_distance": "squared_l2", "nprobe": 1,
                "lower_hnsw_ef": config["lower"]["kmeans_ef"],
                "centroids": [{"shard_id": s, "vector": v.tolist()} for s, v in enumerate(centroids)]}
    write_json(output / "artifact.json", artifact)
    write_json(output / "training.json", {"parameters": opts, "inertia": float(model.inertia_),
               "iterations_run": int(model.n_iter_), "uses_test_queries": False,
               "uses_ground_truth": False, "training_ids_sha256": sha256(output / "training-ids.npy")})
    return membership, counts


def worker(binary, command, request, stem):
    request_path = stem.with_suffix(".request.json")
    result_path = stem.with_suffix(".jsonl")
    partial = stem.with_suffix(".partial.jsonl")
    write_json(request_path, request)
    env = os.environ.copy()
    # Do not inherit a running experiment's generation-dependent selector.
    for name in list(env):
        if name.startswith("ORION_"):
            env.pop(name)
    with stem.with_suffix(".log").open("xb") as log:
        subprocess.run([str(binary), command, str(request_path), str(partial)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    partial.rename(result_path)
    return result_path


def load_routes(path, queries, shards, method, nprobe=None):
    routes = []
    with path.open() as source:
        for index, line in enumerate(source):
            row = json.loads(line)
            targets = row["targets"]
            selected = [t["shard_id"] for t in targets]
            if row["query_index"] != index or not selected or len(set(selected)) != len(selected):
                raise ValueError("invalid native route row/order/duplicate shards")
            if any(not 0 <= s < shards for s in selected) or any(t["ef"] <= 0 for t in targets):
                raise ValueError("invalid shard or EF")
            if method == "kmeans" and (len(selected) != nprobe or any(t["entry_points"] is not None for t in targets)):
                raise ValueError("KMeans route violates nprobe/conventional-entry contract")
            if method == "orion" and any(not t["entry_points"] for t in targets):
                raise ValueError("Orion route is missing MultiEP entries")
            routes.append(targets)
    if len(routes) != queries:
        raise ValueError("route query count mismatch")
    return routes


def evaluate_method(config, manifest, files, binary, output, method, truth, membership, counts):
    shards, query_count, top = manifest["shard_count"], len(truth), truth.shape[1]
    oracle, masks = oracle_matrix(truth, membership, shards)
    np.save(output / "oracle.npy", oracle, allow_pickle=False)
    artifact = files["orion_artifact"] if method == "orion" else output / "artifact.json"
    base_request = {"method": method, "artifact": str(artifact),
                    "queries": str(output.parent / "queries.f32le"),
                    "query_count": query_count, "dimension": manifest["dimension"],
                    "shard_count": shards, "point_count": manifest["point_count"],
                    "physical_point_count": sum(counts), "distance": config["distance"],
                    "datatype": config.get("datatype", "float32"),
                    "layout_sha256": manifest["assignments_sha256"] if method == "orion"
                        else read_json(artifact)["layout_sha256"]}
    if method == "orion":
        profiles = []
        for profile in config["orion_profiles"]:
            profile = dict(profile)
            profile["whitebox"] = dict(profile["whitebox"], distance=config["distance"])
            profiles.append(profile)
    else:
        profiles = [{"name": f"nprobe-{n}", "nprobe": n, "lower_hnsw_ef": config["lower"]["kmeans_ef"]}
                    for n in range(1, shards + 1)]
    all_routes = {}
    tasks = [[] for _ in range(shards)]
    for profile in profiles:
        name = profile["name"]
        request = dict(base_request, **{k: v for k, v in profile.items() if k != "name"})
        result = worker(binary, "route", request, output / f"route-{name}")
        routes = load_routes(result, query_count, shards, method, profile.get("nprobe"))
        all_routes[name] = routes
        for q, targets in enumerate(routes):
            for target in targets:
                tasks[target["shard_id"]].append({"profile": name, "query_index": q, "target": target})
    merged = {p["name"]: [[] for _ in truth] for p in profiles}
    for shard in range(shards):
        if not counts[shard]:
            if method == "orion" and tasks[shard]:
                raise ValueError("Orion selected an empty shard")
            # Empty KMeans partitions still consume a routed shard request.
            continue
        task_path = output / f"tasks-{shard}.jsonl"
        with task_path.open("x") as sink:
            for task in tasks[shard]:
                sink.write(json.dumps(task, allow_nan=False) + "\n")
        if not tasks[shard]:
            continue
        lower = config["lower"]
        request = {"vectors": str(files["vectors"]), "vector_encoding": manifest.get("vector_encoding", "f32le"),
                   "datatype": config.get("datatype", "float32"), "point_count": manifest["point_count"],
                   "dimension": manifest["dimension"], "distance": config["distance"],
                   "ids": str(output / f"ids-{shard}.u64le"),
                   "queries": str(output.parent / "queries.f32le"), "query_count": query_count,
                   "tasks": str(task_path), "work_dir": str(output / "temporary-graphs"),
                   "top_k": top, "m": lower["m"], "ef_construct": lower["ef_construct"], "seed": lower.get("seed", 100)}
        print(f"{method}: build/search shard {shard + 1}/{shards}, copies={counts[shard]}", flush=True)
        result = worker(binary, "lower", request, output / f"lower-{shard}")
        with result.open() as source:
            rows = 0
            for index, line in enumerate(source):
                row = json.loads(line)
                if index >= len(tasks[shard]):
                    raise ValueError("unexpected extra local search result")
                expected = tasks[shard][index]
                if (row["profile"], row["query_index"]) != (expected["profile"], expected["query_index"]):
                    raise ValueError("local search result order mismatch")
                previous = merged[row["profile"]][row["query_index"]]
                merged[row["profile"]][row["query_index"]] = merge_topk(previous, row["points"], top, config["distance"])
                rows += 1
            if rows != len(tasks[shard]):
                raise ValueError("missing local search result")
    return summarize_method(config, manifest, output, method, profiles, all_routes,
                            merged, truth, counts, oracle, masks)


def fanout_groups(rows, shards):
    """E[Recall | actual distinct visited shards = n], including explicit empty bins."""
    groups = []
    for n in range(1, shards + 1):
        group = [r for r in rows if r["searched_shards"] == n]
        groups.append({"searched_shards": n, "query_count": len(group),
                       "recall": float(np.mean([r["recall"] for r in group])) if group else None,
                       "routed_coverage": float(np.mean([r["routed_coverage"] for r in group])) if group else None,
                       "oracle_recall": float(np.mean([r["oracle_at_actual_n"] for r in group]))
                           if group and all("oracle_at_actual_n" in r for r in group) else None})
        oracle = groups[-1]['oracle_recall']
        groups[-1]['oracle_gap'] = oracle - groups[-1]['recall'] if oracle is not None else None
    return groups


def summarize_method(config, manifest, output, method, profiles, all_routes,
                     merged, truth, counts, oracle, masks):
    shards, query_count, top = manifest["shard_count"], len(truth), truth.shape[1]
    curves = []
    summaries, per_query, by_fanout = [], [], []
    for profile in profiles:
        name = profile["name"]
        rows = []
        for q, targets in enumerate(all_routes[name]):
            ann_recall = None
            if merged is not None:
                ids = [point for point, _ in merged[name][q]]
                ann_recall = len(set(ids) & set(map(int, truth[q]))) / top
            covered = coverage(masks[q], targets, top)
            n = len(targets)
            if covered > oracle[q, n] + 1e-12 or (ann_recall is not None and ann_recall > covered + 1e-12):
                raise ValueError("ANN recall <= Real GT coverage <= Oracle invariant violated")
            row = {"method": method, "profile": name, "query_index": q, "searched_shards": n,
                   "recall": covered, "routed_coverage": covered, "oracle_at_actual_n": oracle[q, n],
                   "oracle_gap": float(oracle[q, n] - covered), "ann_recall": ann_recall}
            rows.append(row)
            per_query.append(row)
        mean_n = float(np.mean([r["searched_shards"] for r in rows]))
        mean_recall = float(np.mean([r["recall"] for r in rows]))
        for group in fanout_groups(rows, shards):
            by_fanout.append({"method": method, "profile": name, **group})
            selected_profile = config.get("real_profile", profiles[0]["name"])
            if ((method == "orion" and name == selected_profile) or
                    (method == "kmeans" and group["searched_shards"] == profile["nprobe"])):
                for metric, key in (("Oracle", "oracle_recall"), ("Real", "recall")):
                    curves.append({"method": method, "metric": metric, "profile": name,
                                   "searched_shards": group["searched_shards"],
                                   "recall": group[key], "query_count": group["query_count"]})
                if group["query_count"] and group["recall"] > group["oracle_recall"] + 1e-12:
                    raise ValueError("paired group Real exceeds Oracle")
        summaries.append({"profile": profile, "mean_searched_shards": mean_n,
                          "min_searched_shards": min(r["searched_shards"] for r in rows),
                          "max_searched_shards": max(r["searched_shards"] for r in rows),
                          "real_recall": mean_recall,
                          "mean_routed_coverage": float(np.mean([r["routed_coverage"] for r in rows])),
                          "mean_oracle_at_actual_n": float(np.mean([r["oracle_at_actual_n"] for r in rows])),
                          "mean_oracle_gap": float(np.mean([r["oracle_gap"] for r in rows])),
                          "mean_ann_recall": float(np.mean([r["ann_recall"] for r in rows])) if merged is not None else None})
    write_json(output / "summary.json", {"metric_schema": METRIC_SCHEMA,
               "real_definition": "unique GT coverage of the actually selected shards; exact within-shard assumption",
               "oracle_population": "same queries as Real at each n", "shard_sizes": counts,
               "physical_copies": sum(counts), "replication_factor": sum(counts) / manifest["point_count"],
               "profiles": summaries})
    return curves, per_query, by_fanout


def write_csv(path, rows):
    with path.open("x", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(curves, output, top, shards):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, color in (("orion", "#D55E00"), ("kmeans", "#0072B2")):
        for metric, style, marker in (("Oracle", "--", None), ("Real", "-", "o")):
            rows = sorted((r for r in curves if r["method"] == method and r["metric"] == metric),
                          key=lambda r: r["searched_shards"])
            ax.plot([r["searched_shards"] for r in rows], [r["recall"] if r["recall"] is not None else np.nan for r in rows],
                    label=f'{"Orion" if method == "orion" else "KMeans"} {metric}',
                    color=color, linestyle=style, marker=marker, markersize=4)
    ax.set(xlabel="Actual searched shards n (Real grouped by n)", ylabel=f"Recall@{top}",
           xlim=(1, max(2, shards)), ylim=(0, 1.02))
    ax.set_xticks(range(1, shards + 1))
    ax.grid(alpha=.25)
    ax.legend()
    fig.text(.5, .015, "Real: GT coverage of routed shards. Oracle: best n shards for the SAME query subset.\n"
             "Query groups may differ between methods. Empty groups are N/A.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"locality.{extension}", dpi=180)
    plt.close(fig)


def execute(config, manifest, files, binary, output):
    if config.get("execution", {}).get("mode") == "eight_hosts":
        raise ValueError("eight-host experiment cannot use the single-machine rebuild runner; use shard_locality_archives.py with matched archives")
    if config.get("kmeans", {}).get("reuse_existing_index"):
        raise ValueError("existing-index comparison cannot invoke the rebuild runner")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("--native-binary must be a prebuilt executable; see shard_locality.md")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    print("Checking full dataset and routing artifact checksums...", flush=True)
    hashes = {}
    for key in ("vectors", "orion_artifact"):
        hashes[key] = sha256(files[key])
        if hashes[key] != manifest[key + "_sha256"]:
            raise ValueError(f"{key} checksum mismatch")
    hashes["assignments"] = manifest["assignments_sha256"]  # verified during split_orion
    _, truth = load_queries(config, manifest, output)
    import sklearn
    write_json(output / "provenance.json", {"inputs": {k: str(v) for k, v in files.items()},
               "input_sha256": hashes, "queries_npz_sha256": sha256(config["queries_npz"]),
               "queries_f32le_sha256": sha256(output / "queries.f32le"),
               "truth_sha256": sha256(output / "truth.npy"),
               "native_binary_sha256": sha256(binary), "script_sha256": sha256(Path(__file__)),
               "native_source_sha256": sha256(REPO / "lib/collection/examples/shard_locality.rs"),
               "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
               "query_count": len(truth), "top_k": truth.shape[1], "shard_count": manifest["shard_count"],
               "real_backend": "native routers + single unquantized GraphLayers HNSW per shard",
               "metric_schema": METRIC_SCHEMA,
               "qps_measured": False})
    curves, per_query, by_fanout = [], [], []
    for method in ("orion", "kmeans"):
        folder = output / method
        folder.mkdir()
        if method == "orion":
            membership, counts = split_orion(manifest, files["assignments"], truth, folder)
        else:
            membership, counts = build_kmeans(config, manifest, files["vectors"], truth, folder)
        rows, details, bins = evaluate_method(config, manifest, files, binary, folder, method, truth, membership, counts)
        curves.extend(rows)
        per_query.extend(details)
        by_fanout.extend(bins)
    write_csv(output / "curves.csv", curves)
    write_csv(output / "per_query.csv", per_query)
    write_csv(output / "real_by_fanout.csv", by_fanout)
    plot(curves, output, truth.shape[1], manifest["shard_count"])
    write_json(output / "COMPLETED.json", {"complete": True, "qps_measured": False, "metric_schema": METRIC_SCHEMA})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New independent output directory (required with --execute)")
    parser.add_argument("--native-binary", type=Path, help="Prebuilt collection example shard_locality")
    parser.add_argument("--execute", action="store_true", help="Start heavy work ONLY after the other experiment ends")
    args = parser.parse_args(argv)
    config, manifest, files = load_plan(args.config)
    plan = {"mode": "execute" if args.execute else "plan_only_no_data_scan_or_search",
            "execution": config.get("execution", {"mode": "single_machine"}),
            "required_build_parameters": {k: config["lower"][k] for k in ("m", "ef_construct")},
            "points": manifest["point_count"], "total_shards": manifest["shard_count"],
            "top_k": config.get("top_k", 10), "orion_profiles": len(config["orion_profiles"]),
            "kmeans_nprobe": list(range(1, manifest["shard_count"] + 1)),
            "base_vector_bytes": files["vectors"].stat().st_size,
            "id_file_bytes": 8 * (manifest["total_point_copies"] + manifest["point_count"]),
            "memory": "one lower shard plus all route profiles; upper routing loads the full upper graph",
            "real_x": "integer n; mean recall of queries visiting exactly n shards; empty bins are N/A",
            "qps": False}
    print(json.dumps(plan, indent=2))
    if args.execute:
        if args.output is None or args.native_binary is None:
            parser.error("--execute requires --output and --native-binary")
        execute(config, manifest, files, args.native_binary.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
