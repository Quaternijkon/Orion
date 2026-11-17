"""Small synthetic tests; never read experiment datasets or start servers."""
import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import tarfile
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import shard_locality as locality
from tools import shard_locality_archives as archives
from tools import shard_locality_split as parallel_split


class LocalityTests(unittest.TestCase):
    def test_exact_oracle_matches_exhaustive_subsets(self):
        rng = random.Random(101)
        for _ in range(100):
            masks = [rng.randrange(1 << 5) for _ in range(6)]
            actual = locality.exact_oracle(masks, 5)
            expected = []
            for n in range(7):
                best = 0
                for subset in itertools.combinations(masks, n):
                    union = 0
                    for mask in subset:
                        union |= mask
                    best = max(best, union.bit_count())
                expected.append(best / 5)
            np.testing.assert_array_equal(actual, expected)

    def test_overlap_oracle_is_not_count_sorting_or_greedy(self):
        # A covers 1234; B covers 125; C covers 346. Best two are B+C.
        masks = [0b001111, 0b010011, 0b101100]
        actual = locality.exact_oracle(masks, 6)
        np.testing.assert_allclose(actual, [0, 4 / 6, 1, 1])
        self.assertEqual((masks[0] | masks[1]).bit_count(), 5)
        # Two copies of the same neighbor are still one hit.
        np.testing.assert_allclose(locality.exact_oracle([1, 1, 2], 2), [0, .5, 1, 1])

    def test_kmeans_oracle_equals_sorted_disjoint_counts(self):
        masks = [0b000001, 0b001110, 0b110000]
        np.testing.assert_allclose(locality.exact_oracle(masks, 6), [0, .5, 5 / 6, 1])

    def test_merge_deduplicates_source_ids_and_respects_negative_l2(self):
        merged = locality.merge_topk([(4, -2), (8, -5)],
                                    [{"id": 4, "score": -2}, {"id": 3, "score": -1},
                                     {"id": 2, "score": -2}], 3)
        self.assertEqual(merged, [(3, -1), (2, -2), (4, -2)])
        self.assertEqual(locality.merge_topk([(1, .5)], [{"id": 2, "score": .5}], 1, "Cosine"), [(2, .5)])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            locality.merge_topk([], [{"id": 1, "score": float("nan")}], 1)

    def fixture(self, root):
        base = np.array([[0., 0.], [1., 0.], [10., 0.], [11., 0.]], dtype="<f4")
        queries = np.array([[0., 0.], [10., 0.]], dtype="<f4")
        truth = np.array([[0, 1], [2, 3]], dtype=np.int64)
        base.tofile(root / "base.f32le")
        np.savez(root / "queries.npz", queries=queries, truth=truth)
        (root / "artifact.json").write_text("{}")  # native worker mocked in Python flow test only
        rows = [{"id": 0, "shards": [0]}, {"id": 1, "shards": [0, 1]},
                {"id": 2, "shards": [1]}, {"id": 3, "shards": [1]}]
        with (root / "assignments.jsonl").open("w") as sink:
            for row in rows:
                sink.write(json.dumps(row) + "\n")
        manifest = {"format_version": 1, "dimension": 2, "point_count": 4, "shard_count": 2,
                    "vector_name": "", "orion_generation": 1, "total_point_copies": 5,
                    "orion_artifact_file": "artifact.json", "vectors_file": "base.f32le",
                    "assignments_file": "assignments.jsonl"}
        for key in ("orion_artifact", "vectors", "assignments"):
            manifest[key + "_sha256"] = locality.sha256(root / manifest[key + "_file"])
        locality.write_json(root / "manifest.json", manifest)
        config = {"format_version": 1, "orion_manifest": "manifest.json", "queries_npz": "queries.npz",
                  "top_k": 2, "distance": "Euclid", "datatype": "float32",
                  "lower": {"m": 2, "ef_construct": 8, "seed": 100, "kmeans_ef": 8},
                  "kmeans": {"train_size": 4, "iterations": 10, "seed": 1, "n_init": 1},
                  "orion_profiles": [{"name": "native", "upper_k": 2, "upper_ef_search": 4,
                      "dynamic_ef_base": 4, "dynamic_ef_factor": 2,
                      "whitebox": {"prefix_k": 1, "max_prefix_entries": 0, "max_distance_gap": 1.}}]}
        locality.write_json(root / "config.json", config)
        return config, manifest, base, queries, truth

    def test_default_plan_does_not_scan_vectors_or_execute(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with patch.object(locality, "execute") as run, patch.object(locality, "sha256") as digest:
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    locality.main(["--config", str(root / "config.json")])
                self.assertIn("plan_only_no_data_scan_or_search", out.getvalue())
                run.assert_not_called()
                digest.assert_not_called()
            self.assertFalse((root / "COMPLETED.json").exists())

    def test_assignment_validation_fails_on_corruption_and_gt_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, manifest, _, _, truth = self.fixture(root)
            output = root / "split"
            output.mkdir()
            membership, counts = locality.split_orion(manifest, root / "assignments.jsonl", truth, output)
            self.assertEqual(counts, [2, 3])
            self.assertEqual(membership[1], [0, 1])
            oracle, masks = locality.oracle_matrix(truth, membership, 2)
            np.testing.assert_allclose(oracle, [[0, 1, 1], [0, 1, 1]])
            self.assertEqual(locality.coverage(masks[0], [{"shard_id": 1}], 2), .5)
            with self.assertRaises(KeyError):
                locality.oracle_matrix(truth, {0: [0]}, 2)
            (root / "bad").mkdir()
            manifest["assignments_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum"):
                locality.split_orion(manifest, root / "assignments.jsonl", truth, root / "bad")

    def test_truth_validation_rejects_duplicates_and_mismatched_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, manifest, _, queries, _ = self.fixture(root)
            config["queries_npz"] = str(root / "queries.npz")
            np.savez(root / "queries.npz", queries=queries, truth=np.array([[0, 0], [2, 3]]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                locality.load_queries(config, manifest, root)
            np.savez(root / "queries.npz", queries=queries[:1], truth=np.array([[0, 1], [2, 3]]))
            with self.assertRaisesRegex(ValueError, "aligned"):
                locality.load_queries(config, manifest, root)

    def test_route_validation_never_equates_upper_k_with_fanout(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "route.jsonl"
            path.write_text(json.dumps({"query_index": 0, "targets": [
                {"shard_id": 1, "ef": 44, "entry_points": [3, 4, 5]}]}) + "\n")
            result = locality.load_routes(path, 1, 2, "orion")
            self.assertEqual(len(result[0]), 1)
            with self.assertRaisesRegex(ValueError, "KMeans"):
                locality.load_routes(path, 1, 2, "kmeans", 2)

    def test_small_pipeline_outputs_actual_fanout_and_all_four_series(self):
        # Tests Python orchestration/scoring, not native-HNSW correctness. The
        # stand-in worker runs exact local search on four synthetic vectors.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            config, manifest, files = locality.load_plan(root / "config.json")

            def fake_worker(binary, command, request, stem):
                output = stem.with_suffix(".jsonl")
                rows = []
                if command == "route":
                    for q in range(2):
                        if request["method"] == "orion":
                            selected = [0] if q == 0 else [0, 1]
                            targets = [{"shard_id": s, "ef": 8, "entry_points": [0 if s == 0 else 2]}
                                       for s in selected]
                        else:
                            selected = list(range(request["nprobe"]))
                            targets = [{"shard_id": s, "ef": 8, "entry_points": None} for s in selected]
                        rows.append({"query_index": q, "targets": targets})
                else:
                    base = np.fromfile(request["vectors"], dtype="<f4").reshape(-1, 2)
                    queries = np.fromfile(request["queries"], dtype="<f4").reshape(-1, 2)
                    ids = np.fromfile(request["ids"], dtype="<u8")
                    for line in Path(request["tasks"]).read_text().splitlines():
                        task = json.loads(line)
                        points = [{"id": int(i), "score": -float(np.sum((base[i] - queries[task["query_index"]]) ** 2))}
                                  for i in ids]
                        points.sort(key=lambda p: (-p["score"], p["id"]))
                        rows.append({"profile": task["profile"], "query_index": task["query_index"],
                                     "points": points[:request["top_k"]]})
                output.write_text("".join(json.dumps(r) + "\n" for r in rows))
                return output

            with patch.object(locality, "worker", side_effect=fake_worker):
                with contextlib.redirect_stdout(io.StringIO()):
                    locality.execute(config, manifest, files, Path(sys.executable), root / "out")
            import csv
            with (root / "out/curves.csv").open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual({(r["method"], r["metric"]) for r in rows},
                             {(m, k) for m in ("orion", "kmeans") for k in ("Oracle", "Real")})
            real = [r for r in rows if r["method"] == "orion" and r["metric"] == "Real"]
            self.assertEqual([int(r["searched_shards"]) for r in real], [1, 2])
            self.assertEqual([int(r["query_count"]) for r in real], [1, 1])
            self.assertTrue(all(float(r["recall"]) == 1 for r in real))
            summary = locality.read_json(root / "out/orion/summary.json")
            self.assertEqual(summary["replication_factor"], 1.25)
            for name in ("locality.png", "locality.pdf", "per_query.csv", "real_by_fanout.csv", "COMPLETED.json"):
                self.assertTrue((root / "out" / name).is_file(), name)
            with self.assertRaises(FileExistsError):
                locality.execute(config, manifest, files, Path(sys.executable), root / "out")

    def test_integer_fanout_groups_preserve_missing_bins_and_conditional_means(self):
        rows = [{"searched_shards": 1, "recall": r, "routed_coverage": 1.} for r in [.2, .8]]
        rows += [{"searched_shards": 3, "recall": .9, "routed_coverage": 1.}]
        groups = locality.fanout_groups(rows, 8)
        self.assertEqual([g["searched_shards"] for g in groups], list(range(1, 9)))
        self.assertEqual(groups[0]["recall"], .5)
        self.assertEqual(groups[0]["query_count"], 2)
        self.assertIsNone(groups[1]["recall"])
        self.assertEqual(groups[1]["query_count"], 0)
        self.assertEqual(groups[2]["recall"], .9)

    def archive_fixture(self, root):
        artifact = json.dumps({"logical_point_count": 4}).encode()
        config = {"params": {"shard_number": 2}, "auto_shard_policy": {
            "type": "simple_kmeans", "generation": 1, "artifact_sha256": hashlib.sha256(artifact).hexdigest()}}
        for shard in range(2):
            prefix = f"storage/collections/sift100m/{shard}/segments/abc"
            files = {
                "storage/collections/sift100m/config.json": json.dumps(config).encode(),
                "storage/collections/sift100m/simple_kmeans_router/generation-1.json": artifact,
                prefix + "/segment.json": json.dumps({"config": {"vector_data": {"": {
                    "index": {"type": "hnsw"}}}}}).encode(),
                prefix + "/vector_index/hnsw_config.json": b'{"indexed_vector_count":2}',
                prefix + "/vector_index/graph.bin": b"unchanged-index-bytes"}
            manifest = {k: {"size": len(v), "sha256": hashlib.sha256(v).hexdigest()} for k,v in files.items()}
            with tarfile.open(root / f"host{shard}.tar", "w") as tar:
                for name, data in files.items():
                    info = tarfile.TarInfo("./" + name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            locality.write_json(root / f"host{shard}.manifest.json", manifest)

    def test_archive_metadata_requires_all_shards_and_preserves_graph_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.archive_fixture(root)
            records, _, _ = archives.inspect_archives(root, "kmeans")
            self.assertEqual([r["shard"] for r in records], [0, 1])
            target = root / "copy"
            target.mkdir()
            segments = archives.extract_segments(records[0], target)
            self.assertEqual((Path(segments[0]) / "vector_index/graph.bin").read_bytes(), b"unchanged-index-bytes")
            (root / "host1.tar").unlink()
            with self.assertRaisesRegex(ValueError, "missing archives"):
                archives.inspect_archives(root, "kmeans")

    def test_archive_corruption_and_unsafe_names_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.archive_fixture(root)
            records, _, _ = archives.inspect_archives(root, "kmeans")
            path = root / "host0.manifest.json"
            manifest = locality.read_json(path)
            key = next(k for k in manifest if k.endswith("graph.bin"))
            manifest[key]["sha256"] = "0" * 64
            path.write_text(json.dumps(manifest))
            records[0]["manifest_sha256"] = locality.sha256(path)
            with self.assertRaisesRegex(ValueError, "checksum"):
                archives.extract_segments(records[0], root / "copy")
            for name in ("/absolute/file", "relative/../../file"):
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    archives.member_name(name)

    def test_mismatched_index_builds_are_rejected_before_execution(self):
        config = {"lower": {"m": 32, "ef_construct": 200}}
        collection = {"hnsw_config": {"m": 32, "ef_construct": 200, "full_scan_threshold": 10},
                      "params": {"vectors": {"size": 128, "distance": "Euclid"}}}
        records = [{"shard": 0, "graph_configs": [{"m": 32, "m0": 64, "ef_construct": 200}]}]
        archives.require_matched_indexes(config, [("orion", records, collection), ("kmeans", records, collection)])
        bad = json.loads(json.dumps(collection))
        bad["hnsw_config"]["ef_construct"] = 100
        with self.assertRaisesRegex(ValueError, "required 200"):
            archives.require_matched_indexes(config, [("orion", records, bad), ("kmeans", records, collection)])
        bad_records = [{"shard": 0, "graph_configs": [{"m": 32, "m0": 64, "ef_construct": 100}]}]
        with self.assertRaisesRegex(ValueError, "physical HNSW"):
            archives.require_matched_indexes(config, [("orion", bad_records, collection)])

    def test_eight_hosts_cannot_silently_run_on_one_machine(self):
        with self.assertRaisesRegex(ValueError, "eight-host"):
            locality.execute({"execution": {"mode": "eight_hosts"}}, None, None, None, None)

    def test_upper_build_requires_200_and_does_not_need_full_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {"orion_manifest": str(root / "import.manifest.json"),
                      "lower": {"m": 32, "ef_construct": 200}}
            path = root / "build-manifest.json"
            path.write_text('{"options":{"upper_m":32,"upper_ef_construct":100},"huge_array":[')
            with self.assertRaisesRegex(ValueError, "upper graph"):
                archives.require_upper_build(config)
            path.write_text('{"options":{"upper_m":32,"upper_ef_construct":200},"huge_array":[')
            archives.require_upper_build(config)

    def test_parallel_split_preserves_every_copy_and_source_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vectors = np.arange(113 * 5, dtype='<f4').reshape(113, 5)
            vectors.tofile(root / 'vectors')
            assignments = [{'id': i, 'shards': sorted({i % 3, (i // 7) % 3})} for i in range(113)]
            # No trailing newline; partitions must handle both long and short rows.
            (root / 'assignments').write_text('\n'.join(json.dumps(r) for r in assignments))
            copies = sum(len(r['shards']) for r in assignments)
            counts = parallel_split.split(root/'assignments', root/'vectors', root/'out',
                                          113, 5, copies, shards=3, workers=4)
            for shard in range(3):
                expected = [r['id'] for r in assignments if shard in r['shards']]
                actual = np.fromfile(root/'out'/f'{shard}.u64le', dtype='<u8')
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual((root/'out'/f'{shard}.f32le').read_bytes(), vectors[expected].tobytes())
                self.assertEqual(counts[shard], len(expected))

    def test_automatic_materialization_matches_serial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            _,manifest,vectors,_,_=self.fixture(root)
            available={'worker_count':3,'selected_cpus':sorted(os.sched_getaffinity(0))[:3],
                       'memory_available_bytes':10<<30}
            for name,workers in [('serial',1),('auto',None)]:
                with patch.object(parallel_split.resources,'detect',return_value=available):
                    parallel_split.split(root/'assignments.jsonl',root/'base.f32le',root/name,
                        4,2,manifest['total_point_copies'],shards=2,workers=workers)
            self.assertEqual(locality.read_json(root/'auto/split-proof.json')['workers'],3)
            for shard in range(2):
                for suffix in ['f32le','u64le']:
                    name=f'{shard}.{suffix}'
                    self.assertEqual((root/'serial'/name).read_bytes(),(root/'auto'/name).read_bytes())

    def test_parallel_shard_search_keeps_deterministic_reduction(self):
        import threading
        barrier = threading.Barrier(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root/'artifact.json'
            artifact.write_text(json.dumps({'layout_sha256': 'a'*64, 'physical_point_count': 4}))
            manifest = {'shard_count': 2, 'point_count': 4, 'dimension': 2, 'vector_name': ''}
            config = {'distance': 'Euclid', 'lower': {'kmeans_ef': 64},
                      'collect_ann_diagnostic': True,
                      'execution': {'mode': 'eight_hosts', 'hosts': ['host0', 'host1']}}
            truth = np.array([[0, 1], [2, 3]])
            def route_worker(binary, kind, request, stem):
                path = stem.with_suffix('.jsonl')
                rows = []
                for q in range(2):
                    selected = [q] if request['nprobe'] == 1 else [0, 1]
                    rows.append({'query_index': q, 'targets': [
                        {'shard_id': s, 'ef': 64, 'entry_points': None} for s in selected]})
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows));return path
            def remote_worker(binary, request, stem, host, working):
                shard = int(host[-1]); ids = [2*shard, 2*shard+1]
                barrier.wait(timeout=5)  # both physical shard workers must be dispatched together
                rows = [{'kind': 'membership', 'segment_index': 0, 'point_count': 2,
                         'gt_ids': ids, 'config': {}}]
                for line in Path(request['tasks']).read_text().splitlines():
                    task = json.loads(line)
                    rows.append({'kind': 'search', 'segment_index': 0,
                        'query_index': task['query_index'], 'profile': task['profile'],
                        'points': [{'id': i, 'score': -abs(i-(2*task['query_index']+.2))} for i in ids]})
                path=stem.with_suffix('.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in rows));return path
            with patch.object(locality, 'worker', side_effect=route_worker), \
                 patch.object(archives, 'extract_segments', return_value=['segment']), \
                 patch.object(archives, 'remote_segments', side_effect=remote_worker), \
                 contextlib.redirect_stdout(io.StringIO()):
                curves, details, groups = archives.search_method(config, manifest, {}, Path('/fake'),
                    root, 'kmeans', truth, [{'shard': 0}, {'shard': 1}], artifact)
            self.assertTrue(all(row['recall'] == 1 for row in curves))
            self.assertEqual([row['query_index'] for row in details], [0, 1, 0, 1])
            # The default locality evaluation must send no local ANN tasks,
            # while producing the same primary GT coverage / paired Oracle.
            config.pop('collect_ann_diagnostic')
            gt_only = root/'gt-only'; gt_only.mkdir()
            with patch.object(locality, 'worker', side_effect=route_worker), \
                 patch.object(archives, 'extract_segments', return_value=['segment']), \
                 patch.object(archives, 'remote_segments', side_effect=remote_worker), \
                 contextlib.redirect_stdout(io.StringIO()):
                gt_curves, gt_details, _ = archives.search_method(config, manifest, {}, Path('/fake'),
                    gt_only, 'kmeans', truth, [{'shard': 0}, {'shard': 1}], artifact)
            self.assertEqual(gt_curves, curves)
            self.assertTrue(all(r['ann_recall'] is None for r in gt_details))
            self.assertTrue(all(p.stat().st_size == 0 for p in (gt_only/'kmeans').glob('tasks-*.jsonl')))


if __name__ == "__main__":
    unittest.main()
