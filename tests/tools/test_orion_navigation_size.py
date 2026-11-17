"""Synthetic, local-only checks. Never contact the eight experiment hosts."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


experiment = module("navigation_size", "orion_navigation_size.py")
client = module("navigation_client", "orion_navigation_size_client.py")


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.config = experiment.read(experiment.DEFAULT_CONFIG)

    def test_default_and_stage_without_execute_cannot_start_work(self):
        for argv in ([], ["build"], ["prepare"], ["benchmark"], ["all"], ["plan", "--execute"]):
            with self.subTest(argv=argv), patch.object(experiment, "Experiment") as runner, \
                    patch.object(experiment.subprocess, "run") as run, \
                    patch.object(experiment.subprocess, "Popen") as popen, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                experiment.main(argv)
                plan = json.loads(output.getvalue())
                self.assertEqual(plan["state"], "planned_not_executed")
                runner.assert_not_called()
                run.assert_not_called()
                popen.assert_not_called()

    def test_required_matrix_and_native_rounding(self):
        plan = experiment.make_plan(self.config)
        self.assertEqual([r["upper_nodes"] for r in plan["cases"]], [31250, 15625, 7812, 3906])
        self.assertEqual(self.config["build"]["ef_construct"], 100)
        self.assertEqual(len(plan["search_candidates_per_ratio"]), 36)
        self.assertEqual(len(plan["formal_order"]), 3)
        for order in plan["formal_order"]:
            self.assertEqual(sorted(order), [32, 64, 128, 256])

    def test_rejects_scope_and_budget_drift(self):
        changes = [("shards_per_host", 2), ("hosts", ["host"] * 8),
                   ("denominators", [32, 64]), ("execution_mode", "automatic"),
                   ("server_cpus", "0-23"), ("remote_root", "/tmp")]
        for key, value in changes:
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment.validate(config)
        self.config["build"]["ef_construct"] = 200
        with self.assertRaises(ValueError):
            experiment.validate(self.config)

    def test_selector_never_selects_fast_below_target_and_prefers_close_recall(self):
        rows = [{"recall": .9499, "qps": 999999}, {"recall": .955, "qps": 100},
                {"recall": .959, "qps": 110}, {"recall": .98, "qps": 120}]
        self.assertEqual(experiment.choose_candidate(rows, self.config), rows[2])
        with self.assertRaises(ValueError):
            experiment.choose_candidate(rows[:1], self.config)

    def test_selector_reports_overshoot_instead_of_fabricating_point95(self):
        row = {"recall": .98, "qps": 120}
        self.assertEqual(experiment.choose_candidate([row], self.config), row)

    def test_cpu_threshold_is_strict_and_requires_eight_hosts(self):
        values = dict.fromkeys(self.config["hosts"], 20.)
        values[self.config["hosts"][0]] = 48 * .95
        self.assertFalse(experiment.saturation_evidence(values, self.config)["cpu_saturated"])
        values[self.config["hosts"][0]] = 48 * .95001
        self.assertTrue(experiment.saturation_evidence(values, self.config)["cpu_saturated"])
        values.pop(self.config["hosts"][-1])
        with self.assertRaises(ValueError):
            experiment.saturation_evidence(values, self.config)

    def test_unsaturated_window_cannot_qualify_despite_high_recall(self):
        row = {"recall_uniform_queries": .99, "errors": 0, "cpu_saturated": True,
               "server_cpu_cores": dict.fromkeys(self.config["hosts"], 45.)}
        self.assertFalse(experiment.qualifies_formal(row, self.config))
        row["server_cpu_cores"][self.config["hosts"][0]] = 47.
        self.assertTrue(experiment.qualifies_formal(row, self.config))
        row["recall_uniform_queries"] = .949
        self.assertFalse(experiment.qualifies_formal(row, self.config))

    def test_load_ramp_ignores_fast_unsaturated_plateau(self):
        with tempfile.TemporaryDirectory() as name:
            config = copy.deepcopy(self.config)
            config["run_root"] = str(Path(name) / "orion-navigation-size-test")
            run = experiment.Experiment(config)
            def measure(d, name, subset, workers, lanes, warmup, seconds):
                return {"qps": 200 if lanes < 16 else 100, "busiest_service_cpu_fraction": .98 if lanes == 16 else .90,
                        "cpu_saturated": lanes == 16}
            with patch.object(run, "measure", side_effect=measure) as spy:
                chosen, rows = run.screen_saturation(32, "test")
            self.assertEqual(chosen["lanes"], 16)
            self.assertEqual(spy.call_count, 5)

    def test_startup_requires_eight_voters_and_one_established_leader(self):
        peers = {str(i): {} for i in range(1, 9)}
        states = {str(i): {"peer_id": i, "peers": peers,
                  "raft_info": {"term": 2, "leader": 1, "pending_operations": 0,
                                "is_voter": True, "role": "Leader" if i == 1 else "Follower"}}
                  for i in range(1, 9)}
        self.assertTrue(experiment.cluster_stable(states))
        states["8"]["raft_info"]["is_voter"] = False
        self.assertFalse(experiment.cluster_stable(states))
        states["8"]["raft_info"]["is_voter"] = True
        states["1"]["raft_info"]["role"] = "Candidate"
        self.assertFalse(experiment.cluster_stable(states))


class LayoutSummaryTests(unittest.TestCase):
    def test_create_request_respects_native_full_scan_threshold_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            config = experiment.read(experiment.DEFAULT_CONFIG)
            config["run_root"] = str(Path(directory) / "orion-navigation-size-test")
            run = experiment.Experiment(config)
            bundle = run.root / "d32/layout"
            experiment.save(bundle / "build-manifest.json", {"import_manifest_file": "import.json"})
            experiment.save(bundle / "import.json", {"orion_generation": 1, "orion_artifact_sha256": "a" * 64})
            with patch.object(run, "api", side_effect=RuntimeError("intercept mutation")) as api:
                with self.assertRaisesRegex(RuntimeError, "intercept mutation"):
                    run.install(32, bundle, initial=True)
            body = api.call_args.args[2]
            self.assertEqual(body["hnsw_config"]["full_scan_threshold"], 10)
            self.assertEqual(body["hnsw_config"]["m"], 32)
            self.assertEqual(body["hnsw_config"]["ef_construct"], 100)
            self.assertEqual(body["shard_number"], 8)

    def test_records_real_artifact_size_after_native_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = experiment.read(experiment.DEFAULT_CONFIG)
            config["run_root"] = str(Path(directory) / "orion-navigation-size-test")
            run = experiment.Experiment(config)
            bundle = run.root / "d256/layout"
            bundle.mkdir(parents=True)
            options = {"sample_denominator": 256}
            artifact = {"upper_graph": {"nodes": [1, 2]}}
            experiment.save(bundle / "upper.json", artifact)
            experiment.save(bundle / "build-manifest.json", {
                "algorithm_version": "orion-layout-v5-access-probability-capacity",
                "upper_indices": list(range(3906)), "options": options,
                "shard_counts": [125000] * 8, "upper_graph_sha256": "a" * 64,
                "source_artifact_file": "upper.json"})
            run.write_layout_summary(256, options)
            summary = experiment.read(run.root / "d256/layout-ready.json")
            self.assertEqual(summary["upper_artifact_bytes"], len((bundle / "upper.json").read_bytes()))
            self.assertEqual(summary["upper_nodes"], 3906)
            self.assertEqual(summary["replication_factor"], 1)
            self.assertEqual(summary["build_manifest_sha256"], experiment.sha(bundle / "build-manifest.json"))
            (run.root / "d256/layout-ready.json").unlink()
            with self.assertRaisesRegex(ValueError, "requested options"):
                run.write_layout_summary(256, {"sample_denominator": 128})
            self.assertFalse((run.root / "d256/layout-ready.json").exists())


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.spec = {"workers_per_host": 1, "warmup_seconds": 1, "seconds": 2,
                     "client_cpus": [10], "query_ids": [0, 1]}
        for wid in range(8):
            # Completion at start is included, completion at end is excluded;
            # requests crossing start are counted by completion, as in the protocol.
            batch = np.array([[0., .2, 2], [.8, 1., 2], [1.8, 2., 2], [2.8, 3., 2]])
            np.savez(self.folder / f"w{wid}.npz", batches=batch, query_ids=[1, 0], counts=[2, 2], hits=[20, 18])
            experiment.save(self.folder / f"w{wid}.json", {"errors": [], "affinity": [10], "cpu_marks": [[1, 0], [3, 1]]})

    def test_batch_qps_uses_completed_queries_and_uniform_recall(self):
        result = experiment.summarize_window(self.folder, self.spec)
        self.assertEqual(result["successful_queries"], 32)
        self.assertEqual(result["qps"], 16)
        self.assertAlmostEqual(result["recall_uniform_queries"], .95)
        self.assertEqual(result["pending_batches_start"], 8)
        self.assertEqual(result["pending_batches_end"], 8)

    def test_errors_make_window_invalid(self):
        meta = experiment.read(self.folder / "w0.json")
        meta["errors"] = ["timeout"]
        experiment.save(self.folder / "w0.json", meta)
        with self.assertRaisesRegex(ValueError, "errors/affinity"):
            experiment.summarize_window(self.folder, self.spec)

    def test_missing_worker_and_wrong_subset_are_not_silent(self):
        (self.folder / "w7.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            experiment.summarize_window(self.folder, self.spec)
        np.savez(self.folder / "w7.npz", batches=[[0, 1, 4]], query_ids=[2, 3], counts=[2, 2], hits=[20, 20])
        with self.assertRaisesRegex(ValueError, "wrong query subset"):
            experiment.summarize_window(self.folder, self.spec)

    def test_summary_does_not_mark_partial_runs_complete(self):
        config = experiment.read(experiment.DEFAULT_CONFIG)
        config["run_root"] = str(self.folder)
        experiment.summarize_experiment(config)
        self.assertFalse((self.folder / "COMPLETED.json").exists())
        self.assertTrue((self.folder / "INCOMPLETE.json").exists())
        self.assertTrue(all(r["qps_median"] is None for r in experiment.read(self.folder / "summary.json")))


class WireTests(unittest.TestCase):
    def test_payload_matches_native_searchbatch_schema_without_ef_override(self):
        # Run with the historical Python environment to enable protobuf validation.
        try:
            from qdrant_client.grpc import SearchBatchPoints
        except ImportError:
            self.skipTest("qdrant_client not installed in this Python")
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "queries.npz"
            np.savez(path, queries=np.array([[1, 2], [3, 4]], dtype=np.float32), truth=np.zeros((2, 10)))
            c = client.Client({"collection": "sift1m", "queries_path": str(path)})
            request = SearchBatchPoints.FromString(c.prefix + b"".join(c.bodies) + c.suffix)
        self.assertEqual(request.collection_name, "sift1m")
        self.assertEqual(request.timeout, 60)
        self.assertEqual(len(request.search_points), 2)
        self.assertEqual(list(request.search_points[1].vector), [3, 4])
        for search in request.search_points:
            self.assertEqual(search.limit, 10)
            self.assertFalse(search.HasField("params"))
            self.assertFalse(search.HasField("shard_key_selector"))
            self.assertFalse(search.with_payload.enable)
            self.assertFalse(search.with_vectors.enable)


if __name__ == "__main__":
    unittest.main()
