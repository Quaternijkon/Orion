import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "runtime_profiles", ROOT / "tools/orion_runtime_profiles.py"
)
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)


class RuntimeProfilePlanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "build-manifest.json").write_text(json.dumps({
            "import_manifest_file": "import.manifest.json",
            "upper_indices": list(range(32)),
            "upper_graph_sha256": "a" * 64,
        }))
        (self.source / "import.manifest.json").write_text(
            json.dumps({"orion_generation": 7})
        )
        self.config = {
            "schema_version": 1, "execution_mode": "manual_only",
            "upper_ef_search": 32, "upper_k_values": [2, 8],
            "dynamic_ef_base_values": [20], "dynamic_ef_factor_values": [4],
        }

    def plan(self):
        return profiles.make_plan(
            self.config, self.source, self.root / "profiles",
            self.root / "orion_build_layout", 8,
        )

    def test_plan_preserves_independent_ef_and_linear_lower_budget_without_execution(self):
        with patch.object(profiles.subprocess, "run") as run:
            plan = self.plan()
        run.assert_not_called()
        self.assertFalse((self.root / "profiles").exists())
        self.assertIsNone(plan["search_request_hnsw_ef_override"])
        for case, k, generation in zip(plan["cases"], [2, 8], [8, 9]):
            command = case["command"]
            self.assertEqual(command[command.index("--upper-k") + 1], str(k))
            self.assertEqual(command[command.index("--upper-ef-search") + 1], "32")
            self.assertEqual(command[command.index("--dynamic-ef-base") + 1], "20")
            self.assertEqual(command[command.index("--dynamic-ef-factor") + 1], "4")
            self.assertEqual(case["maximum_requested_lower_ef"], 20 + 4 * k)
            self.assertEqual(case["generation"], generation)

    def test_rejects_invalid_top_k_or_non_linear_budget(self):
        for field, value in [
            ("upper_k_values", [33]),
            ("upper_k_values", [2, 2]),
            ("upper_k_values", [0]),
            ("upper_ef_search", 1),
            ("dynamic_ef_factor_values", [0]),
            ("dynamic_ef_base_values", [0]),
        ]:
            with self.subTest(field=field, value=value):
                original = self.config[field]
                self.config[field] = value
                with self.assertRaises(ValueError):
                    self.plan()
                self.config[field] = original

    def test_rejects_generation_reuse_and_budget_overflow(self):
        with self.assertRaises(ValueError):
            profiles.make_plan(self.config, self.source, self.root / "out", Path("cli"), 7)
        self.config["dynamic_ef_base_values"] = [profiles.MAX_USIZE]
        with self.assertRaises(ValueError):
            self.plan()

    def test_build_rejects_existing_output_before_running_any_command(self):
        executable = self.root / "orion_build_layout"
        executable.write_text("#!/bin/sh\nexit 99\n")
        executable.chmod(0o700)
        plan = self.plan()
        Path(plan["cases"][-1]["output"]).mkdir(parents=True)
        with patch.object(profiles.subprocess, "run") as run:
            with self.assertRaises(FileExistsError):
                profiles.build_profiles(plan)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
