import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "native_layout_args", ROOT / "tools/orion_native_layout.py"
)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class UpperBudgetArgumentsTests(unittest.TestCase):
    def args(self, *extra):
        return layout.parse_args([
            "--hdf5-path", "/unused/input.hdf5", "--output-dir", "/unused/output",
            *extra,
        ])

    def test_decoupled_upper_search_is_a_regular_configuration(self):
        args = self.args("--upper-search-ef", "256", "--upper-k", "16")
        self.assertFalse(args.allow_decoupled_runtime_upper_search)
        layout.validate_args(args)

    def test_equal_defaults_and_compatibility_flag_remain_valid(self):
        layout.validate_args(self.args())
        layout.validate_args(self.args(
            "--upper-search-ef", "256", "--upper-k", "16",
            "--allow-decoupled-runtime-upper-search",
        ))

    def test_invalid_order_and_zero_factor_are_rejected(self):
        for extra in [
            ("--upper-search-ef", "8", "--upper-k", "16"),
            ("--dynamic-ef-factor", "0"),
        ]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                layout.validate_args(self.args(*extra))

    def test_capacity_has_no_default_replication_limit_or_self_vote_requirement(self):
        args = self.args("--partition-method", "capacity", "--kmeans-iters", "0",
                         "--topology-iters", "0", "--disable-topology-refinement")
        layout.validate_args(args)
        self.assertIsNone(args.capacity_max_replication_factor)

    def test_capacity_constraints_reject_invalid_or_inapplicable_options(self):
        for extra in [
            ("--capacity-max-replication-factor", "1.1"),
            ("--partition-method", "capacity", "--capacity-max-replication-factor", "nan"),
            ("--partition-method", "capacity", "--capacity-max-replication-factor", "0.9"),
            ("--partition-method", "capacity", "--capacity-feedback-rounds", "-1"),
        ]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                layout.validate_args(self.args(*extra))


if __name__ == "__main__":
    unittest.main()
