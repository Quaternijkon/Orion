import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
import contextlib
import io


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "native_layout_args", ROOT / "tools/orion_native_layout.py"
)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class UpperBudgetArgumentsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.whitebox=Path(self.temp.name)/'whitebox.json'
        self.whitebox.write_text(json.dumps(dict(prefix_k=8,max_prefix_entries=3,max_distance_gap=1,distance='Cosine')))

    def args(self, *extra):
        return layout.parse_args([
            "--hdf5-path", "/unused/input.hdf5", "--output-dir", "/unused/output",
            '--whitebox-config',str(self.whitebox),
            *extra,
        ])

    def test_decoupled_upper_search_is_a_regular_configuration(self):
        args = self.args("--upper-search-ef", "256", "--upper-k", "16")
        self.assertTrue(args.allow_decoupled_runtime_upper_search)
        layout.validate_args(args)

    def test_equal_defaults_and_fixed_base_remain_valid(self):
        layout.validate_args(self.args())
        self.assertEqual(self.args().partition_method,'capacity')
        for flag,value in [('--partition-method','self-vote'),('--partition-method','capacity'),('--balance-mode','none'),
                           ('--rust-builder-binary','/unused/legacy-builder'),
                           ('--rust-upper-hits-binary','/unused/legacy-exporter'),
                           ('--rust-rebind-binary','/unused/legacy-finalizer')]:
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):self.args(flag,value)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):self.args('--graphless-only')

    def test_invalid_order_and_zero_factor_are_rejected(self):
        for extra in [
            ("--upper-search-ef", "8", "--upper-k", "16"),
            ("--dynamic-ef-factor", "0"),
        ]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                layout.validate_args(self.args(*extra))

    def test_capacity_has_no_default_replication_limit_or_self_vote_requirement(self):
        args = self.args()
        layout.validate_args(args)
        self.assertIsNone(args.capacity_max_replication_factor)
        self.assertEqual(args.topology_iters,0)
        layout.validate_args(self.args('--capacity-max-replication-factor','1.1'))

    def test_capacity_constraints_reject_invalid_or_inapplicable_options(self):
        for extra in [
            ("--capacity-max-replication-factor", "nan"),
            ("--capacity-max-replication-factor", "0.9"),
        ]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                layout.validate_args(self.args(*extra))


if __name__ == "__main__":
    unittest.main()
