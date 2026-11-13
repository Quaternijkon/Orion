"""Capacity acceptance and feedback semantics, independent of ANN timing."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import orion_capacity_partition as capacity


class CapacityPlanningTests(unittest.TestCase):
    def setUp(self):
        self.hits = np.array([[2, 3, 1], [0, 1, 2], [0, 1, 2],
                              [2, 3, 0], [0, 1, 2], [2, 3, 0]])
        self.upper = np.arange(4)
        self.owners = np.array([0, 0, 1, 1])

    def test_actual_counts_include_owner_copies_and_feedback_conserves_weight(self):
        counts, added, weights = capacity.evaluate_memberships(
            self.hits, self.owners, self.upper, 2, feedback=True)
        # Rows0 and2 must also exist in their entry owner's shard.
        self.assertEqual(counts.tolist(), [4, 4])
        self.assertEqual(added, 2)
        self.assertEqual(weights.sum(), 8000)
        self.assertTrue(capacity.feasible(counts, 6, None))
        self.assertFalse(capacity.feasible(counts, 6, 1.1))

    def test_feasible_first_partition_never_computes_feedback_or_runs_solver_again(self):
        calls = []
        def solver(weights):
            calls.append(weights.copy())
            return self.owners
        with patch.object(capacity, 'evaluate_memberships', wraps=capacity.evaluate_memberships) as evaluate:
            owners, history = capacity.choose_partition(self.hits, self.upper, 2, solver)
        self.assertEqual(len(calls), 1)
        self.assertEqual(evaluate.call_count, 1)
        self.assertNotIn('feedback', evaluate.call_args.kwargs)
        self.assertEqual(len(history), 1)
        np.testing.assert_array_equal(owners, self.owners)

    def test_feedback_only_after_infeasible_actual_counts(self):
        proposals = iter([np.array([0, 0, 0, 1]), self.owners])
        calls = []
        def solver(weights):
            calls.append(weights.copy())
            return next(proposals)
        _, history = capacity.choose_partition(self.hits, self.upper, 2, solver)
        self.assertEqual([r['accepted'] for r in history], [False, True])
        self.assertEqual(len(calls), 2)
        self.assertFalse(np.array_equal(calls[0], calls[1]))

    def test_feedback_limit_rejects_infeasible_layout_without_unbounded_retry(self):
        attempts = []
        with self.assertRaisesRegex(ValueError, 'no feasible'):
            capacity.choose_partition(self.hits, self.upper, 2, lambda _: self.owners,
                                      max_feedback_rounds=1, max_replication_factor=1.1,
                                      record=attempts.append)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(not x['accepted'] for x in attempts))

    def test_counter_supports_more_than_eight_shards(self):
        # Each entry's own data row routes to itself, without uint8 membership masks.
        counts, added, _ = capacity.evaluate_memberships(
            np.arange(10)[:, None], np.arange(10), np.arange(10), 10)
        self.assertEqual(counts.tolist(), [1] * 10)
        self.assertEqual(added, 0)

    def test_invalid_limits_and_duplicate_hits_are_rejected(self):
        for limit in [float('nan'), float('inf'), .99]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                capacity.check_limit(limit)
        with self.assertRaises(ValueError):
            capacity.evaluate_memberships(np.array([[0, 0]]), np.array([0]), np.array([0]), 1)


if __name__ == '__main__':
    unittest.main()
