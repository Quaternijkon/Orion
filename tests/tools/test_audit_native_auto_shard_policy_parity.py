from __future__ import annotations

import importlib.util
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO_ROOT / "tools/audit_native_auto_shard_policy_parity.py"
    spec = importlib.util.spec_from_file_location(
        "audit_native_auto_shard_policy_parity", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_student_t_distribution_matches_reference_quantiles():
    module = load_module()

    assert math.isclose(module.student_t_cdf(1.0, 1.0), 0.75, abs_tol=1.0e-12)
    assert math.isclose(
        module.student_t_ppf(0.975, 4.0),
        2.7764451051977987,
        rel_tol=1.0e-11,
    )
    assert math.isclose(
        module.student_t_ppf(0.975, 1.0),
        12.706204736432095,
        rel_tol=1.0e-10,
    )


def test_welch_comparison_reports_direction_and_interval():
    module = load_module()

    result = module.welch_comparison(
        [1455.0, 1468.0, 1459.0],
        [1279.0, 1280.0, 1284.0],
        target_recall=0.9,
        method_a="orion",
        method_b="hash_all",
    )

    assert result["difference_qps_a_minus_b"] > 0.0
    assert result["difference_pct_vs_b"] > 10.0
    assert result["welch_95_ci_low_qps"] > 0.0
    assert 0.0 <= result["welch_two_sided_p_value"] <= 1.0
    assert "not interleaved" in result["design_note"]


def test_execution_identity_keeps_only_the_planner_policy_specific():
    module = load_module()

    assert module.EXECUTION_IDENTITY["hash_all"] == {
        "plan_kind": "all_shards",
        "planner": "explicit_all_shards",
        "executor": "qdrant_all_shards_replica_set",
    }
    assert module.EXECUTION_IDENTITY["orion"]["executor"] == (
        module.EXECUTION_IDENTITY["simple_kmeans"]["executor"]
    )
    assert module.COMMON_EXECUTION_IDENTITY["replica_read_path"] == (
        "ShardReplicaSet::core_search"
    )
    assert module.COMMON_EXECUTION_IDENTITY["global_merge_path"] == (
        "Collection::merge_from_shards"
    )
