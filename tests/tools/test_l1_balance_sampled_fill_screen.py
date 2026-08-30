from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "experiments/l1_balance/sampled_fill_screen.py"
SPEC = importlib.util.spec_from_file_location("sampled_fill_screen_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
screen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = screen
SPEC.loader.exec_module(screen)


def test_deterministic_sample_is_sorted_unique_and_seeded() -> None:
    first = screen.deterministic_sample_indices(1000, 0.01, 7)
    second = screen.deterministic_sample_indices(1000, 0.01, 7)
    other = screen.deterministic_sample_indices(1000, 0.01, 19)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 10
    assert len(np.unique(first)) == 10
    assert np.all(first[1:] > first[:-1])
    assert not np.array_equal(first, other)


def test_occurrence_mass_counts_every_sampled_hit() -> None:
    sample = np.asarray([[0, 1, 1], [1, 2, 3]], dtype=np.int32)
    mass = screen.occurrence_mass(sample, 5)
    np.testing.assert_array_equal(mass, np.asarray([1, 3, 1, 1, 0]))
    assert int(np.sum(mass)) == sample.size


def test_blended_proxy_mass_has_exact_endpoints_and_equal_total() -> None:
    sample = np.asarray([0, 1, 3, 0], dtype=np.uint64)
    upper = np.asarray([1, 1, 1, 1], dtype=np.uint64)
    prior = screen.blended_proxy_mass(sample, upper, 0.0)
    sampled = screen.blended_proxy_mass(sample, upper, 1.0)
    middle = screen.blended_proxy_mass(sample, upper, 0.5)
    np.testing.assert_array_equal(prior, upper * int(np.sum(sample)))
    np.testing.assert_array_equal(sampled, sample * int(np.sum(upper)))
    np.testing.assert_array_equal(middle, upper * 4 + sample * 4)
    assert int(np.sum(middle)) == 32

    with pytest.raises(ValueError, match="sample_weight"):
        screen.blended_proxy_mass(sample, upper, 1.1)


def test_prediction_metrics_are_exact_under_uniform_scale() -> None:
    metrics = screen.prediction_metrics(
        np.asarray([1, 2, 3, 4]), np.asarray([10, 20, 30, 40])
    )
    assert metrics["prediction_mape"] == pytest.approx(0.0)
    assert metrics["prediction_rmse_over_full_mean"] == pytest.approx(0.0)
    assert metrics["prediction_pearson"] == pytest.approx(1.0)
    assert metrics["prediction_spearman"] == pytest.approx(1.0)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_sample_fraction_is_bounded(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        screen.deterministic_sample_indices(10, fraction, 1)
