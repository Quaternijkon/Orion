import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/c1/scripts/c1_hashall_simple_kmeans_summarize.py"
    )
    name = "c1_hashall_simple_kmeans_summarize_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_m_axis_is_numeric_linear_not_categorical_or_logarithmic():
    module = load_module()
    figure, axis = plt.subplots()
    module.configure_linear_m_axis(axis)
    assert axis.get_xscale() == "linear"
    assert np.allclose(axis.get_xlim(), [0.0, 33.0])
    transformed = axis.transData.transform(np.column_stack([module.M_VALUES, np.zeros(6)]))[:, 0]
    gaps = np.diff(transformed)
    assert np.allclose(gaps / gaps[0], [1, 2, 4, 8, 16])
    plt.close(figure)


def test_theory_is_anchored_and_includes_cpu_and_nprobe():
    module = load_module()
    qps = np.asarray([400, 600, 900, 1200, 1600, 2000], dtype=float)
    ef = np.asarray([384, 256, 200, 160, 128, 96], dtype=float)
    nprobe = np.asarray([1, 2, 3, 5, 8, 12], dtype=float)
    theory = module.theoretical_qps(module.M_VALUES, qps, ef, nprobe)
    assert theory["predicted"][0] == qps[0]
    expected = qps[0] * module.M_VALUES * theory["total_query_work"][0] / theory["total_query_work"]
    assert np.allclose(theory["predicted"], expected)
