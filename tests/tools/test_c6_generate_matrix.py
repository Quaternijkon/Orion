from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "experiments/c6/scripts/c6_generate_matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("c6_generate_matrix", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_covers_full_fanout_uniform_and_recommended_adaptive_grids():
    module = load_module()
    matrix = module.generate_matrix(4)
    assert len(matrix["P0"]) == 4 * 7
    assert len(matrix["P1"]) == 7
    assert len(matrix["P2"]) == 4 * 5 * 5
    assert len(matrix["P3"]) == 5 * 5
    assert {row["fixed_p"] for row in matrix["P0"]} == {1, 2, 3, 4}
    assert matrix["matrix_sha256"] == module.canonical_json_sha256(
        {key: value for key, value in matrix.items() if key != "matrix_sha256"}
    )
