import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/c1/scripts/c1_orion_virtual_scale.py"
    )
    name = "c1_orion_virtual_scale_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_orion_boundary_quota_uses_nominal_m_and_live_actual_shards():
    module = load_module()
    m2 = module.quota_schedule(2, [1, 1, 1, 0])
    assert np.allclose(m2[:3], [1.3166666667] * 3, atol=1e-6)
    assert m2[3] == 0.05
    assert sum(m2) == 4.0
    assert np.isclose(sum(module.quota_schedule(8, [2, 2, 2, 1])), 16.0)
    assert module.quota_schedule(32, [8, 8, 8, 8]) == [16.0] * 4


def test_config_retains_exact_and_boundary_points(tmp_path):
    module = load_module()
    layout = tmp_path / "layout"
    layout.mkdir()
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "layouts": {
                    "p1": {"actual_shards": 1, "layout_dir": str(layout)},
                    "p2": {"actual_shards": 3, "layout_dir": str(layout)},
                },
                "points": [
                    {
                        "point_id": "m1-exact-s1",
                        "nominal_m": 1,
                        "boundary": "exact",
                        "layout_key": "p1",
                    },
                    {
                        "point_id": "m2-upper-s3",
                        "nominal_m": 2,
                        "boundary": "upper",
                        "layout_key": "p2",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = module.load_config(path)
    assert config.layouts["p2"].actual_shards == 3
    assert [point.point_id for point in config.points] == [
        "m1-exact-s1",
        "m2-upper-s3",
    ]


def test_collection_and_output_names_include_actual_boundary(tmp_path):
    module = load_module()
    profile = module.LayoutProfile("p7", 10, tmp_path)
    point = module.MeasurementPoint("m8-upper-s10", 8, "upper", "p7")
    assert module.collection_name(profile) == "orion_vscale_glove_p7_20260825"
    assert module.output_point_dir(tmp_path, point, profile) == tmp_path / "m8/upper-s10"


def test_preflight_collection_reuse_is_limited_to_selected_profiles(tmp_path):
    module = load_module()
    p1 = module.LayoutProfile("p1", 1, tmp_path)
    p2 = module.LayoutProfile("p2", 3, tmp_path)
    config = module.ExperimentConfig(
        layouts={"p1": p1, "p2": p2},
        points=(module.MeasurementPoint("m1-exact-s1", 1, "exact", "p1"),),
    )
    collections = [{"name": module.collection_name(p1)}]
    assert module.validate_preflight_collections(
        collections, config, reuse_existing=True
    ) == [module.collection_name(p1)]

    try:
        module.validate_preflight_collections(
            [{"name": "unrelated"}], config, reuse_existing=True
        )
    except RuntimeError as error:
        assert "outside the selected Orion profiles" in str(error)
    else:
        raise AssertionError("unexpected collection should be rejected")


def test_recall_window_defaults_match_formal_fixed_recall_contract():
    module = load_module()
    args = module.parse_args([])
    assert args.target_recall == 0.90
    assert args.max_recall == 0.93


def test_recall_window_must_be_strictly_increasing():
    module = load_module()
    with pytest.raises(ValueError, match="recall window"):
        module.parse_args(["--target-recall", "0.93", "--max-recall", "0.93"])


def test_keep_collection_is_limited_to_reused_recall_probe():
    module = load_module()
    with pytest.raises(ValueError, match="requires both"):
        module.parse_args(["--keep-collection"])
    args = module.parse_args(
        ["--keep-collection", "--recall-only", "--reuse-existing"]
    )
    assert args.keep_collection is True
