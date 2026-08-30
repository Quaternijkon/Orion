from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


def load_module(relative_path: str, name: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module(
    "experiments/l1_balance/prepare_mass_candidates.py",
    "prepare_mass_candidates_test",
)
evaluate = load_module(
    "experiments/l1_balance/evaluate_frozen_candidates.py",
    "evaluate_frozen_candidates_test",
)


def help_text(script: str) -> str:
    completed = subprocess.run(
        [str(PYTHON), str(REPO_ROOT / script), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def test_prepare_process_has_no_l0_or_query_evaluation_cli() -> None:
    text = help_text("experiments/l1_balance/prepare_mass_candidates.py")

    assert "--attachments" not in text
    assert "--attachments-manifest" not in text
    assert "--query-hits" not in text
    assert "--ground-truth" not in text
    assert "--actual-load" not in text
    assert "--multi-assignment" not in text
    assert "--upper-navigation-hits" in text


def test_evaluator_has_no_partitioner_or_mass_estimator_call_path() -> None:
    path = REPO_ROOT / "experiments/l1_balance/evaluate_frozen_candidates.py"
    source = path.read_text(encoding="utf-8")

    assert "partition_l1_by_navigation_mass" not in source
    assert "estimate_upper_navigation_mass" not in source
    assert "estimate_raw_upper_navigation_mass" not in source
    assert "--candidate-manifest" in help_text(
        "experiments/l1_balance/evaluate_frozen_candidates.py"
    )
    assert "--attachments" in help_text(
        "experiments/l1_balance/evaluate_frozen_candidates.py"
    )


def test_prepare_upper_contract_api_has_no_l0_escape_hatch() -> None:
    signature = inspect.signature(prepare.validate_upper_navigation_contract)
    assert set(signature.parameters) == {
        "artifact_path",
        "labels",
        "vectors",
        "navigator_sha256",
        "upper_input_manifest_path",
        "navigation_hits_path",
        "navigation_manifest_path",
    }
    assert not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def test_preregistered_topology_gates_are_fixed_constants() -> None:
    assert prepare.TOPOLOGY_GATE_VERSION == "l1-topology-preregistered-v1"
    assert prepare.EDGE_CUT_MAX_RATIO == 1.03
    assert prepare.RETAINED_DEGREE_MIN_RATIO == 0.95
    assert prepare.RETAINED_DEGREE_P10_MAX_DROP == 0.025
    assert prepare.ISOLATED_FRACTION_MAX_DELTA == 0.01
    assert prepare.ISOLATED_FRACTION_ABSOLUTE_MAX == 0.03
    assert prepare.LARGEST_COMPONENT_MEAN_MAX_DROP == 0.02
    assert prepare.LARGEST_COMPONENT_MIN_FLOOR == 0.25
    parameters = inspect.signature(prepare.topology_gate_results).parameters
    assert set(parameters) == {"observed", "reference"}


def test_evaluator_recomputes_transform_bound_mass_checksum() -> None:
    values = evaluate.np.asarray([1, 2, 3, 4], dtype="<u8")
    main = evaluate.semantic_mass_sha256(
        "production_upper_navigation_top10_self_debiased_floor1",
        "max(1, raw_hit_count - 1)",
        values,
        10,
    )
    raw = evaluate.semantic_mass_sha256(
        "production_upper_navigation_top10_raw_hit_frequency",
        "raw_hit_count",
        values,
        10,
    )

    assert len(main) == 64
    assert main != raw


def test_duplicate_tie_exception_requires_bitwise_equal_predecessor() -> None:
    labels = prepare.np.arange(10, dtype=prepare.np.int64)
    vectors = prepare.np.arange(20, dtype=prepare.np.float32).reshape(10, 2)
    vectors[1] = vectors[0]
    rows = prepare.np.asarray(
        [[(row + offset) % 10 for offset in range(10)] for row in range(10)]
    )
    rows[0, 0], rows[0, 1] = 1, 0

    records, proof = prepare.duplicate_tie_proof(rows, labels, vectors)
    assert records == [
        {
            "row": 0,
            "query_label": 0,
            "self_rank": 1,
            "preceding_labels": [1],
            "vector_bits_sha256": prepare.bytes_sha256(
                prepare.np.ascontiguousarray(vectors[0]).tobytes()
            ),
        }
    ]
    assert len(proof) == 64

    vectors[1, 0] += 1
    try:
        prepare.duplicate_tie_proof(rows, labels, vectors)
    except ValueError as error:
        assert "non-duplicate" in str(error)
    else:
        raise AssertionError("non-duplicate rank predecessor was accepted")
