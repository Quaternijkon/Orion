from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "experiments/l1_balance/run_historical_cnbr_bridge_guarded.py"
)


def load_module():
    name = "l1_balance_historical_cnbr_bridge_guarded_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runner_cli(module, tmp_path: Path, output: Path) -> list[str]:
    source = tmp_path / "input.bin"
    source.write_bytes(b"fixture")
    return [
        "--base-url",
        "http://controller:6333",
        "--hdf5-path",
        str(source),
        "--topology",
        str(source),
        "--deployment-manifest",
        str(source),
        "--output-dir",
        str(output / module.guarded.RUNNER_OUTPUT_NAME),
        "--collection-a",
        module.guarded.FORMAL_ARM_A_COLLECTION,
        "--collection-b",
        "orion_c_cnbr_p32_fixture",
        "--artifact-a",
        str(source),
        "--artifact-b",
        str(source),
        "--layout-dir-a",
        str(tmp_path),
        "--layout-dir-b",
        str(tmp_path),
        "--prepare-manifest-a",
        str(source),
        "--prepare-manifest-b",
        str(source),
        "--construction-cost-audit",
        str(source),
    ]


def test_bridge_guard_accepts_fixed_arm_factory_without_cli_escape_flags(
    tmp_path: Path,
) -> None:
    module = load_module()
    output = tmp_path / "guarded"
    parsed = module.validate_historical_cnbr_runner_contract(
        runner_cli(module, tmp_path, output), output
    )
    arms = module.bridge.historical_cnbr_arm_specs(parsed)
    assert arms["A"].allow_historical_prepare_deployment is True
    assert arms["A"].allow_scaling_layout is True
    assert arms["B"].allow_historical_prepare_deployment is False
    assert arms["B"].allow_l1_partition_layout is True
    assert module.guarded.validate_runner_contract is (
        module.validate_historical_cnbr_runner_contract
    )
    assert module.guarded.validate_guarded_prerequisites is (
        module.validate_historical_cnbr_guarded_prerequisites
    )


def test_bridge_guard_validates_cost_before_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    output = tmp_path / "guarded"
    parsed = module.validate_historical_cnbr_runner_contract(
        runner_cli(module, tmp_path, output), output
    )
    evidence = {
        "status": "PASS",
        "audit_sha256": "a" * 64,
        "aggregate_status": "COMPLETE_FRESH_PROJECTED_INCREMENTAL_GATE_PASS",
        "datasets": {},
    }
    monkeypatch.setattr(
        module.bridge.native_cnbr.cost_gate,
        "validate_construction_cost_v4_pass",
        lambda path: evidence,
    )
    result = module.validate_historical_cnbr_guarded_prerequisites(parsed)
    assert result["status"] == "PASS"
    assert result["construction_cost_v4"] == evidence


def test_bridge_guard_rejects_nonhistorical_arm_a_collection(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "guarded"
    cli = runner_cli(module, tmp_path, output)
    cli[cli.index(module.guarded.FORMAL_ARM_A_COLLECTION)] = "wrong-arm-a"
    with pytest.raises(ValueError, match="frozen Arm H collection"):
        module.validate_historical_cnbr_runner_contract(cli, output)


def test_bridge_guard_rejects_runner_output_outside_wrapper(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "guarded"
    cli = runner_cli(module, tmp_path, output)
    cli[cli.index(str(output / module.guarded.RUNNER_OUTPUT_NAME))] = str(
        tmp_path / "elsewhere"
    )
    with pytest.raises(ValueError, match="runner --output-dir must be"):
        module.validate_historical_cnbr_runner_contract(cli, output)


def test_bridge_guard_rejects_child_owned_benchmark_lock(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "guarded"
    cli = [
        *runner_cli(module, tmp_path, output),
        "--benchmark-lock-fd",
        "9",
        "--benchmark-lock-token",
        "fixture-token",
    ]
    with pytest.raises(ValueError, match="owned exclusively by the wrapper"):
        module.validate_historical_cnbr_runner_contract(cli, output)


def test_bridge_guard_rejects_fixed_role_factory_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    output = tmp_path / "guarded"
    original = module.bridge.historical_cnbr_arm_specs

    def drifted(args):
        arms = original(args)
        object.__setattr__(arms["B"], "allow_l1_partition_layout", False)
        return arms

    monkeypatch.setattr(module.bridge, "historical_cnbr_arm_specs", drifted)
    with pytest.raises(ValueError, match="arm-role contract drifted"):
        module.validate_historical_cnbr_runner_contract(
            runner_cli(module, tmp_path, output), output
        )
