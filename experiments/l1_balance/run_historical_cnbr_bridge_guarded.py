#!/usr/bin/env python3
"""Run the H/C_CNBR bridge through the proven Arm-H placement guard."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c1.scripts import (
    c1_orion_historical_cnbr_bridge_interleaved_ab as bridge,
)
from experiments.l1_balance import run_formal_ab_guarded as guarded


# Reuse the placement/lock orchestration unchanged; its only mutable target is
# the same frozen historical Arm-H collection.  Swap the read-only child runner
# and install the bridge-specific fixed-role CLI validator below.
guarded.ab_runner = bridge
guarded.RUNNER_SCRIPT = Path(bridge.__file__).resolve()


def validate_historical_cnbr_runner_contract(
    runner_cli: Sequence[str], orchestration_output: Path
) -> argparse.Namespace:
    """Validate the bridge's fixed arm roles without legacy CLI escape flags.

    ``run_formal_ab_guarded`` originally validated the historical runner's
    opt-in flags directly on its parsed namespace.  The H/C_CNBR bridge exposes
    no such flags: its arm factory freezes the permissions in code.  Validate
    those exact arm specifications instead, while retaining the guard's lock,
    collection, and output-directory checks.
    """

    parsed = bridge.parse_args(runner_cli)
    bridge.validate_args(parsed)
    if (
        parsed.benchmark_lock_fd is not None
        or parsed.benchmark_lock_token is not None
    ):
        raise ValueError(
            "runner benchmark-lock FD/token are owned exclusively by the wrapper"
        )
    if parsed.collection_a != guarded.FORMAL_ARM_A_COLLECTION:
        raise ValueError(
            "guarded historical H/C_CNBR bridge may mutate placement only for "
            "the frozen Arm H collection "
            f"{guarded.FORMAL_ARM_A_COLLECTION!r}"
        )

    arms = bridge.historical_cnbr_arm_specs(parsed)
    arm_a = arms["A"]
    arm_b = arms["B"]
    role_checks = {
        "arm_a_is_historical_scaling_layout": (
            arm_a.allow_historical_prepare_deployment
            and arm_a.allow_scaling_layout
            and not arm_a.allow_balance_layout
            and not arm_a.allow_l1_partition_layout
        ),
        "arm_b_is_current_l1_partition_layout": (
            not arm_b.allow_historical_prepare_deployment
            and not arm_b.allow_scaling_layout
            and not arm_b.allow_balance_layout
            and arm_b.allow_l1_partition_layout
        ),
    }
    if not all(role_checks.values()):
        raise ValueError(
            "historical H/C_CNBR guarded arm-role contract drifted: "
            f"{role_checks}"
        )

    expected_runner_output = (
        orchestration_output / guarded.RUNNER_OUTPUT_NAME
    ).resolve()
    actual_runner_output = Path(parsed.output_dir).expanduser().resolve()
    if actual_runner_output != expected_runner_output:
        raise ValueError(
            f"runner --output-dir must be {expected_runner_output}, "
            f"found {actual_runner_output}"
        )
    if actual_runner_output.exists():
        raise FileExistsError(actual_runner_output)
    return parsed


guarded.validate_runner_contract = validate_historical_cnbr_runner_contract


def validate_historical_cnbr_guarded_prerequisites(
    runner_args: argparse.Namespace,
) -> dict[str, object]:
    """Reject bad aggregate cost evidence before moving historical Arm H."""

    evidence = bridge.native_cnbr.cost_gate.validate_construction_cost_v4_pass(
        runner_args.construction_cost_audit
    )
    return {
        "status": "PASS",
        "record_type": "historical_h_cnbr_guarded_prerequisites",
        "construction_cost_v4": evidence,
    }


guarded.validate_guarded_prerequisites = (
    validate_historical_cnbr_guarded_prerequisites
)


if __name__ == "__main__":
    raise SystemExit(guarded.main())
