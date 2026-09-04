#!/usr/bin/env python3
"""Run the formal Orion A/B inside one placement-and-benchmark lock scope.

The parent process acquires the canonical deployment benchmark lock exactly
once, snapshots Arm A's current numeric-shard placement, applies the frozen
prepare-manifest placement, starts the existing read-only A/B runner with the
same lock FD/token, and restores the exact snapshot in ``finally``.

This wrapper may move numeric shards only for the explicitly frozen Arm A
collection.  It never creates/deletes collections, installs artifacts, or
changes container resources.  SIGINT, SIGTERM, and SIGHUP are latched; the
current placement operation finishes safely, a live runner is terminated and
reaped, and restore is never asynchronously interrupted.
SIGKILL and host failure cannot be handled in-process, so the immutable
snapshot remains the manual recovery authority in those cases.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import FrameType, TracebackType
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c1.scripts import (  # noqa: E402
    c1_orion_balance_interleaved_ab as ab_runner,
)
from experiments.l1_balance import formal_ab_placement as placement  # noqa: E402
from tools import native_auto_shard_benchmark_lock as benchmark_lock  # noqa: E402


RUNNER_SCRIPT = Path(ab_runner.__file__).resolve()
FORMAL_ARM_A_COLLECTION = "orion_lb_r090_p24_20260825_v2"
RUNNER_OUTPUT_NAME = "runner"
SNAPSHOT_NAME = "arm-a-original-placement.json"
APPLY_PROOF_NAME = "arm-a-apply-proof.json"
RESTORE_PROOF_NAME = "arm-a-restore-proof.json"
ORCHESTRATION_NAME = "orchestration.json"
RECOVERY_NAME = "recovery.json"
RECOVERY_CLEARED_NAME = "recovery-cleared.json"
HANDLED_SIGNALS = tuple(
    value
    for value in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None))
    if value is not None
)


class RunnerProcessFailed(RuntimeError):
    def __init__(self, exit_code: int):
        self.exit_code = int(exit_code)
        super().__init__(f"formal A/B runner exited with status {self.exit_code}")


class OrchestrationSignal(RuntimeError):
    def __init__(self, signum: int):
        self.signum = int(signum)
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        super().__init__(f"guarded formal A/B interrupted by {name}")


class RestoreFailed(RuntimeError):
    pass


@dataclass
class RunnerExecution:
    started: bool = False
    pid: int | None = None
    exit_code: int | None = None
    stopped: bool = False


@dataclass
class SignalController:
    previous: dict[int, Any] | None = None
    pending_signal: int | None = None

    def _handler(self, signum: int, _frame: FrameType | None) -> None:
        if self.pending_signal is None:
            self.pending_signal = int(signum)

    def __enter__(self) -> "SignalController":
        self.previous = {
            int(signum): signal.signal(signum, self._handler)
            for signum in HANDLED_SIGNALS
        }
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        assert self.previous is not None
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)

    def raise_if_pending(self) -> None:
        if self.pending_signal is not None:
            raise OrchestrationSignal(self.pending_signal)


def utc_timestamp() -> str:
    return ab_runner.utc_timestamp()


def fsync_file_and_parent(path: Path) -> None:
    file_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(file_fd)
    finally:
        os.close(file_fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_file_and_parent(path)


def error_record(error: BaseException | None) -> dict[str, Any] | None:
    if error is None:
        return None
    record: dict[str, Any] = {
        "type": type(error).__name__,
        "repr": repr(error),
    }
    if isinstance(error, RunnerProcessFailed):
        record["runner_exit_code"] = error.exit_code
    if isinstance(error, OrchestrationSignal):
        record["signal"] = error.signum
    return record


def normalize_runner_cli(raw: Sequence[str]) -> list[str]:
    values = [str(value) for value in raw]
    if values and values[0] == "--":
        values = values[1:]
    if not values:
        raise ValueError("runner arguments must follow a -- separator")
    for value in values:
        if value in benchmark_lock.LOCK_ARGUMENTS or any(
            value.startswith(f"{argument}=")
            for argument in benchmark_lock.LOCK_ARGUMENTS
        ):
            raise ValueError(
                "runner benchmark-lock FD/token are owned exclusively by the wrapper"
            )
    return values


def validate_output_directory(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output == REPO_ROOT or REPO_ROOT in output.parents:
        raise ValueError("guarded formal A/B output must be outside the repository")
    if output.exists():
        raise FileExistsError(output)
    return output


def validate_runner_contract(
    runner_cli: Sequence[str], orchestration_output: Path
) -> argparse.Namespace:
    parsed = ab_runner.parse_args(runner_cli)
    ab_runner.validate_args(parsed)
    if (
        parsed.benchmark_lock_fd is not None
        or parsed.benchmark_lock_token is not None
    ):
        raise ValueError(
            "runner benchmark-lock FD/token are owned exclusively by the wrapper"
        )
    if parsed.collection_a != FORMAL_ARM_A_COLLECTION:
        raise ValueError(
            "guarded formal A/B may mutate placement only for the frozen Arm A "
            f"collection {FORMAL_ARM_A_COLLECTION!r}"
        )
    if not parsed.allow_historical_prepare_deployment_a:
        raise ValueError("frozen Arm A requires its explicit historical-prepare flag")
    if not parsed.allow_l1_partition_layout_b:
        raise ValueError("Arm B must be the explicitly authorized L1 partition layout")
    if parsed.allow_balance_layout_a or parsed.allow_balance_layout_b:
        raise ValueError("legacy balance-layout exceptions are not allowed")
    if parsed.allow_scaling_layout_b:
        raise ValueError("Arm B must not use the legacy scaling-layout exception")
    expected_runner_output = (orchestration_output / RUNNER_OUTPUT_NAME).resolve()
    actual_runner_output = Path(parsed.output_dir).expanduser().resolve()
    if actual_runner_output != expected_runner_output:
        raise ValueError(
            f"runner --output-dir must be {expected_runner_output}, "
            f"found {actual_runner_output}"
        )
    if actual_runner_output.exists():
        raise FileExistsError(actual_runner_output)
    return parsed


def validate_guarded_prerequisites(
    _runner_args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Comparison-specific hook installed by stricter guarded entrypoints."""

    return None


def terminate_runner_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if process.poll() is None:
            process.terminate()
    try:
        process.wait(timeout=30.0)
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.poll() is None:
                process.kill()
        process.wait()


def run_runner_subprocess(
    runner_cli: Sequence[str],
    held_lock: benchmark_lock.HeldBenchmarkLock,
    signals: SignalController,
    execution: RunnerExecution,
) -> int:
    command = [
        sys.executable,
        str(RUNNER_SCRIPT),
        *runner_cli,
        *held_lock.inheritance_cli_arguments(),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        pass_fds=held_lock.inheritance_pass_fds(),
        start_new_session=True,
    )
    execution.started = True
    execution.pid = int(process.pid)
    try:
        while True:
            if signals.pending_signal is not None:
                terminate_runner_process(process)
                execution.stopped = process.poll() is not None
                raise OrchestrationSignal(signals.pending_signal)
            try:
                execution.exit_code = int(process.wait(timeout=1.0))
            except subprocess.TimeoutExpired:
                continue
            execution.stopped = True
            signals.raise_if_pending()
            return execution.exit_code
    except BaseException:
        if not execution.stopped:
            terminate_runner_process(process)
            execution.stopped = process.poll() is not None
        raise


def verify_runner_lock_evidence(
    runner_output: Path,
    parent_evidence: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = runner_output / "run_manifest.json"
    manifest = ab_runner.load_json_object(manifest_path)
    child = manifest.get("benchmark_lock")
    if not isinstance(child, dict):
        raise RuntimeError("runner manifest lacks benchmark-lock evidence")
    checks = {
        "mode_inherited": child.get("mode") == "inherited",
        "same_path": child.get("path") == parent_evidence.get("path"),
        "same_token_sha256": child.get("token_sha256")
        == parent_evidence.get("token_sha256"),
        "same_owner_pid": child.get("owner_pid")
        == parent_evidence.get("owner_pid"),
        "same_owner_kind": child.get("owner_kind")
        == parent_evidence.get("owner_kind"),
        "distinct_child_process": (
            isinstance(child.get("process_pid"), int)
            and not isinstance(child.get("process_pid"), bool)
            and child.get("process_pid") != parent_evidence.get("process_pid")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "runner did not prove inherited canonical benchmark lock: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "status": "PASS",
        "manifest_path": str(manifest_path),
        "manifest_sha256": placement.sha256_path(manifest_path),
        "checks": checks,
        "runner_evidence": child,
    }


def verify_runner_prerequisite_evidence(
    runner_output: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = runner_output / "run_manifest.json"
    manifest = ab_runner.load_json_object(manifest_path)
    child = manifest.get("adoption_prerequisites")
    expected_cost = expected.get("construction_cost_v4")
    if not isinstance(child, Mapping) or not isinstance(expected_cost, Mapping):
        raise RuntimeError("runner manifest lacks guarded prerequisite evidence")
    child_cost = child.get("construction_cost_v4")
    if not isinstance(child_cost, Mapping):
        raise RuntimeError("runner manifest lacks construction-cost-v4 evidence")
    checks = {
        "child_prerequisites_pass": child.get("status") == "PASS",
        "same_cost_audit_sha256": child_cost.get("audit_sha256")
        == expected_cost.get("audit_sha256"),
        "same_cost_aggregate_status": child_cost.get("aggregate_status")
        == expected_cost.get("aggregate_status"),
        "same_dual_dataset_gate": child_cost.get("datasets")
        == expected_cost.get("datasets"),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "runner did not preserve guarded prerequisite evidence: "
            + json.dumps(checks, sort_keys=True)
        )
    return {"status": "PASS", "checks": checks}


def orchestration_status(
    *,
    primary_error: BaseException | None,
    restore_error: BaseException | None,
    snapshot_created: bool,
    restore_record: dict[str, Any] | None,
) -> str:
    if restore_error is not None:
        return "RESTORE_FAILED"
    if primary_error is not None:
        return (
            "FAILED_RESTORED"
            if restore_record is not None
            else "FAILED_BEFORE_RESTORE"
        )
    if not snapshot_created or restore_record is None:
        return "INCOMPLETE"
    return "PASS"


def run(args: argparse.Namespace) -> Path:
    runner_cli = normalize_runner_cli(args.runner_args)
    output_dir = validate_output_directory(args.output_dir)
    runner_args = validate_runner_contract(runner_cli, output_dir)
    guarded_prerequisites = validate_guarded_prerequisites(runner_args)
    if guarded_prerequisites is not None and (
        not isinstance(guarded_prerequisites, dict)
        or guarded_prerequisites.get("status") != "PASS"
    ):
        raise ValueError("guarded A/B prerequisites did not pass")
    if args.timeout_sec <= 0.0 or args.poll_interval_sec <= 0.0:
        raise ValueError("placement timeout and poll interval must be positive")

    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_path = output_dir / SNAPSHOT_NAME
    apply_path = output_dir / APPLY_PROOF_NAME
    restore_path = output_dir / RESTORE_PROOF_NAME
    orchestration_path = output_dir / ORCHESTRATION_NAME
    recovery_path = output_dir / RECOVERY_NAME
    recovery_cleared_path = output_dir / RECOVERY_CLEARED_NAME

    snapshot: dict[str, Any] | None = None
    apply_record: dict[str, Any] | None = None
    restore_record: dict[str, Any] | None = None
    runner_exit_code: int | None = None
    runner_execution = RunnerExecution()
    runner_lock_verification: dict[str, Any] | None = None
    runner_prerequisite_verification: dict[str, Any] | None = None
    primary_error: BaseException | None = None
    primary_traceback: TracebackType | None = None
    restore_error: BaseException | None = None
    lock_evidence: dict[str, Any] | None = None

    with SignalController() as signals:
        with benchmark_lock.hold_benchmark_lock(
            runner_args.deployment_manifest,
            owner={
                "kind": "orion_formal_ab_guarded",
                "collection_a": runner_args.collection_a,
                "collection_b": runner_args.collection_b,
                "output_dir": str(output_dir),
            },
        ) as held_lock:
            lock_evidence = held_lock.evidence()
            try:
                snapshot = placement.snapshot_collection(
                    base_url=runner_args.base_url,
                    collection=runner_args.collection_a,
                    output=snapshot_path,
                )
                fsync_file_and_parent(snapshot_path)
                snapshot_file_sha256 = placement.sha256_path(snapshot_path)
                write_json(
                    recovery_path,
                    {
                        "schema_version": 1,
                        "record_type": "orion_formal_ab_recovery_checkpoint",
                        "created_at": utc_timestamp(),
                        "status": "RECOVERY_REQUIRED_UNTIL_RESTORE_VERIFIED",
                        "base_url": runner_args.base_url,
                        "collection": runner_args.collection_a,
                        "deployment_manifest": {
                            "path": str(
                                runner_args.deployment_manifest.expanduser().resolve()
                            ),
                            "sha256": placement.sha256_path(
                                runner_args.deployment_manifest.expanduser().resolve()
                            ),
                        },
                        "snapshot": {
                            "path": str(snapshot_path),
                            "snapshot_sha256": snapshot["snapshot_sha256"],
                            "file_sha256": snapshot_file_sha256,
                        },
                        "benchmark_lock": lock_evidence,
                        "manual_recovery_tool": str(
                            Path(placement.__file__).resolve()
                        ),
                    },
                )
                signals.raise_if_pending()
                apply_record = placement.apply_prepare_placement(
                    base_url=runner_args.base_url,
                    collection=runner_args.collection_a,
                    snapshot_path=snapshot_path,
                    snapshot_sha256=str(snapshot["snapshot_sha256"]),
                    prepare_manifest=runner_args.prepare_manifest_a,
                    proof_output=apply_path,
                    dry_run=False,
                    transfer_method=args.transfer_method,
                    timeout_sec=args.timeout_sec,
                    poll_interval_sec=args.poll_interval_sec,
                )
                signals.raise_if_pending()
                runner_exit_code = run_runner_subprocess(
                    runner_cli, held_lock, signals, runner_execution
                )
                if runner_exit_code != 0:
                    raise RunnerProcessFailed(runner_exit_code)
                runner_lock_verification = verify_runner_lock_evidence(
                    Path(runner_args.output_dir).expanduser().resolve(),
                    lock_evidence,
                )
                if guarded_prerequisites is not None:
                    runner_prerequisite_verification = (
                        verify_runner_prerequisite_evidence(
                            Path(runner_args.output_dir).expanduser().resolve(),
                            guarded_prerequisites,
                        )
                    )
                signals.raise_if_pending()
            except BaseException as error:
                primary_error = error
                primary_traceback = error.__traceback__
            finally:
                if snapshot is not None:
                    if runner_execution.started and not runner_execution.stopped:
                        restore_error = RuntimeError(
                            "placement restore skipped because runner process exit "
                            "was not proven; use the durable recovery checkpoint"
                        )
                    else:
                        try:
                            restore_record = placement.restore_snapshot_placement(
                                base_url=runner_args.base_url,
                                collection=runner_args.collection_a,
                                snapshot_path=snapshot_path,
                                snapshot_sha256=str(snapshot["snapshot_sha256"]),
                                proof_output=restore_path,
                                dry_run=False,
                                transfer_method=args.transfer_method,
                                timeout_sec=args.timeout_sec,
                                poll_interval_sec=args.poll_interval_sec,
                            )
                            if recovery_path.is_file():
                                write_json(
                                    recovery_cleared_path,
                                    {
                                        "schema_version": 1,
                                        "record_type": (
                                            "orion_formal_ab_"
                                            "recovery_checkpoint_cleared"
                                        ),
                                        "created_at": utc_timestamp(),
                                        "status": "RESTORED",
                                        "recovery_path": str(recovery_path),
                                        "recovery_sha256": placement.sha256_path(
                                            recovery_path
                                        ),
                                        "restore_proof_path": str(restore_path),
                                        "restore_proof_sha256": restore_record[
                                            "proof_sha256"
                                        ],
                                        "exact_snapshot_verified": restore_record[
                                            "exact_snapshot_verified"
                                        ],
                                    },
                                )
                        except BaseException as error:
                            restore_error = error

            if signals.pending_signal is not None and primary_error is None:
                primary_error = OrchestrationSignal(signals.pending_signal)
                primary_traceback = primary_error.__traceback__

            record = {
                "schema_version": 1,
                "record_type": "orion_formal_ab_guarded_orchestration",
                "created_at": utc_timestamp(),
                "status": orchestration_status(
                    primary_error=primary_error,
                    restore_error=restore_error,
                    snapshot_created=snapshot is not None,
                    restore_record=restore_record,
                ),
                "single_canonical_lock_scope": True,
                "benchmark_lock": lock_evidence,
                "runner_public_command": [
                    sys.executable,
                    str(RUNNER_SCRIPT),
                    *runner_cli,
                ],
                "runner_lock_inheritance": {
                    "runner_started": runner_execution.started,
                    "runner_pid": runner_execution.pid,
                    "runner_stopped": runner_execution.stopped,
                    "fd_passed": runner_execution.started,
                    "token_recorded": False,
                    "mode_expected_in_runner": "inherited",
                    "verification": runner_lock_verification,
                },
                "runner_exit_code": runner_exit_code,
                "guarded_prerequisites": guarded_prerequisites,
                "runner_prerequisite_verification": (
                    runner_prerequisite_verification
                ),
                "arm_a_collection": runner_args.collection_a,
                "arm_b_collection": runner_args.collection_b,
                "snapshot": {
                    "path": str(snapshot_path),
                    "snapshot_sha256": (
                        snapshot.get("snapshot_sha256")
                        if snapshot is not None
                        else None
                    ),
                    "file_sha256": (
                        placement.sha256_path(snapshot_path)
                        if snapshot_path.is_file()
                        else None
                    ),
                },
                "recovery": {
                    "path": str(recovery_path),
                    "written_before_apply": recovery_path.is_file(),
                    "file_sha256": (
                        placement.sha256_path(recovery_path)
                        if recovery_path.is_file()
                        else None
                    ),
                    "cleared_path": str(recovery_cleared_path),
                    "cleared": recovery_cleared_path.is_file(),
                },
                "apply": {
                    "path": str(apply_path),
                    "proof_sha256": (
                        apply_record.get("proof_sha256")
                        if apply_record is not None
                        else None
                    ),
                },
                "restore": {
                    "attempted": snapshot is not None,
                    "path": str(restore_path),
                    "proof_sha256": (
                        restore_record.get("proof_sha256")
                        if restore_record is not None
                        else None
                    ),
                    "exact_snapshot_verified": (
                        restore_record.get("exact_snapshot_verified")
                        if restore_record is not None
                        else False
                    ),
                },
                "latched_signal": signals.pending_signal,
                "primary_error": error_record(primary_error),
                "restore_error": error_record(restore_error),
                "mutation_scope": "arm_a_numeric_shard_placement_only",
                "forbidden_mutations": [
                    "unrelated_collection_mutation",
                    "collection_create_delete_or_update",
                    "artifact_install_or_activation",
                    "container_resource_change",
                ],
            }
            write_json(orchestration_path, record)

            if restore_error is not None:
                message = "formal A/B placement restore failed"
                if primary_error is not None:
                    message += f" after primary failure {primary_error!r}"
                raise RestoreFailed(message) from restore_error
            if primary_error is not None:
                raise primary_error.with_traceback(primary_traceback)

    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transfer-method", choices=("snapshot",), default="snapshot")
    parser.add_argument("--timeout-sec", type=float, default=3600.0)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(json.dumps({"output_dir": str(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        RestoreFailed,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
