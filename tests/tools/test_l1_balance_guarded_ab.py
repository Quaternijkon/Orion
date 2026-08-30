from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "experiments/l1_balance/run_formal_ab_guarded.py"


def load_module():
    name = "l1_balance_guarded_ab_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_runner_inputs(tmp_path: Path, output_dir: Path) -> list[str]:
    dataset = tmp_path / "dataset.hdf5"
    topology = tmp_path / "topology.json"
    deployment = tmp_path / "deployment.json"
    prepare_a = tmp_path / "prepare-a.json"
    prepare_b = tmp_path / "prepare-b.json"
    layout_a = tmp_path / "layout-a"
    layout_b = tmp_path / "layout-b"
    layout_a.mkdir()
    layout_b.mkdir()
    artifact_a = layout_a / "generation-a.json"
    artifact_b = layout_b / "generation-b.json"
    for path in (
        dataset,
        topology,
        deployment,
        prepare_a,
        prepare_b,
        artifact_a,
        artifact_b,
    ):
        path.write_text("{}\n", encoding="utf-8")
    return [
        "--base-url",
        "http://controller:6333",
        "--hdf5-path",
        str(dataset),
        "--topology",
        str(topology),
        "--deployment-manifest",
        str(deployment),
        "--output-dir",
        str(output_dir / "runner"),
        "--collection-a",
        "orion_lb_r090_p24_20260825_v2",
        "--collection-b",
        "orion_l1_balance_candidate",
        "--artifact-a",
        str(artifact_a),
        "--artifact-b",
        str(artifact_b),
        "--layout-dir-a",
        str(layout_a),
        "--layout-dir-b",
        str(layout_b),
        "--prepare-manifest-a",
        str(prepare_a),
        "--prepare-manifest-b",
        str(prepare_b),
        "--allow-historical-prepare-deployment-a",
        "--allow-l1-partition-layout-b",
    ]


class FakeHeldLock:
    def evidence(self):
        return {
            "mode": "acquired",
            "path": "/tmp/benchmark.lock",
            "token_sha256": "a" * 64,
            "owner_pid": 111,
            "owner_kind": "orion_formal_ab_guarded",
            "process_pid": 111,
        }

    def inheritance_cli_arguments(self):
        return [
            "--benchmark-lock-fd",
            "9",
            "--benchmark-lock-token",
            "secret-token",
        ]

    def inheritance_pass_fds(self):
        return (9,)


def install_fake_guard(
    monkeypatch,
    module,
    events,
    *,
    runner_exit=0,
    runner_error=None,
    apply_fails=False,
    restore_fails=False,
):
    @contextlib.contextmanager
    def fake_lock(deployment_manifest, *, owner):
        del deployment_manifest, owner
        events.append("lock_enter")
        try:
            yield FakeHeldLock()
        finally:
            events.append("lock_exit")

    def snapshot_collection(*, base_url, collection, output):
        del base_url, collection
        events.append("snapshot")
        output.write_text('{"snapshot":"synthetic"}\n', encoding="utf-8")
        return {"snapshot_sha256": "b" * 64}

    def apply_prepare_placement(**kwargs):
        events.append("apply")
        if apply_fails:
            raise RuntimeError("synthetic partial apply failure")
        kwargs["proof_output"].write_text('{"apply":true}\n', encoding="utf-8")
        return {"proof_sha256": "c" * 64, "exact_target_verified": True}

    def restore_snapshot_placement(**kwargs):
        events.append("restore")
        if restore_fails:
            raise RuntimeError("synthetic restore failure")
        kwargs["proof_output"].write_text('{"restore":true}\n', encoding="utf-8")
        return {"proof_sha256": "d" * 64, "exact_snapshot_verified": True}

    def run_runner_subprocess(runner_cli, held_lock, signals, execution):
        del signals
        assert held_lock.inheritance_pass_fds() == (9,)
        events.append("runner")
        execution.started = True
        execution.pid = 222
        execution.stopped = True
        if runner_error is not None:
            raise runner_error
        execution.exit_code = runner_exit
        if runner_exit == 0:
            parsed = module.ab_runner.parse_args(runner_cli)
            runner_output = Path(parsed.output_dir)
            runner_output.mkdir(parents=True)
            parent = held_lock.evidence()
            (runner_output / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "benchmark_lock": {
                            **parent,
                            "mode": "inherited",
                            "process_pid": 222,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return runner_exit

    monkeypatch.setattr(module.benchmark_lock, "hold_benchmark_lock", fake_lock)
    monkeypatch.setattr(module.placement, "snapshot_collection", snapshot_collection)
    monkeypatch.setattr(
        module.placement, "apply_prepare_placement", apply_prepare_placement
    )
    monkeypatch.setattr(
        module.placement, "restore_snapshot_placement", restore_snapshot_placement
    )
    monkeypatch.setattr(module, "run_runner_subprocess", run_runner_subprocess)


def test_single_lock_scope_orders_snapshot_apply_runner_restore(monkeypatch, tmp_path):
    module = load_module()
    output = tmp_path / "guarded"
    runner_cli = write_runner_inputs(tmp_path, output)
    events: list[str] = []
    install_fake_guard(monkeypatch, module, events)

    result = module.run(
        module.parse_args(["--output-dir", str(output), "--", *runner_cli])
    )

    assert result == output.resolve()
    assert events == [
        "lock_enter",
        "snapshot",
        "apply",
        "runner",
        "restore",
        "lock_exit",
    ]
    record = json.loads((output / "orchestration.json").read_text())
    assert record["status"] == "PASS"
    assert record["single_canonical_lock_scope"] is True
    assert record["restore"]["exact_snapshot_verified"] is True
    assert record["runner_lock_inheritance"]["mode_expected_in_runner"] == "inherited"
    assert record["runner_lock_inheritance"]["verification"]["status"] == "PASS"
    assert record["recovery"]["written_before_apply"] is True
    assert record["recovery"]["cleared"] is True
    assert "secret-token" not in (output / "orchestration.json").read_text()


def test_runner_failure_still_restores_before_lock_release(monkeypatch, tmp_path):
    module = load_module()
    output = tmp_path / "guarded"
    runner_cli = write_runner_inputs(tmp_path, output)
    events: list[str] = []
    install_fake_guard(monkeypatch, module, events, runner_exit=7)

    with pytest.raises(module.RunnerProcessFailed, match="status 7"):
        module.run(
            module.parse_args(["--output-dir", str(output), "--", *runner_cli])
        )

    assert events[-3:] == ["runner", "restore", "lock_exit"]
    record = json.loads((output / "orchestration.json").read_text())
    assert record["status"] == "FAILED_RESTORED"
    assert record["runner_exit_code"] == 7
    assert record["primary_error"]["runner_exit_code"] == 7
    assert record["restore"]["exact_snapshot_verified"] is True


def test_partial_apply_failure_still_restores_snapshot(monkeypatch, tmp_path):
    module = load_module()
    output = tmp_path / "guarded"
    runner_cli = write_runner_inputs(tmp_path, output)
    events: list[str] = []
    install_fake_guard(monkeypatch, module, events, apply_fails=True)

    with pytest.raises(RuntimeError, match="partial apply failure"):
        module.run(
            module.parse_args(["--output-dir", str(output), "--", *runner_cli])
        )

    assert events == ["lock_enter", "snapshot", "apply", "restore", "lock_exit"]
    record = json.loads((output / "orchestration.json").read_text())
    assert record["status"] == "FAILED_RESTORED"
    assert record["apply"]["proof_sha256"] is None
    assert record["restore"]["exact_snapshot_verified"] is True
    assert record["runner_lock_inheritance"]["runner_started"] is False


def test_runner_interruption_still_restores_snapshot(monkeypatch, tmp_path):
    module = load_module()
    output = tmp_path / "guarded"
    runner_cli = write_runner_inputs(tmp_path, output)
    events: list[str] = []
    install_fake_guard(
        monkeypatch,
        module,
        events,
        runner_error=module.OrchestrationSignal(module.signal.SIGTERM),
    )

    with pytest.raises(module.OrchestrationSignal, match="SIGTERM"):
        module.run(
            module.parse_args(["--output-dir", str(output), "--", *runner_cli])
        )

    assert events[-3:] == ["runner", "restore", "lock_exit"]
    record = json.loads((output / "orchestration.json").read_text())
    assert record["status"] == "FAILED_RESTORED"
    assert record["primary_error"]["signal"] == module.signal.SIGTERM


def test_restore_failure_is_critical_and_recorded(monkeypatch, tmp_path):
    module = load_module()
    output = tmp_path / "guarded"
    runner_cli = write_runner_inputs(tmp_path, output)
    events: list[str] = []
    install_fake_guard(monkeypatch, module, events, restore_fails=True)

    with pytest.raises(module.RestoreFailed, match="restore failed"):
        module.run(
            module.parse_args(["--output-dir", str(output), "--", *runner_cli])
        )

    assert events[-2:] == ["restore", "lock_exit"]
    record = json.loads((output / "orchestration.json").read_text())
    assert record["status"] == "RESTORE_FAILED"
    assert record["restore_error"]["type"] == "RuntimeError"


def test_wrapper_rejects_user_supplied_inherited_lock_credentials(tmp_path):
    module = load_module()
    output = tmp_path / "guarded"
    runner_cli = write_runner_inputs(tmp_path, output)
    with pytest.raises(ValueError, match="owned exclusively by the wrapper"):
        module.run(
            module.parse_args(
                [
                    "--output-dir",
                    str(output),
                    "--",
                    *runner_cli,
                    "--benchmark-lock-fd",
                    "9",
                    "--benchmark-lock-token",
                    "forged",
                ]
            )
        )
    assert not output.exists()


def test_guarded_prerequisite_failure_occurs_before_output_and_lock(
    monkeypatch, tmp_path
):
    module = load_module()
    output = tmp_path / "guarded"
    runner_cli = write_runner_inputs(tmp_path, output)
    entered_lock = False

    def fail_prerequisite(_runner_args):
        raise ValueError("synthetic construction-cost failure")

    @contextlib.contextmanager
    def forbidden_lock(*_args, **_kwargs):
        nonlocal entered_lock
        entered_lock = True
        yield FakeHeldLock()

    monkeypatch.setattr(module, "validate_guarded_prerequisites", fail_prerequisite)
    monkeypatch.setattr(
        module.benchmark_lock, "hold_benchmark_lock", forbidden_lock
    )
    with pytest.raises(ValueError, match="construction-cost failure"):
        module.run(
            module.parse_args(["--output-dir", str(output), "--", *runner_cli])
        )
    assert entered_lock is False
    assert not output.exists()


@pytest.mark.parametrize(
    ("abbreviated_flag", "value"),
    [
        ("--benchmark-lock-f", "9"),
        ("--benchmark-lock-tok", "forged-secret"),
    ],
)
def test_wrapper_rejects_abbreviated_inherited_lock_credentials(
    tmp_path, abbreviated_flag, value
):
    module = load_module()
    output = tmp_path / "guarded"
    runner_cli = write_runner_inputs(tmp_path, output)
    with pytest.raises(ValueError, match="owned exclusively by the wrapper"):
        module.run(
            module.parse_args(
                [
                    "--output-dir",
                    str(output),
                    "--",
                    *runner_cli,
                    abbreviated_flag,
                    value,
                ]
            )
        )
    assert not output.exists()


def test_runner_subprocess_passes_same_fd_and_token_without_recording_it(monkeypatch):
    module = load_module()
    captured = {}

    class FakeProcess:
        pid = 12345

        def wait(self, timeout=None):
            del timeout
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    execution = module.RunnerExecution()
    result = module.run_runner_subprocess(
        ["--base-url", "http://x"],
        FakeHeldLock(),
        module.SignalController(),
        execution,
    )
    assert result == 0
    assert execution.started is True
    assert execution.pid == 12345
    assert execution.exit_code == 0
    assert captured["command"][-4:] == [
        "--benchmark-lock-fd",
        "9",
        "--benchmark-lock-token",
        "secret-token",
    ]
    assert captured["kwargs"]["pass_fds"] == (9,)
    assert captured["kwargs"]["start_new_session"] is True


def test_first_signal_latches_and_second_cannot_interrupt_runner_cleanup(monkeypatch):
    module = load_module()
    calls = []
    controller = module.SignalController()

    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout == 1.0 and len(calls) == 1:
                controller._handler(module.signal.SIGINT, None)
                controller._handler(module.signal.SIGTERM, None)
                raise module.subprocess.TimeoutExpired("runner", timeout)
            self.stopped = True
            return 0

    monkeypatch.setattr(
        module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda pid, signum: calls.append(("killpg", pid, signum)),
    )
    with controller:
        with pytest.raises(module.OrchestrationSignal, match="SIGINT"):
            module.run_runner_subprocess(
                [], FakeHeldLock(), controller, module.RunnerExecution()
            )
    assert controller.pending_signal == module.signal.SIGINT
    assert ("killpg", 12345, module.signal.SIGTERM) in calls
    assert calls[-1] == ("wait", 30.0)


def test_latched_signal_prevents_reentrant_exception_before_defer_scope():
    module = load_module()
    controller = module.SignalController()
    with controller:
        controller._handler(module.signal.SIGINT, None)
        controller._handler(module.signal.SIGTERM, None)
        with pytest.raises(module.OrchestrationSignal, match="SIGINT"):
            controller.raise_if_pending()
    assert controller.pending_signal == module.signal.SIGINT


def test_runner_lock_manifest_must_match_parent_evidence(tmp_path):
    module = load_module()
    runner_output = tmp_path / "runner"
    runner_output.mkdir()
    parent = FakeHeldLock().evidence()
    (runner_output / "run_manifest.json").write_text(
        json.dumps(
            {
                "benchmark_lock": {
                    **parent,
                    "mode": "inherited",
                    "token_sha256": "f" * 64,
                    "process_pid": 222,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not prove inherited"):
        module.verify_runner_lock_evidence(runner_output, parent)


def test_runner_cost_prerequisite_must_match_guarded_sha(tmp_path):
    module = load_module()
    runner_output = tmp_path / "runner"
    runner_output.mkdir()
    (runner_output / "run_manifest.json").write_text(
        json.dumps(
            {
                "adoption_prerequisites": {
                    "status": "PASS",
                    "construction_cost_v4": {
                        "audit_sha256": "b" * 64,
                        "aggregate_status": (
                            "COMPLETE_FRESH_PROJECTED_INCREMENTAL_GATE_PASS"
                        ),
                        "datasets": {"sift1m": {}, "glove-200-angular": {}},
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    expected = {
        "status": "PASS",
        "construction_cost_v4": {
            "audit_sha256": "a" * 64,
            "aggregate_status": (
                "COMPLETE_FRESH_PROJECTED_INCREMENTAL_GATE_PASS"
            ),
            "datasets": {"sift1m": {}, "glove-200-angular": {}},
        },
    }
    with pytest.raises(RuntimeError, match="did not preserve"):
        module.verify_runner_prerequisite_evidence(runner_output, expected)
