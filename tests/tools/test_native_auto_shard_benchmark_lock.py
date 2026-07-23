from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module():
    from tools import native_auto_shard_benchmark_lock

    return native_auto_shard_benchmark_lock


def write_manifest(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    manifest = run_dir / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return manifest


def test_canonical_lock_is_manifest_sibling_and_run_directories_are_independent(
    tmp_path,
):
    module = load_module()
    first_manifest = write_manifest(tmp_path / "run-a")
    second_manifest = write_manifest(tmp_path / "run-b")

    assert module.canonical_benchmark_lock_path(first_manifest) == (
        first_manifest.parent / "benchmark.lock"
    )
    with module.hold_benchmark_lock(first_manifest) as first:
        with module.hold_benchmark_lock(second_manifest) as second:
            assert first.path != second.path
            assert first.mode == second.mode == "acquired"


def test_second_direct_owner_is_rejected_and_exception_releases_lock(tmp_path):
    module = load_module()
    manifest = write_manifest(tmp_path / "run")

    with pytest.raises(LookupError):
        with module.hold_benchmark_lock(manifest, owner={"kind": "first"}) as held:
            assert stat.S_IMODE(held.path.stat().st_mode) == 0o600
            with pytest.raises(module.BenchmarkLockError, match="already held"):
                with module.hold_benchmark_lock(manifest):
                    pass
            raise LookupError("release through context-manager exception")

    with module.hold_benchmark_lock(manifest, owner={"kind": "second"}) as held:
        assert held.owner["kind"] == "second"


def test_parent_child_explicit_fd_and_token_inheritance_does_not_unlock_parent(
    tmp_path,
):
    module = load_module()
    manifest = write_manifest(tmp_path / "run")
    child = """
import json
import sys
from tools import native_auto_shard_benchmark_lock as lock
with lock.hold_benchmark_lock(
    sys.argv[1],
    inherited_fd=int(sys.argv[2]),
    inherited_token=sys.argv[3],
) as held:
    print(json.dumps(held.evidence(), sort_keys=True))
"""

    with module.hold_benchmark_lock(
        manifest, owner={"kind": "matrix-parent"}
    ) as parent:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                child,
                str(manifest),
                str(parent.fd),
                parent.token,
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            pass_fds=parent.inheritance_pass_fds(),
        )
        assert result.returncode == 0, result.stderr
        evidence = json.loads(result.stdout)
        assert evidence["mode"] == "inherited"
        assert evidence["owner_kind"] == "matrix-parent"

        # The child only closed its descriptor; it did not unlock the parent's
        # shared open-file description.
        with pytest.raises(module.BenchmarkLockError, match="already held"):
            with module.hold_benchmark_lock(manifest):
                pass

    with module.hold_benchmark_lock(manifest):
        pass


def test_inherited_identity_is_fail_closed(tmp_path):
    module = load_module()
    manifest = write_manifest(tmp_path / "run")

    with module.hold_benchmark_lock(manifest) as parent:
        duplicate = os.dup(parent.fd)
        try:
            with pytest.raises(module.BenchmarkLockError, match="token"):
                with module.hold_benchmark_lock(
                    manifest,
                    inherited_fd=duplicate,
                    inherited_token="wrong-token",
                ):
                    pass
        finally:
            os.close(duplicate)

    with pytest.raises(module.BenchmarkLockError, match="both FD and token"):
        with module.hold_benchmark_lock(manifest, inherited_fd=99):
            pass


def test_stale_open_file_is_not_accepted_as_an_inherited_lock(tmp_path):
    module = load_module()
    manifest = write_manifest(tmp_path / "run")
    path = module.canonical_benchmark_lock_path(manifest)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.write(fd, b'{"token":"stale"}\n')
        with pytest.raises(module.BenchmarkLockError, match="not holding"):
            with module.hold_benchmark_lock(
                manifest,
                inherited_fd=fd,
                inherited_token="stale",
            ):
                pass
    finally:
        os.close(fd)


def test_cli_plumbing_is_explicit_and_ephemeral_values_are_not_recorded():
    module = load_module()
    parser = argparse.ArgumentParser()
    module.add_cli_arguments(parser)
    args = parser.parse_args(
        [
            module.LOCK_FD_ARGUMENT,
            "17",
            module.LOCK_TOKEN_ARGUMENT,
            "secret-token",
        ]
    )
    assert args.benchmark_lock_fd == 17
    assert args.benchmark_lock_token == "secret-token"
    assert module.strip_cli_arguments(
        [
            "python",
            "tool.py",
            module.LOCK_FD_ARGUMENT,
            "17",
            "--ordinary",
            "value",
            module.LOCK_TOKEN_ARGUMENT,
            "secret-token",
        ]
    ) == ["python", "tool.py", "--ordinary", "value"]
    assert module.strip_cli_arguments(
        [
            "python",
            "tool.py",
            f"{module.LOCK_FD_ARGUMENT}=17",
            "--ordinary=value",
            f"{module.LOCK_TOKEN_ARGUMENT}=secret-token",
        ]
    ) == ["python", "tool.py", "--ordinary=value"]
    assert module.public_namespace(args) == {}
