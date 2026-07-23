#!/usr/bin/env python3

"""Fail-closed, run-scoped serialization for distributed benchmarks.

The canonical lock file lives beside the deployment manifest.  A benchmark
parent may explicitly pass the held file descriptor and its random token to a
child process.  The child adopts that open-file description instead of trying
to flock the path again, which avoids self-deadlock while keeping unrelated
direct invocations fail closed.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


LOCK_FILENAME = "benchmark.lock"
LOCK_FD_ARGUMENT = "--benchmark-lock-fd"
LOCK_TOKEN_ARGUMENT = "--benchmark-lock-token"
LOCK_ARGUMENTS = frozenset((LOCK_FD_ARGUMENT, LOCK_TOKEN_ARGUMENT))


class BenchmarkLockError(RuntimeError):
    """Raised when exclusive benchmark ownership cannot be proven."""


def canonical_benchmark_lock_path(deployment_manifest: str | Path) -> Path:
    manifest = Path(deployment_manifest).expanduser().resolve(strict=True)
    if not manifest.is_file():
        raise FileNotFoundError(f"deployment manifest is not a file: {manifest}")
    return manifest.parent / LOCK_FILENAME


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        LOCK_FD_ARGUMENT,
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        LOCK_TOKEN_ARGUMENT,
        default=None,
        help=argparse.SUPPRESS,
    )


def strip_cli_arguments(arguments: Sequence[str]) -> list[str]:
    """Remove ephemeral inherited-lock arguments from recorded commands."""
    cleaned: list[str] = []
    index = 0
    while index < len(arguments):
        value = str(arguments[index])
        if value in LOCK_ARGUMENTS:
            if index + 1 >= len(arguments):
                raise ValueError(f"{value} requires a value")
            index += 2
            continue
        cleaned.append(value)
        index += 1
    return cleaned


def public_namespace(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(arguments).items()
        if key not in {"benchmark_lock_fd", "benchmark_lock_token"}
    }


def _read_owner_payload(fd: int) -> dict[str, Any] | None:
    try:
        size = os.fstat(fd).st_size
        encoded = os.pread(fd, max(size, 1), 0)
    except OSError:
        return None
    if not encoded.strip():
        return None
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_owner_payload(fd: int, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    os.ftruncate(fd, 0)
    os.pwrite(fd, encoded, 0)
    os.fsync(fd)


def _open_lock_file(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise BenchmarkLockError(f"unable to open benchmark lock {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise BenchmarkLockError(
                f"benchmark lock is not a regular file: {path}"
            )
        os.fchmod(fd, 0o600)
        os.set_inheritable(fd, False)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _format_owner(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "<unavailable>"
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass
class HeldBenchmarkLock:
    path: Path
    fd: int
    token: str
    mode: str
    owner: dict[str, Any]
    _closed: bool = False

    def inheritance_cli_arguments(self) -> list[str]:
        if self._closed:
            raise BenchmarkLockError("cannot inherit a closed benchmark lock")
        return [
            LOCK_FD_ARGUMENT,
            str(self.fd),
            LOCK_TOKEN_ARGUMENT,
            self.token,
        ]

    def inheritance_pass_fds(self) -> tuple[int, ...]:
        if self._closed:
            raise BenchmarkLockError("cannot inherit a closed benchmark lock")
        return (self.fd,)

    def evidence(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "path": str(self.path),
            "mode": self.mode,
            "token_sha256": hashlib.sha256(self.token.encode("utf-8")).hexdigest(),
            "owner_pid": self.owner.get("pid"),
            "owner_kind": self.owner.get("kind"),
            "process_pid": os.getpid(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Closing is sufficient for an owner and is mandatory for an adopted
        # descriptor.  Explicit LOCK_UN could release the shared open-file
        # description while an inherited child is still winding down.
        os.close(self.fd)


def _validate_inherited_lock(
    path: Path,
    inherited_fd: int,
    inherited_token: str,
) -> HeldBenchmarkLock:
    if inherited_fd < 3:
        raise BenchmarkLockError("inherited benchmark lock FD must be at least 3")
    if not inherited_token:
        raise BenchmarkLockError("inherited benchmark lock token must be non-empty")
    try:
        inherited_stat = os.fstat(inherited_fd)
        path_stat = path.stat()
    except OSError as exc:
        raise BenchmarkLockError(
            f"inherited benchmark lock FD is unavailable: {inherited_fd}"
        ) from exc
    if not stat.S_ISREG(inherited_stat.st_mode):
        raise BenchmarkLockError("inherited benchmark lock FD is not a regular file")
    if (inherited_stat.st_dev, inherited_stat.st_ino) != (
        path_stat.st_dev,
        path_stat.st_ino,
    ):
        raise BenchmarkLockError(
            "inherited benchmark lock FD does not refer to the canonical lock: "
            f"{path}"
        )

    owner = _read_owner_payload(inherited_fd)
    if owner is None or owner.get("token") != inherited_token:
        raise BenchmarkLockError(
            "inherited benchmark lock token does not match the canonical lock owner"
        )

    # A separately opened file description must conflict with the inherited
    # one.  If it can acquire the lock, the supplied FD is merely an open stale
    # lock file and does not prove exclusive benchmark ownership.
    probe_fd = _open_lock_file(path)
    try:
        try:
            fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe_fd, fcntl.LOCK_UN)
            raise BenchmarkLockError(
                "inherited benchmark lock FD is not holding an exclusive lock"
            )
    finally:
        os.close(probe_fd)

    # pass_fds makes the descriptor inheritable for exec.  Stop the adopted
    # child from leaking the lock into Cargo, SSH, Docker, or unrelated helpers.
    os.set_inheritable(inherited_fd, False)
    return HeldBenchmarkLock(
        path=path,
        fd=inherited_fd,
        token=inherited_token,
        mode="inherited",
        owner=owner,
    )


def _acquire_lock(path: Path, owner: dict[str, Any] | None) -> HeldBenchmarkLock:
    fd = _open_lock_file(path)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            current_owner = _read_owner_payload(fd)
            raise BenchmarkLockError(
                f"benchmark lock is already held: {path}; "
                f"owner={_format_owner(current_owner)}"
            ) from exc

        token = secrets.token_hex(32)
        owner_payload = {
            **(owner or {}),
            "schema_version": 1,
            "state": "held",
            "token": token,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": strip_cli_arguments(sys.argv),
        }
        _write_owner_payload(fd, owner_payload)
        return HeldBenchmarkLock(
            path=path,
            fd=fd,
            token=token,
            mode="acquired",
            owner=owner_payload,
        )
    except BaseException:
        os.close(fd)
        raise


@contextlib.contextmanager
def hold_benchmark_lock(
    deployment_manifest: str | Path,
    *,
    inherited_fd: int | None = None,
    inherited_token: str | None = None,
    owner: dict[str, Any] | None = None,
) -> Iterator[HeldBenchmarkLock]:
    path = canonical_benchmark_lock_path(deployment_manifest)
    if (inherited_fd is None) != (inherited_token is None):
        raise BenchmarkLockError(
            "inherited benchmark lock requires both FD and token"
        )
    held = (
        _acquire_lock(path, owner)
        if inherited_fd is None
        else _validate_inherited_lock(path, inherited_fd, str(inherited_token))
    )
    try:
        yield held
    finally:
        held.close()


def hold_from_args(
    arguments: argparse.Namespace,
    deployment_manifest: str | Path,
    *,
    owner: dict[str, Any] | None = None,
) -> contextlib.AbstractContextManager[HeldBenchmarkLock]:
    return hold_benchmark_lock(
        deployment_manifest,
        inherited_fd=getattr(arguments, "benchmark_lock_fd", None),
        inherited_token=getattr(arguments, "benchmark_lock_token", None),
        owner=owner,
    )
