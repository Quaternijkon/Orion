"""Testbed discovery and CPU accounting for the containerized Qdrant cluster.

The measurement host runs every Qdrant peer as a Docker container pinned to a
cpuset, so CPU accounting is read from cgroup v2 rather than sampled with a
profiler. All sampling happens outside the measured request path.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

CGROUP_SLICE = Path("/sys/fs/cgroup/system.slice")
PROC_STAT = Path("/proc/stat")
QDRANT_HTTP_PORT = "6333/tcp"
DEFAULT_NAME_FILTER = "qdrant-controller"


class TestbedError(RuntimeError):
    """Raised when the testbed cannot be described precisely enough to measure."""


def parse_cpuset(spec: str) -> tuple[int, ...]:
    """Expand a cpuset specification such as ``0-7,16-23`` into core ids."""
    if not spec.strip():
        raise TestbedError("container has no cpuset pinning; refusing to measure")
    cores: list[int] = []
    for part in spec.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            low, high = chunk.split("-", 1)
            cores.extend(range(int(low), int(high) + 1))
        else:
            cores.append(int(chunk))
    if not cores:
        raise TestbedError(f"cpuset {spec!r} expanded to nothing")
    return tuple(sorted(set(cores)))


@dataclass(frozen=True)
class Peer:
    """One Qdrant container participating in the cluster."""

    name: str
    container_id: str
    cores: tuple[int, ...]
    http_port: int

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.http_port}"

    @property
    def cpu_stat_path(self) -> Path:
        return CGROUP_SLICE / f"docker-{self.container_id}.scope" / "cpu.stat"

    def cpu_usage_usec(self) -> int:
        """Cumulative CPU time consumed by this container, in microseconds."""
        try:
            text = self.cpu_stat_path.read_text(encoding="utf-8")
        except OSError as error:
            raise TestbedError(f"cannot read cgroup stats for {self.name}") from error
        for line in text.splitlines():
            if line.startswith("usage_usec "):
                return int(line.split()[1])
        raise TestbedError(f"cgroup stats for {self.name} lack usage_usec")


def discover_peers(name_filter: str = DEFAULT_NAME_FILTER) -> tuple[Peer, ...]:
    """Enumerate running Qdrant peers, sorted with the controller first."""
    listing = subprocess.run(
        ["docker", "ps", "--filter", f"name={name_filter}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    names = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    if not names:
        raise TestbedError(f"no running containers match name={name_filter!r}")

    template = (
        "{{.Id}}|{{.HostConfig.CpusetCpus}}|"
        '{{(index .NetworkSettings.Ports "' + QDRANT_HTTP_PORT + '" 0).HostPort}}'
    )
    inspected = subprocess.run(
        ["docker", "inspect", "--format", template, *names],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [line.strip() for line in inspected.stdout.splitlines() if line.strip()]
    if len(rows) != len(names):
        raise TestbedError("docker inspect returned an unexpected number of rows")

    peers: list[Peer] = []
    for name, row in zip(names, rows):
        container_id, cpuset, port = row.split("|")
        peers.append(
            Peer(
                name=name,
                container_id=container_id,
                cores=parse_cpuset(cpuset),
                http_port=int(port),
            )
        )
    peers.sort(key=lambda peer: (0 if "controller" in peer.name else 1, peer.name))
    return tuple(peers)


def assert_disjoint(peers: Sequence[Peer], client_cores: Iterable[int]) -> None:
    """Reject a configuration where the load generator shares cores with a peer."""
    client = set(client_cores)
    for peer in peers:
        overlap = client.intersection(peer.cores)
        if overlap:
            raise TestbedError(
                f"client cores overlap {peer.name} on {sorted(overlap)}; "
                "the load generator must not share cores with a peer"
            )


def assert_symmetric(peers: Sequence[Peer]) -> None:
    """Reject asymmetric peer capacity, which confounds any shard comparison."""
    sizes = {peer.name: len(peer.cores) for peer in peers}
    if len(set(sizes.values())) > 1:
        raise TestbedError(f"peers have unequal core counts: {sizes}")


def host_busy_usec() -> int:
    """Cumulative non-idle CPU time across the whole host, in microseconds."""
    fields = PROC_STAT.read_text(encoding="utf-8").split("\n", 1)[0].split()
    if fields[0] != "cpu":
        raise TestbedError("unexpected /proc/stat layout")
    ticks = [int(value) for value in fields[1:]]
    idle = ticks[3] + ticks[4]
    total = sum(ticks[:8])
    hertz = 100  # USER_HZ is 100 on Linux x86_64
    return (total - idle) * 1_000_000 // hertz


def process_cpu_usec(pids: Iterable[int]) -> int:
    """Cumulative CPU time of the given processes, in microseconds."""
    total_ticks = 0
    for pid in pids:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        except OSError as error:
            raise TestbedError(f"client process {pid} vanished mid-run") from error
        total_ticks += int(fields[13]) + int(fields[14])
    return total_ticks * 10_000  # 1 tick = 10_000 us at USER_HZ 100


def describe(peers: Sequence[Peer]) -> str:
    return json.dumps(
        [
            {
                "name": peer.name,
                "cores": list(peer.cores),
                "core_count": len(peer.cores),
                "http_port": peer.http_port,
            }
            for peer in peers
        ],
        indent=2,
    )
