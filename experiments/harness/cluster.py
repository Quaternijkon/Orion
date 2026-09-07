"""Testbed discovery and CPU accounting for the containerized Qdrant cluster.

Every Qdrant peer runs as a Docker container pinned to a cpuset, so CPU
accounting is read from cgroup v2 rather than sampled with a profiler. All
sampling happens outside the measured request path.

Single-host and multi-host use the same code path. A "host" is either ``local``
(this machine, read directly) or an SSH target ``[user@]host`` (read via
``ssh``). Discovery and cgroup reads run per host, so a multi-machine cluster
keeps the same G1/G3 CPU fidelity as the single-host testbed -- provided each
server host is reachable by passwordless SSH and exposes docker to that user.
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
LOCAL_HOST = "local"


class TestbedError(RuntimeError):
    """Raised when the testbed cannot be described precisely enough to measure."""


def _is_local(host: str) -> bool:
    return host in (LOCAL_HOST, "", "127.0.0.1", "localhost")


def host_address(host: str) -> str:
    """The address a client uses to reach services on ``host``."""
    if _is_local(host):
        return "127.0.0.1"
    return host.split("@", 1)[-1]


def run_on_host(host: str, argv: Sequence[str]) -> subprocess.CompletedProcess:
    """Run a command locally or over SSH, returning the completed process."""
    if _is_local(host):
        command: list[str] = list(argv)
    else:
        command = ["ssh", "-o", "BatchMode=yes", host, *argv]
    return subprocess.run(command, capture_output=True, text=True, check=True)


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
    host: str = LOCAL_HOST

    @property
    def is_local(self) -> bool:
        return _is_local(self.host)

    @property
    def base_url(self) -> str:
        return f"http://{host_address(self.host)}:{self.http_port}"

    @property
    def cpu_stat_path(self) -> Path:
        return CGROUP_SLICE / f"docker-{self.container_id}.scope" / "cpu.stat"

    def cpu_usage_usec(self) -> int:
        """Cumulative CPU time consumed by this container, in microseconds.

        Read from cgroup v2 on the peer's own host (directly when local, over
        SSH otherwise), so multi-host runs keep the same CPU accounting as the
        single-host testbed.
        """
        path = str(self.cpu_stat_path)
        try:
            if self.is_local:
                text = self.cpu_stat_path.read_text(encoding="utf-8")
            else:
                text = run_on_host(self.host, ["cat", path]).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise TestbedError(
                f"cannot read cgroup stats for {self.name} on host {self.host!r}"
            ) from error
        for line in text.splitlines():
            if line.startswith("usage_usec "):
                return int(line.split()[1])
        raise TestbedError(f"cgroup stats for {self.name} lack usage_usec")


def _discover_host_peers(host: str, name_filter: str) -> list[Peer]:
    """Enumerate matching Qdrant containers on one host."""
    listing = run_on_host(
        host,
        ["docker", "ps", "--filter", f"name={name_filter}", "--format", "{{.Names}}"],
    )
    names = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    if not names:
        return []

    template = (
        "{{.Id}}|{{.HostConfig.CpusetCpus}}|"
        '{{(index .NetworkSettings.Ports "' + QDRANT_HTTP_PORT + '" 0).HostPort}}'
    )
    inspected = run_on_host(host, ["docker", "inspect", "--format", template, *names])
    rows = [line.strip() for line in inspected.stdout.splitlines() if line.strip()]
    if len(rows) != len(names):
        raise TestbedError(f"docker inspect on {host!r} returned unexpected rows")

    peers: list[Peer] = []
    for name, row in zip(names, rows):
        container_id, cpuset, port = row.split("|")
        peers.append(
            Peer(
                name=name if _is_local(host) else f"{host_address(host)}/{name}",
                container_id=container_id,
                cores=parse_cpuset(cpuset),
                http_port=int(port),
                host=host,
            )
        )
    return peers


def discover_peers(
    name_filter: str = DEFAULT_NAME_FILTER,
    hosts: Sequence[str] = (LOCAL_HOST,),
) -> tuple[Peer, ...]:
    """Enumerate running Qdrant peers across ``hosts``, controller first.

    ``hosts`` is one or more of ``local`` and SSH targets ``[user@]host``. The
    same docker inspection runs on each host, so a multi-machine cluster is just
    a longer host list. Peers keep their host so CPU is read where they run.
    """
    peers: list[Peer] = []
    for host in hosts:
        peers.extend(_discover_host_peers(host, name_filter))
    if not peers:
        raise TestbedError(
            f"no running containers match name={name_filter!r} on hosts {list(hosts)}"
        )
    peers.sort(key=lambda peer: (0 if "controller" in peer.name else 1, peer.name))
    return tuple(peers)


def assert_disjoint(peers: Sequence[Peer], client_cores: Iterable[int]) -> None:
    """Reject a configuration where the load generator shares cores with a peer.

    Only local peers can contend with the client for cores; peers on other hosts
    run on separate CPUs, so their cpusets are unconstrained here.
    """
    client = set(client_cores)
    for peer in peers:
        if not peer.is_local:
            continue
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
