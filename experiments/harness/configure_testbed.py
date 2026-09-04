"""Give every Qdrant peer an identical cpuset, reversibly.

Unequal peer capacity confounds any comparison across shard counts: a peer with
more cores cannot reach the same per-core utilization as its neighbors, so the
server-saturation gate can never pass. ``docker update`` changes the cpuset in
place without restarting the container, and the original assignment is saved so
it can be restored.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cluster import discover_peers

BACKUP_PATH = Path(__file__).resolve().parent / "cpuset_backup.json"


def format_cores(cores: list[int]) -> str:
    return ",".join(str(core) for core in cores)


def apply(name: str, cpuset: str) -> None:
    subprocess.run(
        ["docker", "update", "--cpuset-cpus", cpuset, name],
        capture_output=True,
        text=True,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores-per-peer", type=int, default=8)
    parser.add_argument("--first-core", type=int, default=0)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    peers = discover_peers()

    if args.restore:
        if not BACKUP_PATH.is_file():
            raise SystemExit(f"no backup at {BACKUP_PATH}")
        backup = json.loads(BACKUP_PATH.read_text(encoding="utf-8"))
        for name, cpuset in backup.items():
            apply(name, cpuset)
            print(f"restored {name} -> {cpuset}")
        return 0

    if not BACKUP_PATH.is_file():
        BACKUP_PATH.write_text(
            json.dumps(
                {peer.name: format_cores(list(peer.cores)) for peer in peers}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"saved original cpusets to {BACKUP_PATH}")

    for index, peer in enumerate(peers):
        start = args.first_core + index * args.cores_per_peer
        cpuset = f"{start}-{start + args.cores_per_peer - 1}"
        apply(peer.name, cpuset)
        print(f"{peer.name} -> {cpuset}")

    print("\nverifying:")
    for peer in discover_peers():
        print(f"  {peer.name}: {len(peer.cores)} cores {peer.cores[0]}-{peer.cores[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
