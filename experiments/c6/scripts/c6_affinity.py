#!/usr/bin/env python3
"""Apply and record Qdrant CPU affinity for a live C6 native cluster."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from c6_native_cluster import load_topology, node_root, nodes, run_shell  # noqa: E402
from c6_protocol import utc_timestamp, write_json_atomic  # noqa: E402


CPUSET = re.compile(r"^[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--cluster-run-id", required=True)
    parser.add_argument("--runtime-root", default="/users/dry/orion-c6-runtime")
    parser.add_argument("--cpuset", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not CPUSET.fullmatch(args.cpuset):
        raise ValueError("cpuset must be a comma-separated CPU/range list")
    topology = load_topology(args.topology)
    applied = {}
    for node in nodes(topology):
        root = node_root(args.runtime_root, args.cluster_run_id, node)
        command = (
            f"pid=$(cat {shlex.quote(str(root / 'qdrant.pid'))}); "
            f"taskset -apc {shlex.quote(args.cpuset)} \"$pid\" >/dev/null; "
            "taskset -pc \"$pid\""
        )
        line = run_shell(node, command).stdout.strip().splitlines()[-1]
        actual = line.rsplit(":", 1)[-1].strip()
        if actual != args.cpuset:
            raise RuntimeError(
                f"affinity mismatch for {node['role']}: expected {args.cpuset}, got {actual}"
            )
        applied[str(node["role"])] = actual
    record = {
        "record_type": "c6_cluster_affinity",
        "timestamp": utc_timestamp(),
        "cluster_run_id": args.cluster_run_id,
        "requested_cpuset": args.cpuset,
        "applied": applied,
    }
    write_json_atomic(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
