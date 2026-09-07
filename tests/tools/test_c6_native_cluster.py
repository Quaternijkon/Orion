from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "experiments/c6/scripts/c6_native_cluster.py"


def load_module():
    spec = importlib.util.spec_from_file_location("c6_native_cluster", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def topology_payload():
    return {
        "controller": {
            "role": "controller",
            "ssh_host": "localhost",
            "private_ip": "10.0.0.1",
            "cpuset": "0-3",
            "max_search_threads": 4,
            "optimizer_cpu_budget": 1,
        },
        "workers": [
            {
                "role": f"worker_{index}",
                "ssh_host": f"10.0.0.{index + 2}",
                "private_ip": f"10.0.0.{index + 2}",
                "cpuset": "0-3",
                "max_search_threads": 4,
                "optimizer_cpu_budget": 1,
            }
            for index in range(3)
        ],
        "ports": {"http": 6333, "grpc": 6334, "p2p": 6335},
    }


def test_topology_and_run_id_validation(tmp_path):
    module = load_module()
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(topology_payload()))
    topology = module.load_topology(path)
    assert len(module.nodes(topology)) == 4
    assert module.validate_run_id("c6-sift-m4") == "c6-sift-m4"


def test_start_command_binds_run_scoped_storage_and_bootstrap():
    module = load_module()
    topology = topology_payload()
    controller = topology["controller"]
    worker = topology["workers"][0]
    root = Path("/users/dry/orion-c6-runtime/run/worker_0")
    command = module.start_command(
        topology,
        worker,
        root,
        root / "bin/qdrant",
        root / "config/config.yaml",
    )
    assert "QDRANT__SERVICE__HARDWARE_REPORTING=true" in command
    assert f"QDRANT__STORAGE__STORAGE_PATH={root}/storage" in command
    assert "--bootstrap http://10.0.0.1:6335" in command
    assert "taskset -c 0-3" in command

    controller_command = module.start_command(
        topology,
        controller,
        Path("/tmp/controller"),
        Path("/tmp/controller/qdrant"),
        Path("/tmp/controller/config.yaml"),
    )
    assert "--bootstrap" not in controller_command
    assert "--uri http://10.0.0.1:6335" in controller_command
