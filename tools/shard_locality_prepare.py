#!/usr/bin/env python3
"""Start the matched Orion layout preparation under eight-host experiment leases.

Run from a frozen experiment directory containing source/, config/, bin/.
This phase never searches evaluation queries, calibrates recall, or measures QPS.
It produces the required M32/ef_construct200 layout for subsequent lower indexes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run(root):
    sys.path.insert(0, str(root / "source"))
    import admission
    import orion_build_resources as resources
    child = None

    def save(stage, **fields):
        value = dict(stage=stage, time=time.time(), pid=os.getpid(),
                     qps_measured=False, target_recall_calibration=False,
                     formal_measurement_started=False, **fields)
        tmp = root / "status.tmp.json"
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(root / "status.json")
        print(json.dumps(value), flush=True)

    def stop(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        proof = json.loads((root / "evidence/frozen-inputs.json").read_text())
        for name, expected in proof.items():
            with (root / name).open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                    raise ValueError(f"frozen input changed: {name}")
        request = json.loads((root / "config/build.json").read_text())
        options = request["options"]
        if (options["shard_count"], options["upper_m"], options["upper_ef_construct"]) != (8, 32, 200):
            raise ValueError("only the authorized 8-shard M32/ef_construct200 build is allowed")
        while True:
            if (root / "STOP").exists():
                raise KeyboardInterrupt()
            snapshot = admission.probe_all()
            (root / "evidence/admission.json").write_text(json.dumps(snapshot, indent=2))
            if admission.available(snapshot):
                break
            save("waiting_for_eight_hosts", occupancy=snapshot)
            time.sleep(30)
        with admission.Lease() as lease:
            resources.save_plan(root, snapshot, 'prepare', request['input']['dimension'],
                                request['input']['rows'], admission.HOSTS[0])
            command = [sys.executable, str(root / "source/orion_capacity_partition.py"),
                       "--request", str(root / "config/build.json"),
                       "--whitebox-config", str(root / "config/whitebox.json"),
                       "--rust-layout-binary", str(root / "bin/orion_build_layout"),
                       "--output", str(root / "layout")]
            reuse = root / "prepared-parallel"
            if (reuse / "validated.json").is_file():
                command += ["--reuse-prepared", str(reuse)]
            elif (root / "fast-build-launch.json").is_file():
                # Bootstrap only: the mandatory watcher migrates the bulk pass
                # after enough original records exist for the parity sample.
                command += ["--allow-serial-attachments"]
            (root / "evidence/layout-command.json").write_text(json.dumps(command, indent=2))
            environment = {k: v for k, v in os.environ.items()
                           if not k.startswith(("ORION_", "QDRANT_"))}
            environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            log_name = "layout-resumed.log" if (reuse / "validated.json").is_file() else "layout.log"
            with (root / "logs" / log_name).open("xb") as log:
                child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    lease.check()
                    if (root / "STOP").exists():
                        raise KeyboardInterrupt()
                    save("building_orion_layout", child_pid=child.pid, m=32,
                         ef_construct=200, shards=8, hosts=list(admission.HOSTS))
                    time.sleep(15)
                if child.returncode:
                    raise RuntimeError(f"layout builder exited {child.returncode}; see logs/layout.log")
            build = json.loads((root / "layout/build-manifest.json").read_text())
            if (build["options"]["upper_m"], build["options"]["upper_ef_construct"]) != (32, 200):
                raise ValueError("completed graph build metadata mismatch")
            save("layout_ready_lower_indexes_pending", physical_counts=build["shard_counts"],
                 next_stage="Build eight lower HNSW indexes with M32/ef_construct200, then measure Oracle/Real on eight hosts")
    except KeyboardInterrupt:
        save("interrupted")
    except Exception as error:
        save("failed", error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve())
