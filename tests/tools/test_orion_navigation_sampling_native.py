"""Real native sampling regression at fixed M, using small synthetic vectors."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

BINARY = os.environ.get("ORION_LAYOUT_BINARY")
OLD_BINARY = os.environ.get("ORION_OLD_LAYOUT_BINARY")


@unittest.skipUnless(BINARY, "set ORION_LAYOUT_BINARY to run native preparation")
class NavigationSamplingNativeTests(unittest.TestCase):
    def test_four_densities_keep_m32_and_nested_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            vectors = np.random.default_rng(193).normal(size=(16384, 8)).astype("<f4")
            raw = root / "input.f32le"
            raw.write_bytes(vectors.tobytes())
            previous = None
            for denominator in (32, 64, 128, 256):
                with self.subTest(denominator=denominator):
                    options = dict(generation=1, shard_count=8, distance="Euclid", vector_name="",
                                   sample_denominator=denominator, sample_seed=100, upper_m=32,
                                   upper_ef_construct=100, upper_seed=100, efs=10,
                                   upper_k=16, upper_ef_search=32, dynamic_ef_base=20,
                                   dynamic_ef_factor=4, kmeans_iterations=0, kmeans_seed=0,
                                   topology_iterations=0, bundle_prefix="import")
                    request = dict(input=dict(format_version=1, path=str(raw), rows=len(vectors),
                                              dimension=8, sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
                                   options=options)
                    req = root / f"request-{denominator}.json"
                    req.write_text(json.dumps(request))
                    output = root / f"d{denominator}"
                    completed = subprocess.run([BINARY, "prepare-capacity", "--request", str(req), "--output", str(output)],
                                               text=True, capture_output=True)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    prepared = json.loads((output / "capacity-input.json").read_text())
                    self.assertEqual(prepared["options"]["upper_m"], 32)
                    self.assertEqual(prepared["options"]["upper_ef_construct"], 100)
                    self.assertEqual(len(prepared["upper_indices"]), len(vectors) // denominator)
                    artifact = json.loads((output / prepared["source_artifact_file"]).read_text())
                    self.assertEqual(len(artifact["upper_graph"]["nodes"]), len(vectors) // denominator)
                    for node in artifact["upper_graph"]["nodes"]:
                        for level, links in enumerate(node["neighbors_by_level"]):
                            self.assertLessEqual(len(links), 64 if level == 0 else 32)
                    if previous is not None:
                        self.assertEqual(prepared["upper_indices"], previous[:len(prepared["upper_indices"])])
                    previous = prepared["upper_indices"]
                    if denominator == 32 and OLD_BINARY:
                        old = root / "old-d32"
                        completed = subprocess.run([OLD_BINARY, "prepare-capacity", "--request", str(req), "--output", str(old)],
                                                   text=True, capture_output=True)
                        self.assertEqual(completed.returncode, 0, completed.stderr)
                        old_prepared = json.loads((old / "capacity-input.json").read_text())
                        for field in ("upper_graph_sha256", "upper_identity_sha256", "attachments_sha256", "vectors_sha256"):
                            self.assertEqual(prepared[field], old_prepared[field], field)


if __name__ == "__main__":
    unittest.main()
