"""Small real native build; opt in by supplying ORION_LAYOUT_BINARY."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.orion_capacity_partition import build_capacity

BINARY = os.environ.get('ORION_LAYOUT_BINARY')


@unittest.skipUnless(BINARY, 'set ORION_LAYOUT_BINARY to exercise real Rust builds')
class NativeCapacityTests(unittest.TestCase):
    def test_direct_build_early_acceptance_full_replay_and_original_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(340)
            centers = np.eye(4, 8, dtype=np.float32) * 20
            vectors = np.vstack([c + rng.normal(0, .05, (512, 8)) for c in centers]).astype('<f4')
            raw = root / 'input.f32le'
            raw.write_bytes(vectors.tobytes())
            options = dict(generation=1, shard_count=4, distance='Euclid', vector_name='',
                           sample_denominator=8, sample_seed=100, upper_m=8,
                           upper_ef_construct=64, upper_seed=100, efs=10,
                           upper_k=8, upper_ef_search=8, dynamic_ef_base=1,
                           dynamic_ef_factor=4, kmeans_iterations=0, kmeans_seed=1,
                           topology_iterations=0, bundle_prefix='import')
            request = dict(input=dict(format_version=1, path=str(raw), rows=len(vectors),
                                     dimension=8, sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
                           options=options)
            req = root / 'request.json'
            req.write_text(json.dumps(request))
            commands = []
            def native(*args):
                commands.append(args)
                p = subprocess.run([BINARY, *args], capture_output=True, text=True)
                if p.returncode:
                    self.fail(p.stderr)
                return json.loads(p.stdout)
            final = root / 'capacity'
            built = build_capacity(req, final, native, max_feedback_rounds=0,
                                   max_replication_factor=1.1)
            self.assertEqual([c[0] for c in commands], ['prepare-capacity', 'capacity-partition'])
            plan = json.loads((final / 'capacity-plan.json').read_text())
            self.assertEqual(plan['feedback_rounds_used'], 0)
            self.assertEqual(len(plan['history']), 1)
            self.assertFalse(plan['source_layout_required'])
            self.assertFalse(plan['replica_budget_owner_repair'])
            self.assertEqual(built['topology_iterations'], 0)
            self.assertEqual(built['capacity_max_replication_factor'], 1.1)
            verified = native('verify', '--bundle', str(final), '--replay')
            self.assertEqual(verified['point_count'], 2048)
            self.assertEqual(verified['total_point_copies'], sum(built['shard_counts']))
            # Build the original path afterwards only for parity checking. It was
            # not a prerequisite of capacity construction.
            options.update(kmeans_iterations=10, topology_iterations=50)
            req.write_text(json.dumps(request))
            old = native('build', '--request', str(req), '--output', str(root / 'original'))
            self.assertEqual(old['upper_graph_sha256'], built['upper_graph_sha256'])
            self.assertEqual(old['attachments_sha256'], built['attachments_sha256'])
            with self.assertRaises(FileExistsError):
                build_capacity(req, final, native)


if __name__ == '__main__':
    unittest.main()
