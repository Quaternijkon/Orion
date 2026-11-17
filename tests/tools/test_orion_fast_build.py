"""Fast-launch policy tests; no full dataset scans, SSH, or real child jobs."""
import argparse
import contextlib
import copy
import io
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import orion_fast_build as fast
from tools import orion_capacity_partition as capacity
from tools import shard_locality_archives as archive
from tools import orion_build_admission as admission


class FastBuildTests(unittest.TestCase):
    def profile(self):
        return (fast.read(fast.REPO/'config/shard_locality_sift100m_build.json'),
                fast.read(fast.REPO/'config/shard_locality_sift100m.json'))

    def test_defaults_are_parallel_and_reject_incompatible_profiles(self):
        build, experiment = self.profile()
        fast.validate_profile(build, experiment)
        for change in ('encoding', 'rows', 'efs', 'hosts', 'profiles', 'runtime'):
            b, e = copy.deepcopy(build), copy.deepcopy(experiment)
            if change == 'encoding': b['input']['encoding'] = 'u8'
            if change == 'rows': b['input']['rows'] = 1000
            if change == 'efs': b['options']['efs'] = 11
            if change == 'hosts': e['execution']['hosts'][1] = e['execution']['hosts'][0]
            if change == 'profiles': e['orion_profiles'].append(e['orion_profiles'][0])
            if change == 'runtime': e['orion_profiles'][0]['upper_k'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                fast.validate_profile(b, e)

    def test_admission_checks_real_host_count_cores_and_existing_locks(self):
        local = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        state = {h: {'boot_id': local if i == 0 else f'other-{i}', 'processes': [],
                     'lock_busy': False, 'resources': {'worker_count': 96}}
                 for i,h in enumerate(admission.HOSTS)}
        self.assertTrue(admission.available(state))
        state[admission.HOSTS[1]]['lock_busy'] = True
        self.assertFalse(admission.available(state))
        state[admission.HOSTS[1]]['lock_busy'] = False
        state[admission.HOSTS[1]]['boot_id'] = local
        with self.assertRaisesRegex(ValueError, 'distinct machines'): admission.available(state)

    def test_large_legacy_build_is_blocked_before_native_work_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); request = root/'request.json'
            request.write_text(json.dumps({'input': {'rows': 100000000}}))
            with self.assertRaisesRegex(ValueError, 'orion_fast_build'), patch.object(capacity, 'plan_prepared') as planner:
                capacity.build_capacity(request, root/'output', lambda *_: self.fail('native work started'))
            planner.assert_not_called()
            self.assertFalse((root/'output').exists())

    def test_byte_parity_failure_stops_owned_jobs_without_serial_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fast.save(root/'fast-build-launch.json', {'root': str(root), 'format_version': 1})
            fast.save(root/'status.json', {'stage': 'building_orion_layout'})
            with patch.object(fast, 'find_attachment_stage', return_value=root/'stage'), \
                 patch.object(fast, 'make_parity', side_effect=ValueError('different bytes')), \
                 patch.object(fast, 'stop_owned') as stop:
                with self.assertRaisesRegex(ValueError, 'different bytes'): fast.watch(root)
            stop.assert_called_once()
            result = fast.read(root/'auto-attachments.json')
            self.assertEqual(result['stage'], 'failed')
            self.assertFalse(result['serial_fallback'])

    def test_stop_never_signals_a_reused_or_unrelated_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fast.save(root/'controller.json', {'pid': os.getpid(), 'start_ticks': -1})
            with patch.object(fast.os, 'kill') as kill: fast.stop_owned(root, 'test stop')
            kill.assert_not_called()
            self.assertTrue((root/'STOP').exists())

    def test_plan_only_does_not_create_workspace_or_launch_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); build, experiment = self.profile()
            data = base/'vectors.f32le'
            # Sparse metadata-only fixture; never read/hash 51 GB in a unit test.
            with data.open('wb') as f: f.truncate(100000000 * 128 * 4)
            query = base/'queries.npz'; query.write_bytes(b'preflight-fixture')
            build['input']['path'] = str(data); experiment['queries_npz'] = str(query)
            fast.save(base/'build.json', build); fast.save(base/'experiment.json', experiment)
            binary = base/'binary'; binary.write_text('fixture'); binary.chmod(0o700)
            args = argparse.Namespace(build_request=base/'build.json', experiment_config=base/'experiment.json',
                root=base/'new-run', layout_binary=binary, attachment_binary=binary, search_binary=binary,
                kmeans_archives=base/'archives', python_deps=None, controller_ip=None,
                reuse_prepared=None, execute=False)
            config = {'hnsw_config': {'m': 32, 'ef_construct': 200, 'full_scan_threshold': 10}}
            artifact = {'logical_point_count': 100000000, 'vector_schema': {'dimension': 128, 'distance': 'Euclid'}}
            with patch.object(archive, 'inspect_archives', return_value=([], config, artifact)), \
                 patch.object(fast.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,
                     stdout=json.dumps({'schema_version':1,'unrestricted_positive_indexing_threads':True}))) as checks, \
                 patch.object(fast.subprocess, 'Popen') as spawn, \
                 contextlib.redirect_stdout(io.StringIO()):
                plan = fast.launch(args)
            self.assertEqual(plan['attachment_threads_per_host'], 'auto')
            self.assertEqual(plan['lower_threads_per_host'], 'auto')
            self.assertEqual(plan['split_processes'], 'auto_memory_bounded')
            self.assertFalse(plan['search_resources_changed'])
            self.assertTrue(plan['automatic_parallel_migration'])
            self.assertFalse(plan['serial_fallback'])
            self.assertFalse(args.root.exists())
            spawn.assert_not_called()
            self.assertEqual(checks.call_count, 2)  # native capability and dependency import only

    def test_stale_native_binary_is_rejected_before_expensive_work(self):
        with patch.object(fast.subprocess, 'run', side_effect=subprocess.CalledProcessError(1,['old-worker'])):
            with self.assertRaisesRegex(ValueError,'rebuild'):
                fast.validate_build_binary(Path('/old-worker'),{})
        with patch.object(fast.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,stdout='{}')):
            with self.assertRaisesRegex(ValueError,'rebuild'):
                fast.validate_build_binary(Path('/old-worker'),{})

    def test_parity_checks_the_resolved_thread_count_and_affinity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); stage=root/'stage'; stage.mkdir()
            (root/'evidence').mkdir(); (root/'config').mkdir()
            with (stage/'import.f32le').open('wb') as stream:
                stream.truncate(100000000 * 128 * 4)
            expected = b'\0' * (fast.PARITY_ROWS * 84)
            (stage/'upper-attachments.counted.bin').write_bytes(expected)
            fast.save(root/'config/experiment.json', {'execution': {'hosts': ['controller']}})
            fast.save(root/'evidence/build-resources-prepare.json', {
                'hosts': {'controller': {'selected_cpus': list(range(96))}},
                'attachment_threads_per_host': {'controller': 96}})
            requests = []
            def worker(command, **kwargs):
                request = fast.read(command[-1]); requests.append((command,request))
                output = Path(request['output']); output.mkdir()
                (output/'hits.bin').write_bytes(expected)
            with patch.object(fast.subprocess, 'run', side_effect=worker):
                fast.make_parity(root, stage)
            self.assertEqual([request['threads'] for _,request in requests], [1,96])
            self.assertEqual(requests[-1][0][:3], ['taskset','-c',','.join(map(str,range(96)))])
            self.assertEqual(fast.read(root/'parallel-attachments/parity/verified.json')['thread_counts'], [1,96])


if __name__ == '__main__': unittest.main()
