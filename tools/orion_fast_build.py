#!/usr/bin/env python3
"""Reproducible fast SIFT100M build/search launcher; default is plan-only.

The short serial attachment prefix is only a parity reference. A watcher
automatically migrates the remaining rows to eight hosts. Failure stops owned
jobs instead of silently completing a large serial pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import re

if __package__:
    from . import orion_build_resources as resources
else:
    import orion_build_resources as resources

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
PARITY_ROWS = 8192


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def owned(record, root):
    try:
        proc = Path('/proc') / str(record['pid'])
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        return (fields[0] != 'Z' and int(fields[19]) == record['start_ticks']
                and str(root).encode() in (proc/'cmdline').read_bytes())
    except FileNotFoundError:
        return False


def stop_owned(root, reason):
    (root/'STOP').write_text(reason + '\n')
    for name in ['controller.json', 'continuation-process.json']:
        path = root/name
        if path.exists():
            record = read(path)
            if owned(record, root):
                try: os.kill(record['pid'], signal.SIGTERM)
                except ProcessLookupError: pass


def find_attachment_stage(root, state):
    child = state.get('child_pid')
    if child is None: return None
    try:
        children = Path(f'/proc/{child}/task/{child}/children').read_text().split()
        for pid in children:
            proc = Path('/proc')/pid
            if b'prepare-capacity' not in (proc/'cmdline').read_bytes(): continue
            for fd in (proc/'fd').iterdir():
                try:
                    path = Path(os.readlink(fd))
                    if (path.name == 'upper-attachments.counted.bin' and path.is_relative_to(root)
                            and fd.stat().st_size >= PARITY_ROWS * 84):
                        return path.parent
                except OSError: pass
    except FileNotFoundError: pass
    return None


def make_parity(root, stage):
    import numpy as np
    work = root/'parallel-attachments'
    work.mkdir()
    parity = work/'parity'
    parity.mkdir()
    complete_rows = (stage/'upper-attachments.counted.bin').stat().st_size // 84
    selected = np.sort(np.random.default_rng(2718).choice(complete_rows, PARITY_ROWS, replace=False))
    vectors = np.memmap(stage/'import.f32le', dtype='<f4', mode='r', shape=(100000000, 128))
    hits = np.memmap(stage/'upper-attachments.counted.bin', dtype='u1', mode='r', shape=(complete_rows, 84))
    vectors[selected].tofile(parity/'vectors.f32le')
    hits[selected].tofile(parity/'expected.bin')
    np.save(parity/'rows.npy', selected, allow_pickle=False)
    expected = (parity/'expected.bin').read_bytes()
    plan = resources.read_plan(root, 'prepare')
    controller = read(root/'config/experiment.json')['execution']['hosts'][0]
    cpus = plan['hosts'][controller]['selected_cpus']
    thread_counts = sorted({1, plan['attachment_threads_per_host'][controller]})
    for threads in thread_counts:
        request = dict(artifact=str(stage/'upper-source-generation-1.json'), vectors=str(parity/'vectors.f32le'),
                       rows=PARITY_ROWS, dimension=128, efs=10, threads=threads,
                       output=str(parity/f'out-{threads}'))
        path = parity/f'request-{threads}.json'
        save(path, request)
        with (parity/f'worker-{threads}.log').open('x') as log:
            subprocess.run(['taskset', '-c', resources.cpu_list(cpus[:threads]),
                            str(root/'bin/orion_attachment_parallel'), 'export', str(path)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        if (parity/f'out-{threads}/hits.bin').read_bytes() != expected:
            raise ValueError(f'{threads}-thread output differs from the original native attachment prefix')
    save(parity/'verified.json', {'rows': PARITY_ROWS, 'sampled_from_completed_rows': complete_rows,
                                 'thread_counts': thread_counts, 'cpu_affinity': cpus,
                                 'byte_identical_to_original_native_prepare_capacity': True})


def watch(root):
    marker = read(root/'fast-build-launch.json')
    if marker['root'] != str(root) or marker['format_version'] != 1:
        raise ValueError('not an owned fast-build workspace')
    status_path = root/'auto-attachments.json'
    process = None
    try:
        while True:
            if (root/'STOP').exists(): return
            if not (root/'status.json').exists():
                time.sleep(2); continue
            state = read(root/'status.json')
            if state['stage'] in ('failed', 'interrupted'):
                raise RuntimeError('upstream preparation stopped before parallel migration')
            stage = find_attachment_stage(root, state)
            if stage is not None: break
            if state['stage'] == 'layout_ready_lower_indexes_pending':
                raise RuntimeError('serial preparation completed without the required parallel stage')
            save(status_path, {'stage': 'waiting_for_frozen_upper_graph', 'time': time.time()})
            time.sleep(2)
        save(status_path, {'stage': 'checking_native_byte_parity', 'time': time.time()})
        make_parity(root, stage)
        save(status_path, {'stage': 'running_eight_host_attachment_stage', 'time': time.time()})
        with (root/'parallel-attachments/controller.log').open('x') as log:
            process = subprocess.Popen([sys.executable, str(root/'source/shard_locality_parallel.py'), '--root', str(root)],
                                       stdout=log, stderr=subprocess.STDOUT)
            fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
            save(root/'parallel-attachments/controller.json', {'pid': process.pid, 'start_ticks': int(fields[19]), 'time': time.time()})
            if process.wait(): raise RuntimeError('parallel attachment migration failed')
        if not read(root/'prepared-parallel/validated.json')['validated']:
            raise RuntimeError('parallel preparation lacks native validation')
        save(status_path, {'stage': 'parallel_preparation_complete', 'time': time.time()})
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
        save(status_path, {'stage': 'failed', 'time': time.time(), 'error': str(error), 'serial_fallback': False})
        stop_owned(root, f'Automatic parallel attachment stage failed: {error}')
        raise


def validate_profile(build, experiment):
    options, data = build['options'], build['input']
    execution = experiment['execution']
    if (data['rows'], data['dimension'], data.get('encoding', 'f32le')) != (100000000, 128, 'f32le'):
        raise ValueError('this tested preset requires SIFT100M / 128-dimensional f32le input')
    expected = dict(generation=1, shard_count=8, upper_m=32, upper_ef_construct=200,
                    efs=10, bundle_prefix='import', distance='Euclid')
    if any(options[k] != value for k, value in expected.items()):
        raise ValueError('build request differs from the tested M32/ef_construct200 eight-shard preset')
    hosts = execution['hosts']
    if (execution['mode'] != 'eight_hosts' or len(hosts) != 8 or len(set(hosts)) != 8 or
            any(not isinstance(h,str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9.-]*',h) for h in hosts)):
        raise ValueError('eight distinct hosts are required')
    if len(experiment['orion_profiles']) != 1 or experiment['kmeans'] != {'reuse_existing_index': True}:
        raise ValueError('one fixed Orion profile and reused KMeans indexes are required')
    if (experiment['lower']['m'], experiment['lower']['ef_construct']) != (32, 200):
        raise ValueError('lower construction must match 32/200')
    profile = experiment['orion_profiles'][0]
    for key in ('upper_k', 'upper_ef_search', 'dynamic_ef_base', 'dynamic_ef_factor'):
        if options[key] != profile[key]: raise ValueError(f'build/search profile mismatch: {key}')


def validate_build_binary(binary, env):
    try:
        result = subprocess.run([str(binary), '--build-capabilities'], env=env, check=True,
                                capture_output=True, text=True, timeout=30)
        capabilities = json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError) as error:
        raise ValueError('rebuild the shard_locality native worker: automatic build threads require a current binary') from error
    if (not isinstance(capabilities, dict) or capabilities.get('schema_version') != 1 or
            capabilities.get('unrestricted_positive_indexing_threads') is not True):
        raise ValueError('native worker does not support automatic build threads; rebuild shard_locality')


def launch(args):
    build, experiment = read(args.build_request), read(args.experiment_config)
    validate_profile(build, experiment)
    reuse = args.reuse_prepared.resolve() if args.reuse_prepared else None
    if reuse:
        prepared = read(reuse/'capacity-input.json')
        if (not read(reuse/'validated.json').get('validated') or prepared['options'] != build['options'] or
                prepared['raw_vectors_sha256'] != build['input']['sha256']):
            raise ValueError('reused preparation does not match the complete request and source identity')
    for path in [Path(build['input']['path']), Path(experiment['queries_npz'])]:
        if not path.is_file(): raise FileNotFoundError(path)
    if Path(build['input']['path']).stat().st_size != 100000000 * 128 * 4:
        raise ValueError('SIFT100M input file shape mismatch')
    binaries = {'orion_build_layout': args.layout_binary, 'orion_attachment_parallel': args.attachment_binary,
                'shard_locality_worker': args.search_binary}
    for path in binaries.values():
        if not path.is_file() or not os.access(path, os.X_OK): raise ValueError(f'executable missing: {path}')
    if __package__:
        from . import shard_locality_archives as archive
    else:
        import shard_locality_archives as archive
    records, collection, artifact = archive.inspect_archives(args.kmeans_archives, 'kmeans')
    archive.require_matched_indexes(experiment, [('kmeans', records, collection)])
    if (artifact['logical_point_count'] != 100000000 or artifact['vector_schema']['dimension'] != 128 or
            artifact['vector_schema']['distance'] != 'Euclid'):
        raise ValueError('KMeans control is not a matching SIFT100M-sized Euclid collection')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('ORION_', 'QDRANT_'))}
    if args.python_deps:
        env['PYTHONPATH'] = str(args.python_deps.resolve()) + os.pathsep + env.get('PYTHONPATH', '')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               ORION_BUILD_HOSTS=json.dumps(experiment['execution']['hosts']),
               ORION_BUILD_CONTROLLER_IP=args.controller_ip or experiment['execution']['hosts'][0])
    validate_build_binary(args.search_binary, env)
    subprocess.run([sys.executable, '-c', 'import numpy,pymetis'], env=env, check=True, capture_output=True)
    plan = dict(mode='execute' if args.execute else 'plan_only', root=str(args.root.resolve()),
                hosts=experiment['execution']['hosts'], attachment_threads_per_host='auto',
                split_processes='auto_memory_bounded', lower_threads_per_host='auto',
                lower_cpu_affinity='all_available_physical_cores',
                search_resources_changed=False,
                resource_resolution='per host after admission; evidence/build-resources-*.json',
                automatic_parallel_migration=reuse is None, reuse_prepared=str(reuse) if reuse else None,
                serial_fallback=False, qps=False)
    print(json.dumps(plan, indent=2))
    if not args.execute: return plan
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'config', 'bin', 'evidence', 'logs']: (root/name).mkdir()
    modules = ['orion_fast_build.py', 'orion_build_resources.py', 'orion_capacity_partition.py', 'orion_access_balance.py',
               'shard_locality.py', 'shard_locality_archives.py', 'shard_locality_split.py',
               'shard_locality_prepare.py', 'shard_locality_continue.py', 'shard_locality_parallel.py']
    for name in modules: shutil.copy2(TOOLS/name, root/'source'/name)
    shutil.copy2(TOOLS/'orion_build_admission.py', root/'source/admission.py')
    for name, path in binaries.items(): shutil.copy2(path, root/'bin'/name)
    save(root/'config/build.json', build)
    save(root/'config/experiment.json', experiment)
    save(root/'config/whitebox.json', dict(experiment['orion_profiles'][0]['whitebox'], distance=experiment['distance']))
    save(root/'runtime.json', {'kmeans_archives': str(args.kmeans_archives.resolve())})
    if reuse:
        # Inputs remain immutable; the canonical builder rechecks their hashes.
        (root/'prepared-parallel').symlink_to(reuse, target_is_directory=True)
        save(root/'evidence/reused-preparation.json', {'source': str(reuse),
             'manifest_sha256': sha(reuse/'capacity-input.json')})
    save(root/'evidence/frozen-inputs.json', {str(p.relative_to(root)): sha(p)
          for folder in ['source', 'config', 'bin'] for p in (root/folder).iterdir()})
    save(root/'fast-build-launch.json', {'format_version': 1, **plan, 'time': time.time()})
    jobs = [(['--root', str(root)], 'shard_locality_prepare.py', 'controller.json'),
            (['--root', str(root)], 'shard_locality_continue.py', 'continuation-process.json')]
    if reuse is None:
        jobs.append((['--watch-root', str(root)], 'orion_fast_build.py', 'auto-attachments-process.json'))
    try:
        for arguments, script, record_name in jobs:
            with (root/'logs'/f'{script}.log').open('x') as log:
                process = subprocess.Popen([sys.executable, str(root/'source'/script), *arguments], env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
            save(root/record_name, {'pid': process.pid, 'start_ticks': int(fields[19]), 'time': time.time()})
    except BaseException:
        stop_owned(root, 'launcher failed'); raise
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--build-request', type=Path, default=REPO/'config/shard_locality_sift100m_build.json')
    parser.add_argument('--experiment-config', type=Path, default=REPO/'config/shard_locality_sift100m.json')
    parser.add_argument('--layout-binary', type=Path)
    parser.add_argument('--attachment-binary', type=Path)
    parser.add_argument('--search-binary', type=Path)
    parser.add_argument('--kmeans-archives', type=Path)
    parser.add_argument('--python-deps', type=Path)
    parser.add_argument('--controller-ip')
    parser.add_argument('--reuse-prepared', type=Path,
                        help='Reuse a validated accelerated preparation with identical request/input identities')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.watch_root:
        def interrupt(*_): raise KeyboardInterrupt()
        signal.signal(signal.SIGINT, interrupt); signal.signal(signal.SIGTERM, interrupt)
        watch(args.watch_root.resolve()); return
    if any(getattr(args, name) is None for name in ['root', 'layout_binary', 'attachment_binary', 'search_binary', 'kmeans_archives']):
        parser.error('--root, all three binaries, and --kmeans-archives are required')
    launch(args)


if __name__ == '__main__':
    main()
