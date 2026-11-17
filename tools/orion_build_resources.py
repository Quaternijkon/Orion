"""Resource discovery for experiment preparation only; never configures searches.

Use one allowed logical CPU per physical core, across all NUMA nodes. Respect
affinity and cgroup quotas rather than treating SMT threads as physical cores.
This file is also executed on build hosts, so it has no third-party imports.
"""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile


def cgroup_directories(proc=Path('/proc'), root=Path('/sys/fs/cgroup')):
    """Find visible cgroup ancestors, including namespace-rooted mounts (v1/v2)."""
    memberships = {}
    for line in (proc/'self/cgroup').read_text().splitlines():
        _, controllers, path = line.split(':', 2)
        for controller in controllers.split(','):
            memberships[controller] = path
    found = set()
    for line in (proc/'self/mountinfo').read_text().splitlines():
        before, after = line.split(' - ', 1)
        fields, filesystem = before.split(), after.split()
        if filesystem[0] not in ('cgroup', 'cgroup2'):
            continue
        mount = Path(fields[4].replace('\\040', ' '))
        # The root argument permits isolated synthetic tests.
        if not mount.is_relative_to('/sys/fs/cgroup'):
            continue
        mount = root / mount.relative_to('/sys/fs/cgroup')
        controllers = [''] if filesystem[0] == 'cgroup2' else filesystem[2].split(',')
        for controller in controllers:
            path = memberships.get(controller)
            if path is None:
                continue
            mounted_root = Path(fields[3])
            path = Path(path)
            if '..' in path.parts:
                # Some container runtimes expose a host-relative membership
                # outside this namespace. Only the mounted subtree is visible.
                directory = mount
            elif path.is_relative_to(mounted_root):
                directory = mount / path.relative_to(mounted_root)
            else:
                directory = mount / str(path).lstrip('/')
            while directory.is_relative_to(mount):
                found.add(directory)
                if directory == mount:
                    break
                directory = directory.parent
    return sorted(found)


def cgroup_limits(directories):
    cpu_limits, memory_limits = [], []
    for directory in directories:
        cpu = directory/'cpu.max'
        if cpu.is_file():
            quota, period = cpu.read_text().split()
            if quota != 'max':
                cpu_limits.append(int(quota) / int(period))
        elif (directory/'cpu.cfs_quota_us').is_file():
            quota = int((directory/'cpu.cfs_quota_us').read_text())
            if quota > 0:
                cpu_limits.append(quota / int((directory/'cpu.cfs_period_us').read_text()))
        for maximum, current in [('memory.max', 'memory.current'),
                                 ('memory.limit_in_bytes', 'memory.usage_in_bytes')]:
            if (directory/maximum).is_file() and (directory/current).is_file():
                limit = (directory/maximum).read_text().strip()
                if limit != 'max':
                    memory_limits.append(max(0, int(limit) - int((directory/current).read_text())))
    return (min(cpu_limits) if cpu_limits else None,
            min(memory_limits) if memory_limits else None)


def detect(proc=Path('/proc'), sys_cpu=Path('/sys/devices/system/cpu'),
           cgroup_root=Path('/sys/fs/cgroup'), allowed=None):
    allowed = sorted(os.sched_getaffinity(0) if allowed is None else allowed)
    cores = {}
    numa = {}
    for cpu in allowed:
        directory = sys_cpu/f'cpu{cpu}'
        topology = directory/'topology'
        key = tuple((topology/name).read_text().strip()
                    for name in ('physical_package_id', 'core_id'))
        if any(int(value) < 0 for value in key):
            raise ValueError(f'physical CPU topology unavailable for CPU {cpu}')
        cores.setdefault(key, cpu)
        numa[cpu] = next((node.name for node in directory.glob('node[0-9]*')), 'unknown')
    physical = sorted(cores.values())
    if not physical:
        raise ValueError('no allowed physical CPU cores')
    quota, memory_limit = cgroup_limits(cgroup_directories(proc, cgroup_root))
    count = min(len(physical), max(1, math.ceil(quota))) if quota is not None else len(physical)
    memory = next(int(line.split()[1])*1024 for line in (proc/'meminfo').read_text().splitlines()
                  if line.startswith('MemAvailable:'))
    if memory_limit is not None:
        memory = min(memory, memory_limit)
    selected = physical[:count]
    return {'allowed_logical_cpus': allowed, 'physical_cpus': physical,
            'selected_cpus': selected, 'worker_count': count, 'cpu_quota': quota,
            'numa_nodes': {node: [cpu for cpu in selected if numa[cpu] == node]
                           for node in sorted({numa[cpu] for cpu in selected})},
            'memory_available_bytes': memory}


def cpu_list(cpus):
    if not cpus:
        raise ValueError('empty build CPU affinity')
    return ','.join(map(str, cpus))


def split_workers(resources, dimension, rows):
    # Each spawned interpreter imports NumPy and gathers/copies a 16K-row block.
    # Bound those private allocations; mmap pages are shared by all processes.
    per_worker = (256 << 20) + 3 * 16384 * dimension * 4
    memory_workers = resources['memory_available_bytes'] * 4 // 5 // per_worker
    if memory_workers < 1:
        raise ValueError('insufficient available memory for shard materialization')
    return min(resources['worker_count'], memory_workers, rows)


def resolve(snapshot, dimension, rows, controller):
    hosts = {host: state['resources'] for host, state in snapshot.items()}
    for host, resources in hosts.items():
        if resources['worker_count'] != len(resources['selected_cpus']) or resources['worker_count'] < 1:
            raise ValueError(f'invalid build resources for {host}')
    return {'policy': 'all_available_physical_cores', 'hosts': hosts,
            'attachment_threads_per_host': {host: row['worker_count'] for host, row in hosts.items()},
            'lower_threads_per_host': {host: row['worker_count'] for host, row in hosts.items()},
            'split_processes': split_workers(hosts[controller], dimension, rows),
            'search_resources_changed': False}


def save_plan(root, snapshot, phase, dimension=128, rows=100000000, controller=None):
    controller = controller or next(iter(snapshot))
    # Local materialization/export inherits this process's affinity/cgroup,
    # which can differ from an SSH session on the same machine.
    snapshot = dict(snapshot)
    snapshot[controller] = dict(snapshot[controller], resources=detect())
    plan = resolve(snapshot, dimension, rows, controller)
    path = Path(root)/'evidence'/f'build-resources-{phase}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(plan, indent=2) + '\n')
    temporary.replace(path)
    return plan


def read_plan(root, phase):
    return json.loads((Path(root)/'evidence'/f'build-resources-{phase}.json').read_text())


def weighted_ranges(start, end, weights):
    total = sum(weights)
    if not weights or min(weights) < 1 or end < start:
        raise ValueError('invalid attachment ranges/worker weights')
    boundaries, accumulated = [start], 0
    for weight in weights:
        accumulated += weight
        boundaries.append(start + (end-start)*accumulated//total)
    return list(zip(boundaries, boundaries[1:]))


def _digest_range(path, offset, size):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ValueError(f'truncated file while hashing: {path}')
            digest.update(block)
            remaining -= len(block)
    return {'size': size, 'sha256': digest.hexdigest()}


def hash_tar(path, workers):
    # Independent file descriptors: a shared TarFile seek cursor is not thread safe.
    with tarfile.open(path, 'r:') as archive:
        entries = [(m.name, m.offset_data, m.size) for m in archive if m.isfile()]
    def one(entry):
        name, offset, size = entry
        return name, _digest_range(path, offset, size)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, len(entries)))) as pool:
        return dict(pool.map(one, entries))


def hash_tree(root, workers):
    root = Path(root)
    files = sorted(path for path in (root/'storage').rglob('*') if path.is_file())
    def one(path):
        return str(path.relative_to(root)), _digest_range(path, 0, path.stat().st_size)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, len(files)))) as pool:
        return dict(pool.map(one, files))


if __name__ == '__main__':
    print(json.dumps(detect()))
