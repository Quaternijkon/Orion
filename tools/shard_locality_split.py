"""Parallel, order-preserving materialization of existing native memberships."""
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import shutil

if __package__:
    from . import orion_build_resources as resources
else:
    import orion_build_resources as resources


def byte_ranges(path, workers):
    """Partition JSONL at newline boundaries, never splitting or duplicating a row."""
    path = Path(path)
    size = path.stat().st_size
    offsets = [0]
    with path.open('rb') as source:
        for i in range(1, workers):
            position = size * i // workers
            if position:
                source.seek(position - 1)
                source.readline()
                position = source.tell()
            offsets.append(position)
    offsets.append(size)
    return list(zip(offsets, offsets[1:]))


def _split_range(args):
    import numpy as np
    index, start, end, assignment, vectors, count, dimension, shards, output = args
    folder = Path(output) / f'part-{index:03}'
    folder.mkdir()
    source = np.memmap(vectors, dtype='<f4', mode='r', shape=(count, dimension))
    vector_files = [(folder / f'{s}.f32le').open('xb') for s in range(shards)]
    id_files = [(folder / f'{s}.u64le').open('xb') for s in range(shards)]
    counts = [0] * shards
    first, previous, rows = None, None, 0
    try:
        with Path(assignment).open('rb') as stream:
            stream.seek(start)
            while stream.tell() < end:
                positions = [[] for _ in range(shards)]
                batch_start, batch_count = None, 0
                for _ in range(16384):
                    if stream.tell() >= end:
                        break
                    row = json.loads(stream.readline())
                    point, owners = row['id'], row['shards']
                    if (isinstance(point, bool) or not isinstance(point, int) or not 0 <= point < count or
                            (previous is not None and point != previous + 1)):
                        raise ValueError('noncontiguous or invalid point IDs')
                    if not owners or len(set(owners)) != len(owners):
                        raise ValueError('invalid duplicate/empty membership')
                    if first is None: first = point
                    if batch_start is None: batch_start = point
                    previous = point
                    for s in owners:
                        if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < shards:
                            raise ValueError('invalid shard ID')
                        positions[s].append(batch_count)
                    batch_count += 1
                    rows += 1
                block = source[batch_start:batch_start + batch_count]
                for shard, pos in enumerate(positions):
                    if not pos: continue
                    vector_files[shard].write(block[pos].tobytes())
                    id_files[shard].write((np.asarray(pos, dtype='<u8') + batch_start).tobytes())
                    counts[shard] += len(pos)
    finally:
        for stream in vector_files + id_files:
            stream.close()
    return dict(index=index, first=first, last=previous, rows=rows, counts=counts)


def _pin_worker(cpus):
    os.sched_setaffinity(0, cpus)


def split(assignment, vectors, output, count, dimension, expected_copies, shards=8, workers=None,
          progress=None, cpus=None):
    if count < 1 or dimension < 1:
        raise ValueError('empty shard materialization input')
    if workers is None:
        available = resources.detect()
        workers = resources.split_workers(available, dimension, count)
        cpus = available['selected_cpus'] if cpus is None else cpus
    if workers < 1:
        raise ValueError('materialization workers must be positive')
    workers = min(workers, count)
    cpus = sorted(os.sched_getaffinity(0)) if cpus is None else cpus
    output = Path(output)
    output.mkdir(exist_ok=False)
    ranges = byte_ranges(assignment, workers)
    args = [(i, a, b, str(assignment), str(vectors), count, dimension, shards, str(output))
            for i, (a, b) in enumerate(ranges)]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_pin_worker, initargs=(cpus,)) as pool:
        jobs = [pool.submit(_split_range, arg) for arg in args]
        for job in concurrent.futures.as_completed(jobs):
            results.append(job.result())
            if progress: progress(sum(r['rows'] for r in results), len(results))
    results.sort(key=lambda row: row['index'])
    cursor = 0
    for result in results:
        if not result['rows']: continue
        if result['first'] != cursor or result['last'] != cursor + result['rows'] - 1:
            raise ValueError('range boundary lost or duplicated IDs')
        cursor += result['rows']
    counts = [sum(r['counts'][s] for r in results) for s in range(shards)]
    if cursor != count or sum(counts) != expected_copies:
        raise ValueError('parallel split point/copy count mismatch')

    def merge(task):
        shard, suffix, width = task
        path = output / f'{shard}.{suffix}'
        with path.open('xb') as sink:
            for i in range(workers):
                with (output / f'part-{i:03}' / path.name).open('rb') as source:
                    shutil.copyfileobj(source, sink, 8 << 20)
        if path.stat().st_size != counts[shard] * width:
            raise ValueError('parallel split byte size mismatch')
    tasks = [(shard, suffix, width) for shard in range(shards)
             for suffix, width in [('u64le', 8), ('f32le', dimension * 4)]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(cpus), len(tasks))) as pool:
        list(pool.map(merge, tasks))
    (output / 'split-proof.json').write_text(json.dumps({'counts': counts, 'ranges': results,
        'source_order_preserved': True, 'workers': workers, 'cpu_affinity': cpus}, indent=2))
    for i in range(workers):
        shutil.rmtree(output / f'part-{i:03}')
    return counts
