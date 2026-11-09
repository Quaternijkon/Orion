#!/usr/bin/env python3
"""Paired, same-index hash baseline comparison of two frozen repository builds.

Uses the native-hash experiment's client and collection preparation protocol.
All processes and files belong to a separate experiment directory.
"""
import argparse
import asyncio
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import aiohttp

ROOT = Path('/home/wangyiduo/qdrant-repo-ab-20261022')
PROTOCOL = Path(__file__).resolve().parents[1] / 'docs/experiments/qdrant-hash-sift1m-20261022/driver_snapshot.py'
spec = importlib.util.spec_from_file_location('native_hash', PROTOCOL)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.ROOT = ROOT
base.PORT = 17333
base.__file__ = str(Path(__file__).resolve())


def stop(n, label):
    def one(host):
        directory = ROOT / f'n{n}'
        # PIDs are accepted only when the executable is one of this study's builds.
        return base.ssh(host, f'''set -e
p=$(cat {directory}/pid)
exe=$(readlink /proc/$p/exe || true)
case "$exe" in
 {ROOT}/bin/orion|{ROOT}/bin/orion_rust) kill -TERM "$p" ;;
 "") ;;
 *) echo "Refusing to stop unexpected executable: $exe" >&2; exit 1 ;;
esac
for i in $(seq 1 60); do
 if ! kill -0 "$p" 2>/dev/null; then mv {directory}/pid {directory}/pid-{label}; cp {directory}/server.log {directory}/server-{label}.log; exit 0; fi
 state=$(awk '{{print $3}}' /proc/$p/stat 2>/dev/null || true)
 if [ "$state" = Z ]; then mv {directory}/pid {directory}/pid-{label}; cp {directory}/server.log {directory}/server-{label}.log; exit 0; fi
 sleep 1
done
exit 1''')
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(one, base.HOSTS[:n]))


def start(n, variant):
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda h: base.ssh(h, f'ln -sfn {ROOT}/bin/{variant} {ROOT}/bin/qdrant'), base.HOSTS[:n]))
    base.deploy(n)


def graph_manifest(n):
    # Exact graph bytes are checked across every binary switch. No rebuilding is allowed.
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        values = list(pool.map(lambda h: base.ssh(h,
            f"find {ROOT}/n{n}/storage -type f -path '*/vector_index/*' -print0 | sort -z | xargs -0 -r sha256sum"), base.HOSTS[:n]))
    if not all('graph' in v for v in values):
        raise RuntimeError('No graph files found on at least one host')
    return dict(zip(base.HOSTS[:n], values))


def resources(n):
    code=f'''import json,time,os
from pathlib import Path
p=int(Path("{ROOT}/n{n}/pid").read_text())
threads=[]
for entry in Path(f"/proc/{{p}}/task").iterdir():
 try:
  raw=(entry/'stat').read_text(); fields=raw.rsplit(')',1)[1].split()
  threads.append(dict(tid=int(entry.name),name=raw.split('(',1)[1].rsplit(')',1)[0],
   user_ticks=int(fields[11]),system_ticks=int(fields[12]),cpu=int(fields[36])))
 except FileNotFoundError:pass
print(json.dumps(dict(pid=p,monotonic=time.monotonic(),ticks_per_second=os.sysconf('SC_CLK_TCK'),
 threads=threads,loadavg=Path('/proc/loadavg').read_text(),net=Path('/proc/net/dev').read_text())))'''
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        values=list(pool.map(lambda h:json.loads(base.ssh(h,'python3 -c '+base.shlex.quote(code))),base.HOSTS[:n]))
    return dict(zip(base.HOSTS[:n],values))


async def ready(n, out, variant):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        for _ in range(90):
            try:
                states = await asyncio.gather(*(base.api(session,h,'GET','/cluster') for h in base.HOSTS[:n]))
                info = (await base.api(session,base.HOSTS[0],'GET','/collections/sift1m'))['result']
                placements = await asyncio.gather(*(base.api(session,h,'GET','/collections/sift1m/cluster') for h in base.HOSTS[:n]))
                good = (all(len(s['result']['peers']) == n for s in states)
                    and info['status'] == 'green' and info['optimizer_status'] == 'ok'
                    and info['points_count'] == 1000000 and info['indexed_vectors_count'] >= 990000
                    and all(len(p['result']['local_shards']) == 1
                        and p['result']['local_shards'][0]['state'] == 'Active'
                        and not p['result']['shard_transfers'] for p in placements))
                if good:
                    versions=await asyncio.gather(*(base.api(session,h,'GET','/') for h in base.HOSTS[:n]))
                    base.save(out/'versions.json',versions)
                    expected=json.loads((ROOT/'sources.json').read_text())[variant]['commit']
                    if not all(v['version']=='1.17.1' and v['commit']==expected for v in versions):
                        raise ValueError('Running server version/commit does not match the intended variant')
                    base.save(out/'collection.json',info)
                    base.save(out/'placement.json',placements)
                    return
            except (aiohttp.ClientError,RuntimeError):
                pass
            await asyncio.sleep(2)
    raise RuntimeError('Restarted cluster is not ready')


async def quality(out):
    with base.h5py.File(ROOT/'sift.hdf5') as f:
        queries, truth = f['test'][:], f['neighbors'][:, :10]
    ids = base.np.full((len(queries),10),-1,dtype=base.np.int64)
    scores = base.np.full((len(queries),10),base.np.nan)
    recall = base.np.zeros(len(queries))
    semaphore = asyncio.Semaphore(32)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        async def one(i):
            async with semaphore:
                answer = (await base.api(session,base.HOSTS[0],'POST','/collections/sift1m/points/search',
                    dict(vector=queries[i].tolist(),limit=10,params={'hnsw_ef':128},
                         with_payload=False,with_vector=False)))['result']
                if len(answer) != 10:
                    raise RuntimeError('Incomplete top-10 response')
                ids[i] = [p['id'] for p in answer]
                scores[i] = [p['score'] for p in answer]
                recall[i] = len(set(ids[i]) & set(truth[i]))/10
        await asyncio.gather(*(one(i) for i in range(len(queries))))
    base.np.save(out/'recall_per_query.npy',recall)
    base.np.savez_compressed(out/'query_results.npz',ids=ids,scores=scores)
    base.save(out/'recall.json',dict(queries=len(queries),recall_at_10=float(recall.mean()),hnsw_ef=128))


def run(args):
    with (ROOT/'sift.hdf5').open('rb') as stream:
        digest=base.hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!='dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984':
        raise RuntimeError('Dataset checksum mismatch')
    results = Path(args.output)
    results.mkdir(parents=True,exist_ok=True)
    sequence = ['orion','orion_rust','orion_rust','orion','orion','orion_rust']
    for n in args.nodes:
        folder = results/f'n{n}'
        if (folder/'complete.json').exists():
            continue
        folder.mkdir(exist_ok=True)
        if not (folder/'graphs.json').exists():
            start(n,'orion')
            try:
                asyncio.run(base.prepare(n,folder))
            finally:
                stop(n,args.run_label+'build')
            base.save(folder/'graphs.json',graph_manifest(n))
        if args.mode=='prepare':
            continue
        graphs = json.loads((folder/'graphs.json').read_text())
        for block,variant in enumerate(sequence):
            out = folder/f'block{block}-{variant}'
            if (out/'results.json').exists():
                continue
            out.mkdir(exist_ok=True)
            if graph_manifest(n) != graphs:
                raise RuntimeError('Graph bytes changed before restart')
            start(n,variant)
            try:
                asyncio.run(ready(n,out,variant))
                asyncio.run(quality(out))
                base.measure(n,out,32,'warmup',15)
                measurements=[]
                for c in ([1,32,128] if block%2==0 else [128,32,1]):
                    if args.run_label:
                        base.save(out/f'c{c}-r{block}-resources-before.json',resources(n))
                    value=base.measure(n,out,c,block,args.seconds)
                    if args.run_label:
                        base.save(out/f'c{c}-r{block}-resources-after.json',resources(n))
                    value.update(variant=variant,block=block)
                    measurements.append(value)
            finally:
                stop(n,f'{args.run_label}block{block}-{variant}')
            if graph_manifest(n) != graphs:
                raise RuntimeError('Graph bytes changed during measurement')
            base.save(out/'results.json',measurements)
        base.save(folder/'complete.json',dict(nodes=n,sequence=sequence,graph_bytes_unchanged=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','prepare','worker'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--nodes',type=int,nargs='+',default=list(range(1,9)))
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--concurrency',type=int,default=1)
    parser.add_argument('--seconds',type=float,default=20)
    parser.add_argument('--run-label',default='')
    args=parser.parse_args()
    if args.mode=='worker':
        asyncio.run(base.load_worker(args))
    else:
        run(args)
