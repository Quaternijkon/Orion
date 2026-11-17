#!/usr/bin/env python3
"""Continue a frozen matched experiment after layout preparation.

Build one native M32/ef_construct200 lower index on each of eight hosts,
retain verified archives, and measure the two approved recall curves. No QPS.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time


def main(root):
    source = root / "source-build48" if (root / "source-build48").is_dir() else root / "source"
    sys.path.insert(0, str(source))
    import numpy as np
    import admission
    import orion_build_resources as resources
    import shard_locality as core
    import shard_locality_archives as archive
    hosts = list(admission.HOSTS)
    resource_plan = None
    native_binary = root/'bin/shard_locality_worker'
    if not native_binary.is_file():
        native_binary = root/'bin/shard_locality_build48'  # historical frozen workspaces
    runtime = core.read_json(root/'runtime.json') if (root/'runtime.json').is_file() else {}
    km_folder = Path(runtime.get('kmeans_archives', '/home/wangyiduo/ann-native-orion-aligned-20261114/archives/sift100m/kmeans'))
    token = root.name
    remote = Path('/dev/shm') / token

    def save(stage, **data):
        value = dict(stage=stage, time=time.time(), pid=os.getpid(), qps_measured=False, **data)
        path = root / "continuation.tmp.json"
        path.write_text(json.dumps(value, indent=2)); path.replace(root / "continuation.json")
        print(json.dumps(value), flush=True)

    def health(lease=None):
        if (root / 'STOP').exists(): raise KeyboardInterrupt()
        if lease is not None: lease.check()

    def ssh(host, code):
        result = subprocess.run(admission.ssh_args(host) + ['python3 -c ' + shlex.quote(code)],
                                text=True, capture_output=True, check=True, timeout=120)
        return result.stdout

    def cleanup(host):
        # Stop only this run's recorded process group, never another experiment.
        code = f'''import pathlib,json,os,signal
p=pathlib.Path({str(remote)!r});marker=p/'OWNER'
if not p.exists():raise SystemExit(0)
assert marker.read_text()=={token!r}
f=p/'build-process.json'
if f.exists():
 r=json.loads(f.read_text());pid=r['pid']
 try:
  stat=pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
  cmd=pathlib.Path('/proc',str(pid),'cmdline').read_bytes()
  if int(stat[19])==r['start_ticks'] and {str(remote).encode()!r} in cmd:os.killpg(pid,signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError):pass
'''
        ssh(host, code)

    def split_vectors(manifest):
        import shard_locality_split
        workers=resource_plan['split_processes']
        counts = shard_locality_split.split(
            root/'layout'/manifest['assignments_file'], root/'layout'/manifest['vectors_file'],
            root/'shard-inputs', manifest['point_count'], manifest['dimension'],
            manifest['total_point_copies'], shards=len(hosts), workers=workers,
            cpus=resource_plan['hosts'][hosts[0]]['selected_cpus'],
            progress=lambda rows,parts: save('splitting_shard_inputs',rows=rows,completed_parts=parts,processes=workers))
        core.write_json(root/'evidence/shard-counts.json',counts)
        return counts

    def build_one(shard, host, manifest, collection_config):
        threads=resource_plan['lower_threads_per_host'][host]
        cpus=resource_plan['hosts'][host]['selected_cpus']
        ssh(host, f"import pathlib,shutil; p=pathlib.Path({str(remote)!r}); "
            "assert not p.exists(),'independent staging already exists'; "
            f"assert shutil.disk_usage('/dev/shm').free>60*1024**3; p.mkdir(); (p/'OWNER').write_text({token!r})")
        local=root/'shard-inputs'
        request={'vectors':str(remote/f'{shard}.f32le'),'ids':str(remote/f'{shard}.u64le'),
                 'output':str(remote/f'storage/collections/sift100m/{shard}/segments'),
                 'dimension':128,'distance':'Euclid','m':32,'ef_construct':200,'threads':threads,'seed':100}
        request_path=root/'config'/f'build-shard-{shard}.json'
        core.write_json(request_path,request)
        with (root/'logs'/f'lower-{shard}.log').open('x') as log:
            paths=[native_binary,local/f'{shard}.f32le',local/f'{shard}.u64le',request_path]
            subprocess.run(['rsync','-a','--protect-args',*map(str,paths),f'{host}:{remote}/'],check=True,stdout=log,stderr=subprocess.STDOUT)
            argv=['taskset','-c',resources.cpu_list(cpus),str(remote/native_binary.name),
                  'build-segment',str(remote/request_path.name),str(remote/'built.jsonl')]
            code=f'''import subprocess,pathlib,json
p=pathlib.Path({str(remote)!r})
c=subprocess.Popen({argv!r},start_new_session=True)
s=pathlib.Path('/proc',str(c.pid),'stat').read_text().rsplit(')',1)[1].split()
(p/'build-process.json').write_text(json.dumps({{'pid':c.pid,'start_ticks':int(s[19])}}))
raise SystemExit(c.wait())'''
            subprocess.run(admission.ssh_args(host)+['python3 -c '+shlex.quote(code)],check=True,stdout=log,stderr=subprocess.STDOUT)
        config_path=remote/'storage/collections/sift100m/config.json'
        ssh(host,f"import pathlib,json; pathlib.Path({str(config_path)!r}).write_text({json.dumps(collection_config)!r})")
        dest=root/'orion-archives';dest.mkdir(exist_ok=True)
        archive_path=dest/f'{host}.tar'
        # Archive just this host's one immutable data shard and its configuration.
        with archive_path.open('xb') as sink:
            subprocess.run(admission.ssh_args(host)+[shlex.join(['tar','-C',str(remote),'-cf','-','storage'])],
                           stdout=sink,check=True)
        # Eight archives may arrive together; divide the controller CPU budget
        # between them instead of creating eight full-size hashing pools.
        hash_workers=max(1,resource_plan['hosts'][hosts[0]]['worker_count']//len(hosts))
        records={archive.member_name(name):value for name,value in
                 resources.hash_tar(archive_path,hash_workers).items()}
        core.write_json(dest/f'{host}.manifest.json',records)
        # Compare hashes computed independently on the source host to the archive.
        actual=json.loads(ssh(host,
            'import json\nbuild_resources={"__name__":"orion_build_resources"}\n'
            f'exec({Path(resources.__file__).read_text()!r},build_resources)\n'
            f'print(json.dumps(build_resources["hash_tree"]({str(remote)!r},{threads})))'))
        if actual!=records:raise ValueError(f'{host}: archive/source mismatch')
        core.write_json(dest/f'{host}.verified.json',{'host':host,'shard':shard,'verified':True,
                                                   'build_threads':threads,'cpu_affinity':cpus})
        ssh(host,f"import pathlib,shutil; p=pathlib.Path({str(remote)!r}); assert (p/'OWNER').read_text()=={token!r}; shutil.rmtree(p)")
        return shard

    def measure(config, manifest, orion_records, km_records, artifact):
        output=root/'results';output.mkdir()
        config['orion_manifest']=str(root/'layout/import.manifest.json')
        core.write_json(output/'config.json',config)
        core.write_json(output/'kmeans-artifact.json',artifact)
        _,truth=core.load_queries(config,manifest,output)
        np.unique(truth).astype('<u8').tofile(output/'truth-ids.u64le')
        core.write_json(output/'provenance.json',{'hosts':hosts,'shards_per_host':1,
            'm':32,'ef_construct':200,'orion_build_threads_by_host':resource_plan['lower_threads_per_host'],
            'kmeans_historical_build_threads':8,
            'qps_measured':False,'target_recall_calibration':False,
            'queries_sha256':core.sha256(output/'queries.f32le'),'truth_sha256':core.sha256(output/'truth.npy'),
            'native_binary_sha256':core.sha256(native_binary),
            'orion_artifact_sha256':manifest['orion_artifact_sha256'],
            'kmeans_artifact_sha256':core.sha256(output/'kmeans-artifact.json'),
            'query_count':len(truth),'top_k':10,'metric_schema':core.METRIC_SCHEMA,
            'source_script_sha256':core.sha256(Path(__file__))})
        files={k:(root/'layout'/manifest[k+'_file']).resolve() for k in ['vectors','assignments','orion_artifact']}
        all_rows=[[],[],[]]
        for method,records,path in [('orion',orion_records,files['orion_artifact']),('kmeans',km_records,output/'kmeans-artifact.json')]:
            save('measuring_recall',method=method)
            result=archive.search_method(config,manifest,files,native_binary,output,method,truth,records,path)
            for total,rows in zip(all_rows,result):total.extend(rows)
        for name,rows in zip(['curves','per_query','real_by_fanout'],all_rows):core.write_csv(output/f'{name}.csv',rows)
        core.plot(all_rows[0],output,10,8)
        core.write_json(output/'COMPLETED.json',{'complete':True,'qps_measured':False,'hosts':hosts,'m':32,'ef_construct':200,'metric_schema':core.METRIC_SCHEMA})
        archive.save_status(output, 'complete', methods=['orion', 'kmeans'], query_count=len(truth))

    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        while True:
            health()
            if not (root/'status.json').is_file():
                time.sleep(2);continue
            state=core.read_json(root/'status.json')
            if state['stage']=='layout_ready_lower_indexes_pending':break
            if state['stage'] in ('failed','interrupted'):raise RuntimeError('layout preparation did not finish successfully')
            save('waiting_for_layout',layout_stage=state['stage']);time.sleep(30)
        while True:
            health()
            snapshot=admission.probe_all()
            if admission.available(snapshot):break
            save('waiting_for_eight_hosts');time.sleep(30)
        with admission.Lease() as lease:
            try:
                manifest=core.read_json(root/'layout/import.manifest.json')
                resource_plan=resources.save_plan(root,snapshot,'lower',manifest['dimension'],
                                                  manifest['point_count'],hosts[0])
                config=core.read_json(root/'config/experiment.json')
                config['orion_manifest']=str(root/'layout/import.manifest.json')
                archive.require_upper_build(config)
                km_records,km_config,artifact=archive.inspect_archives(km_folder,'kmeans')
                archive.require_matched_indexes(config,[('kmeans',km_records,km_config)])
                split_vectors(manifest)
                collection_config=json.loads(json.dumps(km_config))
                # Collection policy is identical across archives; each physical
                # graph and its build request record the resolved host budget.
                collection_config['hnsw_config']['max_indexing_threads']=0
                collection_config['auto_shard_policy']={'type':'orion','generation':manifest['orion_generation'],
                                                        'artifact_sha256':manifest['orion_artifact_sha256']}
                pool=concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts))
                pending={pool.submit(build_one,s,h,manifest,collection_config) for s,h in enumerate(hosts)}
                try:
                    while pending:
                        health(lease)
                        done,pending=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
                        for f in done:f.result()
                        save('building_eight_lower_indexes',remaining=len(pending),m=32,ef_construct=200,
                             threads_per_host=resource_plan['lower_threads_per_host'])
                except BaseException:
                    for host in hosts:
                        try:cleanup(host)
                        except Exception:pass
                    raise
                finally:pool.shutdown(wait=True)
                orion_records,orion_config,_=archive.inspect_archives(root/'orion-archives','orion')
                archive.require_matched_indexes(config,[('orion',orion_records,orion_config),('kmeans',km_records,km_config)])
                core.write_json(root/'evidence/matched-index-preflight.json',{'orion':orion_records,'kmeans':km_records,
                    'orion_config':orion_config,'kmeans_config':km_config,'qps_measured':False})
                measure(config,manifest,orion_records,km_records,artifact)
                save('complete')
            finally:
                for host in hosts:
                    try:cleanup(host)
                    except Exception:pass
    except KeyboardInterrupt:save('interrupted')
    except Exception as error:
        save('failed',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    main(parser.parse_args().root.resolve())
