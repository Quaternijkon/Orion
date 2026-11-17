#!/usr/bin/env python3
"""Migrate an active attachment pass to 8 hosts without changing the graph/results.

Keep original preparation and its leases alive until a complete replacement is
validated. Retain complete prefix records; every worker replays the same parity
sample and independently checks graph/vector/output checksums. Resume the
canonical capacity builder with --reuse-prepared after an audited handover.
"""
import concurrent.futures
import hashlib
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
    sys.path.insert(0,str(root/'source'))
    import admission
    import orion_build_resources as resources
    resource_plan = resources.read_plan(root, 'prepare')
    work=root/'parallel-attachments'
    binary=root/'bin/orion_attachment_parallel'
    prepare=json.loads((root/'controller.json').read_text())
    continuation=json.loads((root/'continuation-process.json').read_text())
    old_py=json.loads((root/'status.json').read_text())['child_pid']
    children=Path(f'/proc/{old_py}/task/{old_py}/children').read_text().split()
    if len(children)!=1:raise RuntimeError('ambiguous native preparation process')
    native=int(children[0]);stage=None
    for fd in Path(f'/proc/{native}/fd').iterdir():
        try:
            target=Path(os.readlink(fd))
            if target.name=='upper-attachments.counted.bin':stage=target.parent
        except OSError:pass
    if stage is None:raise RuntimeError('native preparation is not in the attachment stage')
    prefix_file=stage/'upper-attachments.counted.bin'
    graph=stage/'upper-source-generation-1.json'
    vectors=stage/'import.f32le'
    remote=Path('/dev/shm')/f'orion-attachments-{os.getpid()}'
    halted=False;handover=False;cont_stopped=False
    record_bytes=84;parity_rows=8192
    parity=work/'parity'
    if not json.loads((parity/'verified.json').read_text())['byte_identical_to_original_native_prepare_capacity']:
        raise RuntimeError('missing serial/parallel parity proof')

    def sha(path):
        with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    def write(path,value):
        path=Path(path);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
    def status(state,**details):
        value={'stage':state,'time':time.time(),'pid':os.getpid(),
               'threads_per_host':resource_plan['attachment_threads_per_host'],**details}
        write(work/'status.json',value);print(json.dumps(value),flush=True)
    def owned(record,needle):
        p=Path('/proc')/str(record['pid'])
        s=(p/'stat').read_text().rsplit(')',1)[1].split()
        return int(s[19])==record['start_ticks'] and needle.encode() in (p/'cmdline').read_bytes()
    def health():
        if (root/'STOP').exists():raise KeyboardInterrupt()
        if not owned(prepare,'shard_locality_prepare.py'):raise RuntimeError('original lease controller disappeared')
    def remote_python(host,code):
        p=subprocess.run(admission.ssh_args(host)+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=120)
        return p.stdout
    def copy_bytes(source,dest,n):
        while n:
            data=source.read(min(n,8<<20))
            if not data:raise EOFError('short range input')
            dest.write(data);n-=len(data)
    def stop_remote(host):
        remote_python(host,f'''import pathlib,os,signal,json
p=pathlib.Path({str(remote)!r})
if not (p/'process.json').exists():raise SystemExit(0)
r=json.loads((p/'process.json').read_text());pid=r['pid']
try:
 s=pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
 cmd=pathlib.Path('/proc',str(pid),'cmdline').read_bytes()
 if int(s[19])==r['start_ticks'] and {str(remote).encode()!r} in cmd:os.killpg(pid,signal.SIGTERM)
except (FileNotFoundError,ProcessLookupError):pass
''')
    graph_sha=sha(graph)
    def one(index,start,end):
        host=admission.HOSTS[index];folder=work/f'host-{index}';folder.mkdir()
        threads=resource_plan['attachment_threads_per_host'][host]
        cpus=resource_plan['hosts'][host]['selected_cpus']
        affinity=['taskset','-c',resources.cpu_list(cpus)]
        chunk=folder/'vectors.f32le'
        with chunk.open('xb') as out:
            with (parity/'vectors.f32le').open('rb') as src:shutil.copyfileobj(src,out,8<<20)
            with vectors.open('rb') as src:
                src.seek(start*128*4);copy_bytes(src,out,(end-start)*128*4)
        vectors_sha=sha(chunk)
        local=index==0
        location=folder if local else remote
        request={'artifact':str(graph if local else remote/graph.name),
                 'vectors':str(chunk if local else remote/chunk.name),'rows':end-start+parity_rows,
                 'dimension':128,'efs':10,'threads':threads,'output':str(location/'export')}
        write(folder/'request.json',request)
        with (folder/'worker.log').open('x') as log:
            if local:
                command=affinity+[str(binary),'export',str(folder/'request.json')]
            else:
                remote_python(host,f"import pathlib,shutil; p=pathlib.Path({str(remote)!r}); assert not p.exists(); assert shutil.disk_usage('/dev/shm').free>12*1024**3; p.mkdir()")
                subprocess.run(['rsync','-a','--protect-args',str(binary),str(graph),str(chunk),str(folder/'request.json'),f'{host}:{remote}/'],check=True,stdout=log,stderr=subprocess.STDOUT)
                argv=affinity+[str(remote/binary.name),'export',str(remote/'request.json')]
                code=f'''import pathlib,subprocess,json
p=pathlib.Path({str(remote)!r});c=subprocess.Popen({argv!r},start_new_session=True)
s=pathlib.Path('/proc',str(c.pid),'stat').read_text().rsplit(')',1)[1].split()
(p/'process.json').write_text(json.dumps({{'pid':c.pid,'start_ticks':int(s[19])}}))
raise SystemExit(c.wait())'''
                command=admission.ssh_args(host)+['python3 -c '+shlex.quote(code)]
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if local:
                ticks=int(Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
                write(folder/'process.json',{'pid':process.pid,'start_ticks':ticks})
            if process.wait():raise RuntimeError(f'{host}: export failed; see worker.log')
            if not local:
                (folder/'export').mkdir()
                subprocess.run(['rsync','-a','--protect-args',f'{host}:{remote}/export/',str(folder/'export')+'/'],check=True,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads((folder/'export/manifest.json').read_text())
        hits=folder/'export/hits.bin'
        if (result['rows']!=end-start+parity_rows or result['threads']!=threads or result['artifact_sha256']!=graph_sha or
                result['vectors_sha256']!=vectors_sha or result['hits_sha256']!=sha(hits)):
            raise RuntimeError(f'worker checksum/range mismatch: {host}')
        with hits.open('rb') as stream:
            if stream.read(parity_rows*record_bytes)!=(parity/'expected.bin').read_bytes():
                raise RuntimeError(f'cross-machine byte parity failed: {host}')
        if hits.stat().st_size!=(end-start+parity_rows)*record_bytes:raise RuntimeError('record shape mismatch')
        proof={'host':host,'start':start,'end':end,'parity_rows':parity_rows,'byte_parity':True,
               'cpu_affinity':cpus,'output':str(hits),**result}
        write(folder/'verified.json',proof)
        if not local:
            remote_python(host,f"import pathlib,shutil;p=pathlib.Path({str(remote)!r}); assert p.parent==pathlib.Path('/dev/shm') and p.name.startswith('orion-attachments-'); shutil.rmtree(p)")
        return proof

    try:
        health()
        os.kill(native,signal.SIGSTOP);halted=True
        # Only complete fixed-width records are retained; the native BufWriter
        # may have an incomplete last record which is recomputed in the suffix.
        prefix_rows=prefix_file.stat().st_size//record_bytes
        total=100000000
        ranges=resources.weighted_ranges(prefix_rows,total,
            [resource_plan['attachment_threads_per_host'][host] for host in admission.HOSTS])
        write(work/'migration.json',{'native_pid':native,'original_controller':prepare,'original_child_pid':old_py,
              'continuation':continuation,'source_stage':str(stage),'prefix_rows':prefix_rows,'ranges':ranges,
              'artifact_sha256':graph_sha,'experiment_parameters_unchanged':True})
        prepared=root/'prepared-parallel';prepared.mkdir()
        for name in ['import.f32le','upper-source-generation-1.json']:
            os.link(stage/name,prepared/name)
        with prefix_file.open('rb') as src,(prepared/'upper-attachments.counted.bin').open('xb') as out:
            copy_bytes(src,out,prefix_rows*record_bytes)
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=len(admission.HOSTS))
        pending={pool.submit(one,i,a,b) for i,(a,b) in enumerate(ranges)};proofs=[]
        try:
            while pending:
                health()
                done,pending=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
                for job in done:proofs.append(job.result())
                status('distributed_attachment_generation',completed_hosts=len(proofs),retained_prefix_rows=prefix_rows,
                       worker_ranges=ranges,remaining_hosts=len(pending))
        except BaseException:
            path=work/'host-0/process.json'
            if path.exists():
                record=json.loads(path.read_text())
                if owned(record,str(work/'host-0/request.json')):
                    os.killpg(record['pid'],signal.SIGTERM)
            for host in admission.HOSTS[1:]:
                try:stop_remote(host)
                except Exception:pass
            raise
        finally:pool.shutdown(wait=True)
        with (prepared/'upper-attachments.counted.bin').open('ab') as out:
            for proof in sorted(proofs,key=lambda p:p['start']):
                with Path(proof['output']).open('rb') as src:
                    src.seek(parity_rows*record_bytes);copy_bytes(src,out,(proof['end']-proof['start'])*record_bytes)
            out.flush();os.fsync(out.fileno())
        status('validating_merged_preparation',records=total)
        write(work/'finalize-request.json',{'request':str(root/'config/build.json'),'directory':str(prepared)})
        with (work/'finalize.log').open('x') as log:
            subprocess.run([str(binary),'finalize',str(work/'finalize-request.json')],stdout=log,stderr=log,check=True)
        health()
        write(work/'completed.json',{'validated':True,'prefix_rows':prefix_rows,'total_rows':total,
                                    'workers':proofs,'artifact_sha256':graph_sha})
        # All inputs are preserved and validated before interrupting the old
        # suspended producer. SIGKILL deliberately avoids its TempDir cleanup.
        if not owned(continuation,'shard_locality_continue.py'):raise RuntimeError('continuation process changed')
        os.kill(continuation['pid'],signal.SIGSTOP);cont_stopped=True
        os.kill(prepare['pid'],signal.SIGSTOP)
        for pid in [old_py,native,prepare['pid']]:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        handover=True;halted=False
        # The replacement controller skips only the already validated frozen
        # graph+attachment phase; native access/capacity logic is unchanged.
        prepare_source=Path(__file__).with_name('shard_locality_prepare.py')
        prepare_target=root/'source/shard_locality_prepare.py'
        if prepare_source.resolve()!=prepare_target.resolve():
            shutil.copy2(prepare_source,prepare_target)
        write(root/'status.json',{'stage':'restarting_from_parallel_prepared','time':time.time()})
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (root/'logs/controller-resumed.log').open('x') as log:
            child=subprocess.Popen(['python3',str(root/'source/shard_locality_prepare.py'),'--root',str(root)],
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ticks=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        write(root/'controller.json',{'pid':child.pid,'start_ticks':ticks,'time':time.time(),'root':str(root)})
        os.kill(continuation['pid'],signal.SIGCONT);cont_stopped=False
        status('complete_resumed_canonical_builder',controller_pid=child.pid,retained_prefix_rows=prefix_rows)
    except BaseException as error:
        if halted and not handover:
            try:os.kill(native,signal.SIGCONT)
            except ProcessLookupError:pass
        if cont_stopped:
            try:os.kill(continuation['pid'],signal.SIGCONT)
            except ProcessLookupError:pass
        status('failed',error=str(error),original_builder_resumed=halted and not handover)
        raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGINT,interrupt)
    signal.signal(signal.SIGTERM,interrupt)
    main(parser.parse_args().root.resolve())
