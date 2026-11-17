"""Read-only occupancy checks and nonblocking SSH-held benchmark leases."""
import concurrent.futures
import contextlib
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import time

if __package__:
    from . import orion_build_resources as resources
else:
    import orion_build_resources as resources

DEFAULT_HOSTS = ('10.1.1.1', '10.1.1.2', '10.1.1.3', '10.1.1.4', '10.1.1.5',
                 '10.1.2.3', '10.1.2.4', '10.1.2.5')
HOSTS = tuple(json.loads(os.environ.get('ORION_BUILD_HOSTS', json.dumps(DEFAULT_HOSTS))))
CONTROLLER_IP = os.environ.get('ORION_BUILD_CONTROLLER_IP', HOSTS[0])
LOCK = '/tmp/orion-eight-host-benchmark.lock'


def ssh_args(host):
    # A source-bound management socket avoids the controller's Meta route.
    return ['ssh', '-b', CONTROLLER_IP, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
            '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3', host]


def remote_python(host, source, timeout=60):
    result = subprocess.run(ssh_args(host) + ['python3 -'], input=source,
                            text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{host}: {result.stderr[-2000:]}')
    return json.loads(result.stdout)


PROBE = r'''
import pathlib,os,time,json,fcntl,subprocess
rows=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  if stat[0] in ('Z','T'):continue
  args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  if not args:continue
  comm=(p/'comm').read_text().strip()
  script=''
  if comm.startswith(('python','pypy','bash','sh')):
   for arg in args[1:]:
    if arg=='-c':break
    if not arg.startswith('-'):script=pathlib.Path(arg).name;break
  foreign=(comm.startswith(('qdrant','orion')) or comm in ('cargo','rustc','cc1','cc1plus','ld.lld','fio','stress','stress-ng','iperf3')
           or any(s in script for s in ('benchmark','experiment','ann_five_load','ann_100m_ingest','prepare_layout')))
  if foreign:
   affinity=next(line.split(':',1)[1].strip() for line in (p/'status').read_text().splitlines() if line.startswith('Cpus_allowed_list:'))
   rows.append({'pid':int(p.name),'start_ticks':int(stat[19]),'state':stat[0],'comm':comm,'argv':args[:3],'affinity':affinity})
 except (OSError,ValueError):pass
busy=False
# Observe locks without briefly acquiring one during the other task's phase
# transition. The actual admission later uses nonblocking flock on all hosts.
lock_path=pathlib.Path('/tmp/orion-eight-host-benchmark.lock')
if lock_path.exists():
 stat=lock_path.stat();key=(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino)
 for line in pathlib.Path('/proc/locks').read_text().splitlines():
  for field in line.split():
   parts=field.split(':')
   if len(parts)==3:
    try:match=(int(parts[0],16),int(parts[1],16),int(parts[2]))==key
    except ValueError:continue
    if match:busy=True
memory={}
for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
 key,rest=line.split(':',1)
 if key in ('MemAvailable','SwapFree'):memory[key]=int(rest.split()[0])*1024
print(json.dumps({'time':time.time(),'processes':rows,'lock_busy':busy,'memory':memory,
 'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 'resources':build_resources['detect']()}))
'''

# Ship the same standard-library-only probe to every host, including the
# controller's SSH context. No installation or fixed CPU numbering is required.
PROBE = ('build_resources={"__name__":"orion_build_resources"}\n'
         f'exec({Path(resources.__file__).read_text()!r}, build_resources)\n' + PROBE)


def probe_all():
    def one(host):
        try:
            return host, remote_python(host, PROBE)
        except Exception as error:
            return host, {'error': str(error), 'lock_busy': True}
    with concurrent.futures.ThreadPoolExecutor(len(HOSTS)) as pool:
        return dict(pool.map(one, HOSTS))


def available(snapshot):
    if len(snapshot) == 8 and all(not row.get('error') for row in snapshot.values()):
        if not all(row.get('resources', {}).get('worker_count', 0) > 0 for row in snapshot.values()):
            raise ValueError('no available physical build cores on a host')
        if len({row['boot_id'] for row in snapshot.values()}) != 8:
            raise ValueError('host addresses do not identify eight distinct machines')
        local_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if snapshot[HOSTS[0]]['boot_id'] != local_boot:
            raise ValueError('the first configured host must be this controller')
    return len(snapshot) == 8 and all(not row.get('error') and not row.get('lock_busy')
                                     and not row.get('processes') for row in snapshot.values())


class Lease:
    def __init__(self):
        self.processes = []

    def __enter__(self):
        code = (f"import fcntl,sys,json,os;f=open({LOCK!r},'a+');"
                "fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);"
                "print(json.dumps({'leased':True,'pid':os.getpid()}),flush=True);sys.stdin.read()")
        try:
            for host in sorted(HOSTS):
                process = subprocess.Popen(ssh_args(host) + ['python3 -c ' + shlex.quote(code)],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True)
                self.processes.append((host, process))
                if not select.select([process.stdout], [], [], 12)[0]:
                    raise RuntimeError(f'Lease timeout: {host}')
                row = process.stdout.readline()
                if not row or not json.loads(row).get('leased'):
                    raise RuntimeError(f'Lease held by another task: {host}')
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def check(self):
        if any(p.poll() is not None for _, p in self.processes):
            raise RuntimeError('A host lease was lost; stop only owned preparation processes')

    def __exit__(self, *_):
        for _, process in reversed(self.processes):
            with contextlib.suppress(OSError, ValueError):
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()
        self.processes.clear()
