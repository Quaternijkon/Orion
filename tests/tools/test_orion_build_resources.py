"""Synthetic hardware and byte-equivalence checks; no servers or datasets."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import orion_build_resources as resources


class ResourceTests(unittest.TestCase):
    def fixture(self, root):
        proc, cpu, cgroup = root/'proc', root/'cpu', root/'cgroup'
        (proc/'self').mkdir(parents=True); cgroup.mkdir()
        (proc/'self/cgroup').write_text('0::/experiment/child\n')
        (proc/'self/mountinfo').write_text('1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n')
        (proc/'meminfo').write_text('MemAvailable: 104857600 kB\n')
        # 192 logical CPUs / 96 physical cores across two NUMA nodes.
        for logical in range(192):
            directory=cpu/f'cpu{logical}'; topology=directory/'topology'; topology.mkdir(parents=True)
            (topology/'physical_package_id').write_text(str((logical%96)//48))
            (topology/'core_id').write_text(str(logical%48))
            (directory/f'node{(logical%96)//48}').mkdir()
        return proc,cpu,cgroup

    def test_uses_all_physical_cores_across_numa_without_counting_smt_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc,cpu,cgroup=self.fixture(Path(tmp))
            detected=resources.detect(proc,cpu,cgroup,range(192))
            self.assertEqual(detected['worker_count'],96)
            self.assertEqual(detected['selected_cpus'],list(range(96)))
            self.assertEqual({node:len(cpus) for node,cpus in detected['numa_nodes'].items()},
                             {'node0':48,'node1':48})
            restricted=resources.detect(proc,cpu,cgroup,[7,103,150,151])
            self.assertEqual(restricted['selected_cpus'],[7,150,151])

    def test_nested_cpu_and_memory_quotas_limit_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc,cpu,cgroup=self.fixture(Path(tmp))
            child=cgroup/'experiment/child'; child.mkdir(parents=True)
            (child/'cpu.max').write_text('max 100000')
            (child.parent/'cpu.max').write_text('250000 100000')
            (child/'memory.max').write_text(str(8<<30));(child/'memory.current').write_text(str(1<<30))
            (child.parent/'memory.max').write_text(str(4<<30));(child.parent/'memory.current').write_text(str(2<<30))
            detected=resources.detect(proc,cpu,cgroup,range(192))
            self.assertEqual(detected['worker_count'],3)
            self.assertEqual(detected['memory_available_bytes'],2<<30)
            detected['worker_count']=96
            self.assertEqual(resources.split_workers(detected,128,100000000),5)
            self.assertEqual(resources.split_workers(detected,128,2),2)
            detected['memory_available_bytes']=0
            with self.assertRaisesRegex(ValueError,'memory'):
                resources.split_workers(detected,128,100)

    def test_namespaced_mount_and_v1_quotas(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc,cpu,cgroup=self.fixture(Path(tmp))
            (proc/'self/mountinfo').write_text('1 0 0:1 /experiment /sys/fs/cgroup rw - cgroup2 cgroup rw\n')
            (cgroup/'child').mkdir();(cgroup/'cpu.max').write_text('400000 100000')
            self.assertEqual(resources.detect(proc,cpu,cgroup,range(192))['worker_count'],4)
            (proc/'self/cgroup').write_text('1:cpu,cpuacct:/job\n2:memory:/job\n')
            (proc/'self/mountinfo').write_text(
                '1 0 0:1 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n'
                '2 0 0:2 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n')
            (cgroup/'cpu/job').mkdir(parents=True);(cgroup/'memory/job').mkdir(parents=True)
            (cgroup/'cpu/cpu.cfs_quota_us').write_text('200000')
            (cgroup/'cpu/cpu.cfs_period_us').write_text('100000')
            (cgroup/'memory/job/memory.limit_in_bytes').write_text(str(2<<30))
            (cgroup/'memory/job/memory.usage_in_bytes').write_text(str(1<<30))
            detected=resources.detect(proc,cpu,cgroup,range(192))
            self.assertEqual(detected['worker_count'],2)
            self.assertEqual(detected['memory_available_bytes'],1<<30)

    def test_heterogeneous_hosts_get_distinct_budgets_and_contiguous_ranges(self):
        large={'worker_count':96,'selected_cpus':list(range(96)),'memory_available_bytes':100<<30}
        small={'worker_count':24,'selected_cpus':list(range(24)),'memory_available_bytes':100<<30}
        plan=resources.resolve({'a':{'resources':large},'b':{'resources':small}},128,100000000,'a')
        self.assertEqual(plan['lower_threads_per_host'],{'a':96,'b':24})
        self.assertEqual(plan['attachment_threads_per_host'],{'a':96,'b':24})
        self.assertEqual(plan['split_processes'],96)
        self.assertFalse(plan['search_resources_changed'])
        self.assertEqual(resources.weighted_ranges(10,110,[96,24]),[(10,90),(90,110)])
        ranges=resources.weighted_ranges(17,100000000,[96,96,24,48,96,24,96,96])
        self.assertEqual(ranges[0][0],17);self.assertEqual(ranges[-1][1],100000000)
        self.assertTrue(all(a[1]==b[0] for a,b in zip(ranges,ranges[1:])))

    def test_parallel_hashing_matches_independent_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); storage=root/'storage'; storage.mkdir()
            data={'empty':b'', 'large':b'abcdef'*(1<<19), 'small':b'xyz'}
            archive=root/'index.tar'
            with tarfile.open(archive,'w:') as tar:
                for name,content in data.items():
                    (storage/name).write_bytes(content)
                    info=tarfile.TarInfo('storage/'+name);info.size=len(content)
                    tar.addfile(info,io.BytesIO(content))
            expected={'storage/'+name:{'size':len(content),'sha256':hashlib.sha256(content).hexdigest()}
                      for name,content in data.items()}
            self.assertEqual(resources.hash_tar(archive,96),expected)
            self.assertEqual(resources.hash_tree(root,96),expected)


if __name__=='__main__':unittest.main()
