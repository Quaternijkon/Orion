import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import shard_locality as core
from tools import shard_locality_gt as gt


class GTCoverageTests(unittest.TestCase):
    def test_ann_misses_do_not_change_real_and_oracle_uses_exact_same_subset(self):
        truth = np.array([[0, 1], [2, 3]])
        members = {0:[0], 1:[1], 2:[0], 3:[0]}
        oracle, masks = core.oracle_matrix(truth, members, 3)
        # All-query Oracle[1] is .75, but the n=1 query's Oracle is only .5.
        routes = {'fixed': [[{'shard_id':0}], [{'shard_id':0}, {'shard_id':1}]]}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            curves, rows, _ = core.summarize_method({'real_profile':'fixed'},
                {'shard_count':3, 'point_count':4}, folder, 'orion', [{'name':'fixed'}],
                routes, {'fixed':[[],[]]}, truth, [3,1,0], oracle, masks)
            values = {(r['metric'],r['searched_shards']):r for r in curves}
            self.assertEqual(values['Real',1]['recall'], .5)
            self.assertEqual(values['Oracle',1]['recall'], .5)
            self.assertEqual(values['Oracle',1]['query_count'], 1)
            self.assertEqual(rows[0]['ann_recall'], 0)
            self.assertEqual(values['Real',2]['recall'], 1)
            for metric in ['Oracle','Real']:
                self.assertIsNone(values[metric,3]['recall'])
                self.assertEqual(values[metric,3]['query_count'], 0)

    def test_cross_shard_gt_replicas_count_only_once(self):
        truth = np.array([[1,2,3]])
        members = {1:[0,1], 2:[0], 3:[1]}
        oracle, _ = core.oracle_matrix(truth,members,2)
        routes = [[{'shard_id':0},{'shard_id':1}]]
        rows = gt.paired_groups('orion','fixed',routes,truth,members,oracle,[[],[0]])
        self.assertEqual(rows[1]['real_recall'],1)
        self.assertEqual(rows[1]['gap_percentage_points'],0)
        with self.assertRaisesRegex(ValueError,'visited shard count'):
            gt.paired_groups('orion','fixed',routes,truth,members,oracle,[[0],[]])

    def test_reanalysis_uses_routes_and_memberships_without_any_ann_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            truth=np.array([[0,1],[2,3]],dtype=np.int64);np.save(source/'truth.npy',truth)
            core.write_json(source/'config.json',{'top_k':2,'real_profile':'fixed'})
            core.write_json(source/'provenance.json',{'truth_sha256':core.sha256(source/'truth.npy')})
            core.write_json(source/'COMPLETED.json',{'complete':True})
            members={0:[0],1:[1],2:[0],3:[0]}
            oracle,_=core.oracle_matrix(truth,members,3)
            for method in ['orion','kmeans']:
                d=source/method;d.mkdir()
                profiles=[{'name':'fixed'}] if method=='orion' else [{'name':f'nprobe-{n}','nprobe':n} for n in range(1,4)]
                core.write_json(d/'summary.json',{'shard_sizes':[3,1,0],'physical_copies':4,'replication_factor':1,
                    'profiles':[{'profile':p} for p in profiles]})
                core.write_json(d/'gt-memberships.json',members);np.save(d/'oracle.npy',oracle)
                for p in profiles:
                    rows=[]
                    for q in range(2):
                        selected=([0] if q==0 else [0,1]) if method=='orion' else list(range(p['nprobe']))
                        rows.append({'query_index':q,'targets':[{'shard_id':s,'ef':32,
                            'entry_points':[0] if method=='orion' else None} for s in selected]})
                    (d/f'route-{p["name"]}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):curves=gt.reanalyze(source,root/'out')
            self.assertTrue(core.read_json(root/'out/audit.json')['paired_real_not_above_oracle'])
            self.assertEqual(core.read_json(root/'out/audit.json')['lower_searches_run'],0)
            value=next(r for r in curves if r['method']=='orion' and r['metric']=='Oracle' and r['searched_shards']==1)
            self.assertEqual(value['recall'],.5)
            self.assertTrue((root/'out/common_orion_query_groups.csv').exists())
            with self.assertRaises(FileExistsError):gt.reanalyze(source,root/'out')


if __name__=='__main__':unittest.main()
