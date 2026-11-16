from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools import orion_access_balance as access
from tools import orion_capacity_partition as capacity


class AccessBalanceTests(unittest.TestCase):
    def test_requests_are_deduplicated_per_query_not_normalized_hit_counts(self):
        selected=access.select_shards(np.array([[0,0,0,1]]),np.array([[0,.1,.2,.3]]),2,
            {'prefix_k':1,'max_prefix_entries':0,'max_distance_gap':1})
        stats=access.access_statistics(selected)
        self.assertEqual(stats['access_probability'],[1.,1.])
        self.assertEqual(stats['mean_fanout'],2.)

    def test_whitebox_uses_prefix_support_and_distance_gap_together(self):
        labels=np.array([[0,1,1,2]])
        distances=np.array([[0,1,2,3]],dtype='float32')
        selected=access.select_shards(labels,distances,3,
            {'prefix_k':3,'max_prefix_entries':1,'max_distance_gap':.5})
        self.assertEqual(selected.tolist(),[[True,True,False]])
        with self.assertRaises(ValueError):access.select_shards(labels,distances[:,::-1],3,
            {'prefix_k':3,'max_prefix_entries':1,'max_distance_gap':.5})

    def test_refinement_improves_actual_request_peak_and_keeps_capacity_and_cut(self):
        owners=np.repeat(np.arange(4),2)
        data=np.array(list(range(8))+[1]*23+[2]*23+[4]*23+[6]*23)[:,None]
        counts,_,_=capacity.evaluate_memberships(data,owners,np.arange(8),4)
        self.assertEqual(counts.tolist(),[25]*4)
        edges=[(0,1,1),(0,2,10),(0,3,10),(1,4,1),(1,6,1)]
        adjacency=[];weights=[];xadj=[0]
        for node in range(8):
            for a,b,w in edges:
                if a==node:adjacency.append(b);weights.append(w)
                if b==node:adjacency.append(a);weights.append(w)
            xadj.append(len(adjacency))
        probe=np.array([[2,0],[3,0],[4,1],[6,1]])
        distances=np.tile([0.,1.],(4,1)).astype('float32')
        result,report=access.refine(owners,data,counts,probe,distances,
            {'prefix_k':1,'max_prefix_entries':0,'max_distance_gap':2},
            np.array(xadj),np.array(adjacency),np.array(weights))
        self.assertEqual(report['before']['maximum_access_probability'],1.)
        self.assertEqual(report['after']['maximum_access_probability'],.5)
        self.assertEqual(report['after']['mean_fanout'],1.5)
        self.assertLessEqual(report['navigation_cut_change'],0)
        actual,_,_=capacity.evaluate_memberships(data,result,np.arange(8),4)
        self.assertTrue(capacity.feasible(actual,len(data),None))
        np.testing.assert_array_equal(owners,np.repeat(np.arange(4),2))

    def test_capacity_bound_cannot_spend_nonexistent_headroom(self):
        self.assertEqual(access.conservative_affected_budget([120,80]),0)
        self.assertGreater(access.conservative_affected_budget([100,100]),0)


if __name__=='__main__':unittest.main()
