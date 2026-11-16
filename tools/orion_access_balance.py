"""Fixed access-probability refinement of a native capacity partition.

Each selected shard costs one unit per proxy query, regardless of how many upper
hits it owns. No query truth, measured CPU cost or learned model is accepted.
Refinement cannot increase mean fanout or the navigation-cooccurrence edge cut.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

MAX_PASSES = 4
MAX_MOVES = 4096
MAX_EVALUATIONS = 200_000


def select_shards(labels, distances, shards, whitebox):
    """Vectorized replay of native whitebox selection, with one request/shard."""
    labels=np.asarray(labels)
    distances=np.asarray(distances,dtype=np.float32)
    if labels.ndim!=2 or labels.shape!=distances.shape or labels.shape[1]==0:
        raise ValueError('invalid proxy hit shape')
    if (labels<0).any() or (labels>=shards).any() or not np.isfinite(distances).all():
        raise ValueError('invalid proxy owner/distance')
    prefix=int(whitebox['prefix_k']);support=int(whitebox['max_prefix_entries']);gap=float(whitebox['max_distance_gap'])
    if not 0<prefix<=labels.shape[1] or not 0<=support<prefix or not np.isfinite(gap) or gap<0:
        raise ValueError('invalid fixed whitebox parameters')
    if (np.diff(distances,axis=1)<0).any():raise ValueError('upper hits must be ordered')
    row=np.arange(len(labels))[:,None]
    counts=np.zeros((len(labels),shards),dtype=np.int32)
    prefix_counts=np.zeros_like(counts)
    best=np.full((len(labels),shards),np.inf,dtype=np.float32)
    np.add.at(counts,(row,labels),1)
    np.add.at(prefix_counts,(row,labels[:,:prefix]),1)
    np.minimum.at(best,(row,labels),distances)
    # Native rule subtracts in f64 and rounds once to f32 before comparison.
    gaps=(best.astype(np.float64)-distances[:,0,None].astype(np.float64)).astype(np.float32)
    result=(counts>0)&((prefix_counts>support)|(gaps.astype(np.float64)<=gap))
    if not result.any(axis=1).all():raise ValueError('whitebox lost the global-best owner')
    return result


def access_statistics(selected):
    counts=np.asarray(selected,dtype=np.int64).sum(axis=0)
    probability=counts/len(selected)
    return {'query_count':len(selected),'requests_per_shard':counts.tolist(),
            'access_probability':probability.tolist(),'mean_fanout':float(probability.sum()),
            'maximum_access_probability':float(probability.max()),
            'max_over_mean':float(counts.max()/counts.mean())}


def objective(counts):
    return int(counts.max()),sum(int(v)**2 for v in counts),int(counts.sum())


def conservative_affected_budget(counts):
    """Capacity remains safe even if every affected point changes all memberships.

    For B affected rows: new max <= old max+B and new total >= old total-(S-1)B.
    Enforce the same native bound 5*S*max <= 6*total, in integer arithmetic.
    """
    s=len(counts)
    return max(0,(6*sum(map(int,counts))-5*s*int(max(counts)))//(5*s+6*(s-1)))


def refine(owners, data_hits, counts, probe_hits, distances, whitebox,
           xadj, adjacency, edge_weights, *, max_physical_copies=None, progress=None):
    owners=np.asarray(owners,dtype=np.int64).copy()
    probe_hits=np.asarray(probe_hits,dtype=np.int64)
    if probe_hits.ndim!=2 or len(probe_hits)==0 or (probe_hits<0).any() or (probe_hits>=len(owners)).any():
        raise ValueError('invalid proxy node IDs')
    if (np.diff(np.sort(probe_hits,axis=1),axis=1)==0).any():raise ValueError('duplicate proxy upper hits')
    shards=len(counts)
    selected=select_shards(owners[probe_hits],distances,shards,whitebox)
    initial=access_statistics(selected);load=selected.sum(axis=0,dtype=np.int64)
    frequency=np.bincount(probe_hits.ravel(),minlength=len(owners))
    order=np.argsort(probe_hits.ravel(),kind='stable')
    query_rows=(order//probe_hits.shape[1]).astype(np.int32)
    offsets=np.r_[0,np.cumsum(frequency)]
    incidence=np.bincount(np.asarray(data_hits).ravel(),minlength=len(owners))
    remaining=conservative_affected_budget(counts)
    if max_physical_copies is not None and shards>1:
        remaining=min(remaining,max(0,(int(max_physical_copies)-sum(map(int,counts)))//(shards-1)))
    initial_budget=remaining
    candidates=np.flatnonzero(frequency)
    candidates=candidates[np.argsort(-frequency[candidates],kind='stable')]
    moves=[];evaluations=0;cut_change=0
    for pass_number in range(MAX_PASSES):
        if load.max()==load.min() or remaining==0:break
        changes_before=len(moves)
        for node in candidates:
            source=int(owners[node])
            if load[source]!=load.max():continue
            charge=int(incidence[node])+1  # owner-copy completion can affect this upper row too
            if charge>remaining:continue
            start,end=int(xadj[node]),int(xadj[node+1])
            neighbors=np.asarray(adjacency[start:end],dtype=np.int64)
            weights=np.asarray(edge_weights[start:end],dtype=np.int64)
            affinity=np.bincount(owners[neighbors],weights=weights,minlength=shards)
            destinations=[s for s in range(shards) if load[s]<load[source] and affinity[s]>=affinity[source] and affinity[s]>0]
            if not destinations:continue
            rows=query_rows[offsets[node]:offsets[node+1]]
            old_selected=selected[rows]
            mapped=owners[probe_hits[rows]]
            best=None
            for destination in sorted(destinations,key=lambda s:(int(load[s]),-affinity[s],s)):
                evaluations+=1
                new_labels=mapped.copy();new_labels[probe_hits[rows]==node]=destination
                new_selected=select_shards(new_labels,distances[rows],shards,whitebox)
                delta=new_selected.sum(axis=0,dtype=np.int64)-old_selected.sum(axis=0,dtype=np.int64)
                proposed=load+delta
                if proposed.sum()>load.sum() or objective(proposed)>=objective(load):continue
                candidate=(objective(proposed),destination,proposed,new_selected)
                if best is None or candidate[:2]<best[:2]:best=candidate
            if best is not None:
                _,destination,load,new_selected=best
                owners[node]=destination;selected[rows]=new_selected;remaining-=charge
                change=int(affinity[source]-affinity[destination]);assert change<=0;cut_change+=change
                moves.append({'node':int(node),'from':source,'to':destination,'affected_bound':charge,
                              'cut_change':change,'peak_requests':int(load.max()),'total_requests':int(load.sum())})
                if progress and len(moves)%128==0:progress({'pass':pass_number,'moves':len(moves),'evaluations':evaluations,'statistics':access_statistics(selected)})
            if len(moves)>=MAX_MOVES or evaluations>=MAX_EVALUATIONS:break
        if len(moves)==changes_before or len(moves)>=MAX_MOVES or evaluations>=MAX_EVALUATIONS:break
    replay=select_shards(owners[probe_hits],distances,shards,whitebox)
    if not np.array_equal(selected,replay):raise RuntimeError('incremental probability replay mismatch')
    final=access_statistics(replay)
    assert final['mean_fanout']<=initial['mean_fanout']+1e-12
    assert final['maximum_access_probability']<=initial['maximum_access_probability']+1e-12
    return owners,{'algorithm':'orion-access-probability-capacity-v1','equal_cost_per_shard_request':True,
                   'uses_external_queries':False,'uses_ground_truth':False,'uses_measured_search_cost':False,
                   'before':initial,'after':final,'accepted_moves':len(moves),'evaluations':evaluations,
                   'navigation_cut_change':cut_change,'affected_rows_bound':initial_budget-remaining,
                   'capacity_safe_affected_budget':initial_budget,'moves':moves}


def load_probes(folder:Path,meta,upper_ids,checked_path):
    manifest=json.loads((folder/'access-probes.json').read_text())
    if manifest['format_version']!=1 or manifest['uses_external_queries'] or manifest['uses_ground_truth']:
        raise ValueError('not a construction-only access probe bundle')
    for key in ['upper_graph_sha256','upper_identity_sha256']:
        if manifest[key]!=meta[key]:raise ValueError('proxy graph identity mismatch')
    if manifest['source_vectors_sha256']!=meta['vectors_sha256'] or manifest['source_point_count']!=meta['point_count']:
        raise ValueError('proxy data identity mismatch')
    if not manifest['excludes_upper_sample'] or not manifest['equal_cost_per_shard_request']:
        raise ValueError('unsupported proxy semantics')
    if (manifest['top_k'],manifest['search_ef'])!=(meta['options']['upper_k'],meta['options']['upper_ef_search']):
        raise ValueError('proxy runtime budget mismatch')
    q,k=manifest['query_count'],manifest['top_k']
    ids_path=checked_path(folder,manifest['query_ids_file'],manifest['query_ids_sha256'])
    hits_path=checked_path(folder,manifest['hit_ids_file'],manifest['hit_ids_sha256'])
    distance_path=checked_path(folder,manifest['distances_file'],manifest['distances_sha256'])
    query_ids=np.fromfile(ids_path,dtype='<u8')
    if len(query_ids)!=q or len(np.unique(query_ids))!=q or (query_ids>=meta['point_count']).any() or np.isin(query_ids,upper_ids).any():
        raise ValueError('invalid or self-hit proxy rows')
    if hits_path.stat().st_size!=q*k*8 or distance_path.stat().st_size!=q*k*4:raise ValueError('proxy matrix size mismatch')
    labels=np.memmap(hits_path,dtype='<u8',mode='r',shape=(q,k))
    distances=np.memmap(distance_path,dtype='<f4',mode='r',shape=(q,k))
    order=np.argsort(upper_ids);sorted_ids=upper_ids[order];position=np.searchsorted(sorted_ids,labels)
    if (position>=len(upper_ids)).any() or not np.array_equal(sorted_ids[position],labels):raise ValueError('unknown proxy upper label')
    return order[position],distances,manifest
