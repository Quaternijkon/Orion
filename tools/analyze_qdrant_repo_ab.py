#!/usr/bin/env python3
"""Summarize same-index, paired hash measurements; refuse incomplete topologies."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def analyze(source, output, nodes=range(1,9)):
    rows, quality = [], []
    for n in nodes:
        folder=source/f'n{n}'
        done=json.loads((folder/'complete.json').read_text())
        assert done['graph_bytes_unchanged']
        blocks=[]
        reference=None
        for block,variant in enumerate(done['sequence']):
            d=folder/f'block{block}-{variant}'
            runs=json.loads((d/'results.json').read_text())
            assert len(runs)==3 and {r['concurrency'] for r in runs}=={1,32,128}
            for run in runs:
                assert run['variant']==variant and not run['errors']
                c=run['concurrency']; samples=[]; workers=[]
                for w in range(min(4,c)):
                    prefix=d/f'c{c}-r{block}-w{w}'
                    with np.load(str(prefix)+'.npz') as f:samples.append(f['latency_ms'].copy())
                    workers.append(json.loads(Path(str(prefix)+'.json').read_text()))
                lat=np.concatenate(samples)
                assert all(not w['errors'] for w in workers)
                wall=max(w['end_monotonic'] for w in workers)-min(w['start_monotonic'] for w in workers)
                assert len(lat)==run['requests'] and np.isclose(len(lat)/wall,run['qps'])
                assert np.isclose(lat.mean(),run['latency_mean_ms'])
                for p in (50,95,99):assert np.isclose(np.percentile(lat,p),run[f'latency_p{p}_ms'])
            with np.load(d/'query_results.npz') as f:
                ids,scores=f['ids'].copy(),f['scores'].copy()
            assert ids.shape==(10000,10)
            if reference is None:reference=(ids,scores)
            quality.append(dict(nodes=n,block=block,variant=variant,
                ordered_id_mismatch_queries=int(np.any(ids!=reference[0],axis=1).sum()),
                id_set_mismatch_queries=int(np.any(np.sort(ids,axis=1)!=np.sort(reference[0],axis=1),axis=1).sum()),
                score_exact_mismatch_queries=int(np.any(scores!=reference[1],axis=1).sum()),
                score_mismatch_queries=int(np.any(~np.isclose(scores,reference[1],rtol=1e-6,atol=1e-6),axis=1).sum()),
                recall_at_10=json.loads((d/'recall.json').read_text())['recall_at_10']))
            blocks.append({r['concurrency']:r for r in runs})
        for c in (1,32,128):
            row=dict(nodes=n,concurrency=c)
            for variant,indices in [('orion',(0,3,4)),('orion_rust',(1,2,5))]:
                for key in ('qps','latency_mean_ms','latency_p50_ms','latency_p95_ms','latency_p99_ms'):
                    values=[blocks[i][c][key] for i in indices]
                    row[f'{variant}_{key}']=median(values)
                    row[f'{variant}_{key}_min']=min(values)
                    row[f'{variant}_{key}_max']=max(values)
            for key in ('qps','latency_mean_ms','latency_p99_ms'):
                differences=[(blocks[b][c][key]/blocks[a][c][key]-1)*100 for a,b in [(0,1),(3,2),(4,5)]]
                row[key+'_paired_change_pct']=median(differences)
                row[key+'_paired_change_pct_min']=min(differences)
                row[key+'_paired_change_pct_max']=max(differences)
            rows.append(row)
    output.mkdir(parents=True,exist_ok=True)
    for name,data in [('summary',rows),('quality',quality)]:
        with (output/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    fig,axes=plt.subplots(3,3,figsize=(14,11))
    for i,c in enumerate((1,32,128)):
        group=[r for r in rows if r['concurrency']==c]
        for j,(key,label) in enumerate([('qps','QPS'),('latency_mean_ms','Mean latency (ms)'),('latency_p99_ms','p99 latency (ms)')]):
            ax=axes[i,j]
            for variant in ('orion','orion_rust'):
                k=f'{variant}_{key}'
                ax.errorbar([r['nodes'] for r in group],[r[k] for r in group],
                    yerr=[[r[k]-r[k+'_min'] for r in group],[r[k+'_max']-r[k] for r in group]],
                    marker='o',capsize=3,label=variant)
            ax.set(xlabel='Machines',ylabel=label,title=f'Concurrency {c}',xticks=range(1,9));ax.grid(alpha=.3);ax.legend()
    fig.suptitle('SIFT1M same-index hash baseline / median and range of 3 runs')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);fig.savefig(output/'comparison.pdf')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for c in (1,32,128):
        group=[r for r in rows if r['concurrency']==c];key='qps_paired_change_pct'
        ax.errorbar([r['nodes'] for r in group],[r[key] for r in group],
            yerr=[[r[key]-r[key+'_min'] for r in group],[r[key+'_max']-r[key] for r in group]],
            marker='o',capsize=3,label=f'C={c}')
    ax.axhline(0,color='black',linewidth=.8)
    ax.set(xlabel='Machines',ylabel='Paired QPS change: B / A - 1 (%)',xticks=range(1,9),
           title='Orion-rust versus Orion hash mode / median and range of 3 pairs')
    ax.grid(alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(output/'qps_difference.png',dpi=180);fig.savefig(output/'qps_difference.pdf')
    (output/'audit.json').write_text(json.dumps(dict(topologies=len(nodes),measured_runs=18*len(nodes),
        raw_metrics_verified=True,graph_bytes_unchanged=True,queries_per_quality_audit=10000),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--nodes',type=int,nargs='+',default=list(range(1,9)))
    args=parser.parse_args();analyze(args.source,args.output,args.nodes)
