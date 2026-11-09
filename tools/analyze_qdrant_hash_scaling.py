#!/usr/bin/env python3
"""Audit and summarize completed native Qdrant SIFT1M scaling measurements."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyze(source, output):
    rows = []
    for n in range(1, 9):
        folder = source / f"n{n}"
        runs = json.loads((folder / "results.json").read_text())
        assert len(runs) == 9, (n, "missing runs")
        assert {(r['concurrency'], r['repeat']) for r in runs} == {
            (c, r) for c in (1, 32, 128) for r in range(3)}
        collection = json.loads((folder / "collection.json").read_text())
        assert collection['points_count'] == 1000000
        assert collection['indexed_vectors_count'] >= 990000
        assert collection['status'] == 'green'
        placement = json.loads((folder / "placement.json").read_text())
        assert len(placement) == n
        shards = []
        points = 0
        for host in placement:
            state = host['result']
            assert not state['shard_transfers']
            assert len(state['local_shards']) == 1
            shard = state['local_shards'][0]
            assert shard['state'] == 'Active'
            shards.append(shard['shard_id'])
            points += shard['points_count']
        assert len(set(shards)) == n and points == 1000000
        recall = np.load(folder / "recall_per_query.npy")
        assert recall.shape == (10000,)
        for run in runs:
            c, rep = run['concurrency'], run['repeat']
            workers = min(c, 4)
            samples, summaries = [], []
            for worker in range(workers):
                prefix = folder / f"c{c}-r{rep}-w{worker}"
                with np.load(str(prefix) + '.npz') as data:
                    samples.append(data['latency_ms'].copy())
                summaries.append(json.loads(Path(str(prefix) + '.json').read_text()))
            latencies = np.concatenate(samples)
            assert all(not s['errors'] for s in summaries)
            assert len(latencies) == run['requests']
            wall = max(s['end_monotonic'] for s in summaries) - min(s['start_monotonic'] for s in summaries)
            assert np.isclose(len(latencies) / wall, run['qps'])
            assert np.isclose(latencies.mean(), run['latency_mean_ms'])
            for p in (50, 95, 99):
                assert np.isclose(np.percentile(latencies, p), run[f'latency_p{p}_ms'])
        for c in (1, 32, 128):
            group = [r for r in runs if r['concurrency'] == c]
            row = dict(nodes=n, concurrency=c, recall_at_10=float(recall.mean()))
            for key in ('qps', 'latency_mean_ms', 'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms'):
                values = [r[key] for r in group]
                row[key] = median(values)
                row[key + '_min'] = min(values)
                row[key + '_max'] = max(values)
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c in (1, 32, 128):
        group = [r for r in rows if r['concurrency'] == c]
        for ax, key, label in zip(axes, ('qps', 'latency_mean_ms', 'latency_p99_ms'),
                                  ('QPS', 'Mean latency (ms)', 'p99 latency (ms)')):
            ax.errorbar([r['nodes'] for r in group], [r[key] for r in group],
                        yerr=[[r[key]-r[key+'_min'] for r in group],
                              [r[key+'_max']-r[key] for r in group]],
                        marker='o', capsize=3, label=f'C={c}')
            ax.set(xlabel='Machines (8 cores each)', ylabel=label, xticks=range(1, 9))
            ax.grid(alpha=.3)
    for ax in axes:
        ax.legend()
    fig.suptitle('SIFT1M / native hash sharding / median and range of 3 repeats')
    fig.tight_layout()
    fig.savefig(output / 'scaling.png', dpi=180)
    fig.savefig(output / 'scaling.pdf')
    (output / 'audit.json').write_text(json.dumps({
        'source': str(source.resolve()), 'topologies': 8, 'runs': 72,
        'raw_request_metrics_verified': True,
        'placement_and_point_counts_verified': True,
    }, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    analyze(args.source, args.output)
