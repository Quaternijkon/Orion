#!/usr/bin/env python3
"""Re-score frozen routes using GT coverage and Oracle on the same query subsets.

No index building, network requests, lower HNSW searches or new route settings.
Output is a new directory; source experiment evidence is never overwritten.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import shard_locality as core
else:
    import shard_locality as core


def paired_groups(method, profile, routes, truth, memberships, oracle, query_groups):
    """Independent set-based calculation on explicitly identified query groups."""
    rows = []
    for n, indices in enumerate(query_groups, start=1):
        real, upper = [], []
        for q in indices:
            selected = {t['shard_id'] for t in routes[q]}
            if len(selected) != n:
                raise ValueError('query group does not match actual visited shard count')
            gt = set(map(int, truth[q]))
            real.append(sum(bool(selected.intersection(memberships[p])) for p in gt) / len(gt))
            upper.append(float(oracle[q, n]))
        mean_real = float(np.mean(real)) if real else None
        mean_oracle = float(np.mean(upper)) if upper else None
        if real and mean_real > mean_oracle + 1e-12:
            raise ValueError('paired Real exceeds Oracle')
        rows.append({'method': method, 'profile': profile, 'searched_shards': n,
                     'query_count': len(indices), 'real_recall': mean_real, 'oracle_recall': mean_oracle,
                     'gap_percentage_points': (mean_oracle-mean_real)*100 if real else None})
    return rows


def reanalyze(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    if not core.read_json(source/'COMPLETED.json').get('complete'):
        raise ValueError('source experiment must be complete')
    config = core.read_json(source/'config.json')
    provenance = core.read_json(source/'provenance.json')
    if core.sha256(source/'truth.npy') != provenance['truth_sha256']:
        raise ValueError('source truth checksum mismatch')
    truth = np.load(source/'truth.npy', allow_pickle=False)
    if (truth.ndim != 2 or truth.shape[1] != config.get('top_k', 10) or not len(truth)
            or truth.dtype.kind not in 'iu' or (truth < 0).any()
            or any(len(set(row)) != len(row) for row in truth)):
        raise ValueError('invalid distinct GT IDs')
    output.mkdir(parents=True, exist_ok=False)
    all_curves, all_details, all_groups, all_population = [], [], [], []
    inputs = {'truth.npy': core.sha256(source/'truth.npy'), 'config.json': core.sha256(source/'config.json'),
              'provenance.json': core.sha256(source/'provenance.json')}
    if 'queries_sha256' in provenance and core.sha256(source/'queries.f32le') != provenance['queries_sha256']:
        raise ValueError('source query checksum mismatch')
    cached = {}
    for method in ('orion', 'kmeans'):
        folder = output/method; folder.mkdir()
        summary = core.read_json(source/method/'summary.json')
        inputs[f'{method}/summary.json'] = core.sha256(source/method/'summary.json')
        counts = summary['shard_sizes']; shards = len(counts)
        profiles = [p['profile'] for p in summary['profiles']]
        members_path = source/method/'gt-memberships.json'
        memberships = {int(k): v for k,v in core.read_json(members_path).items()}
        for point in np.unique(truth):
            owners = memberships.get(int(point))
            if (not owners or len(set(owners)) != len(owners) or
                    any(type(s) is not int or not 0 <= s < shards for s in owners)):
                raise ValueError(f'invalid GT membership: {point}')
            if method == 'kmeans' and len(owners) != 1:
                raise ValueError('KMeans GT memberships must be disjoint')
        oracle, masks = core.oracle_matrix(truth, memberships, shards)
        if not np.array_equal(oracle, np.load(source/method/'oracle.npy', allow_pickle=False)):
            raise ValueError('fresh exact Oracle does not match the archived per-query Oracle')
        inputs[f'{method}/gt-memberships.json'] = core.sha256(members_path)
        inputs[f'{method}/oracle.npy'] = core.sha256(source/method/'oracle.npy')
        routes = {}
        for profile in profiles:
            path = source/method/f'route-{profile["name"]}.jsonl'
            routes[profile['name']] = core.load_routes(path, len(truth), shards, method, profile.get('nprobe'))
            inputs[str(path.relative_to(source))] = core.sha256(path)
        logical_count = summary['physical_copies'] / summary['replication_factor']
        manifest = {'shard_count': shards, 'point_count': round(logical_count)}
        curves, details, groups = core.summarize_method(config, manifest, folder, method, profiles,
                                                       routes, None, truth, counts, oracle, masks)
        independent = []
        for profile in profiles:
            name = profile['name']
            indices = [[q for q,r in enumerate(routes[name]) if len(r) == n] for n in range(1, shards+1)]
            independent.extend(paired_groups(method, name, routes[name], truth, memberships, oracle, indices))
        for actual, check in zip(groups, independent):
            if actual['query_count'] != check['query_count']:
                raise ValueError('independent group count mismatch')
            for key, other in [('recall', 'real_recall'), ('oracle_recall', 'oracle_recall')]:
                if actual[key] is None:
                    if check[other] is not None: raise ValueError('empty group mismatch')
                elif abs(actual[key]-check[other]) > 1e-12:
                    raise ValueError('independent set/bitmask calculation mismatch')
        all_curves.extend(curves); all_details.extend(details); all_groups.extend(groups)
        for n in range(1, shards+1):
            all_population.append({'method': method, 'searched_shards': n, 'query_count': len(truth),
                                   'oracle_recall': float(oracle[:,n].mean())})
        cached[method] = dict(routes=routes, oracle=oracle, memberships=memberships, shards=shards)

    # Optional cross-method comparison on the SAME Orion query groups, using
    # the already recorded KMeans nprobe=n routes. No searches are rerun.
    selected = config.get('real_profile', next(iter(cached['orion']['routes'])))
    orion_routes = cached['orion']['routes'][selected]
    common_groups = [[q for q,r in enumerate(orion_routes) if len(r) == n]
                     for n in range(1, cached['orion']['shards']+1)]
    common = paired_groups('orion', selected, orion_routes, truth, cached['orion']['memberships'],
                           cached['orion']['oracle'], common_groups)
    for n, indices in enumerate(common_groups, start=1):
        # Other groups are deliberately empty for this nprobe setting.
        per_n = [[] for _ in common_groups]; per_n[n-1] = indices
        name = f'nprobe-{n}'
        common.append(paired_groups('kmeans', name, cached['kmeans']['routes'][name], truth,
                      cached['kmeans']['memberships'], cached['kmeans']['oracle'], per_n)[n-1])
    for name, rows in [('curves',all_curves), ('per_query',all_details), ('paired_groups',all_groups),
                       ('oracle_all_queries',all_population), ('common_orion_query_groups',common)]:
        core.write_csv(output/f'{name}.csv', rows)
    lookup = {(r['method'], r['profile'], r['searched_shards'], r['metric']):r for r in all_curves}
    gaps = []
    for row in all_curves:
        if row['metric'] != 'Real': continue
        upper = lookup[row['method'], row['profile'], row['searched_shards'], 'Oracle']
        gaps.append({'method':row['method'], 'profile':row['profile'], 'searched_shards':row['searched_shards'],
                     'query_count':row['query_count'], 'real_recall':row['recall'], 'oracle_recall':upper['recall'],
                     'gap_percentage_points':100*(upper['recall']-row['recall']) if row['recall'] is not None else None})
    core.write_csv(output/'routing_gap.csv', gaps)
    core.plot(all_curves, output, truth.shape[1], cached['orion']['shards'])
    core.write_json(output/'config.json', dict(config, collect_ann_diagnostic=False))
    definition = {'metric_schema': core.METRIC_SCHEMA,
        'real': 'Fraction of distinct GT IDs present in the union of actually selected shards; assume exact local search.',
        'oracle': 'Maximum distinct GT coverage in at most n shards, averaged over exactly the same query IDs as Real[n].',
        'empty_groups': 'Both Real and Oracle are N/A; zero query count.',
        'main_population': 'Orion: actual fanout groups. KMeans: all queries for nprobe=n.',
        'cross_method_population': 'common_orion_query_groups.csv uses Orion fanout groups for BOTH methods.',
        'all_query_oracle': 'oracle_all_queries.csv is supplementary, not mixed into the paired primary curves.',
        'routing_constraints_preserved': True, 'lower_ann_results_used': False,
        'source': str(source), 'source_hashes': inputs, 'script_sha256': core.sha256(Path(__file__))}
    core.write_json(output/'metric_definition.json', definition)
    core.write_json(output/'audit.json', {'passed': True, 'independent_set_and_bitmask_results_match': True,
        'recomputed_oracle_matches_archived_matrix': True, 'paired_real_not_above_oracle': True,
        'lower_searches_run': 0, 'new_routes_computed': 0, 'per_query_rows': len(all_details)})
    core.write_json(output/'COMPLETED.json', {'complete': True, 'metric_schema': core.METRIC_SCHEMA,
                                           'reanalyzed_existing_routes': True, 'qps_measured': False})
    return all_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    for row in reanalyze(args.source, args.output):
        print(json.dumps(row, allow_nan=False))


if __name__ == '__main__': main()
