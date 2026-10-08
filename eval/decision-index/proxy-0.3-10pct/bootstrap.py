#!/usr/bin/env python3
"""Paired complete-group bootstrap using the kit's unchanged benchmark scorers."""
import argparse
import collections
import json
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
from pathlib import Path
from sample import read_ids
from score import load_proxy


def official_index(rows, results):
    from decision_index.scoring import added, index02
    from decision_index.scoring.index import static_score, chance_baselines
    from decision_index.scoring.report import benchmark_summary
    s = index02.spec('0.3')
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[r['_evaluation']['catalog_id']].append(r)
    values = {}
    for n, rr in grouped.items():
        if n in s['track_scored']:
            transformed = {rid: dict(status=r['status'], answers=(r.get('response') or {}).get('answers', {})) for rid, r in results.items() if r.get('catalog_id') == n}
            track = static_score(n, rr, transformed, chance_baselines())
            values[n] = index02.benchmark_value(n, s, track=track)
        elif str(n) in s['added']:
            values[n] = index02.benchmark_value(n, s, native=added.report(n, rr, results))
        else:
            metrics = {int(k): (v['name'], v['key']) for k, v in s.get('metrics', {}).items()}
            native = benchmark_summary(None, results, '', rows=rr, metrics=metrics)['benchmarks'][0]
            values[n] = index02.benchmark_value(n, s, native=native)
    scores, areas = index02.aggregate(values, s)
    return scores[s['headline']], {a['id']: a['skill'] * 100 for a in areas}


def group_strata(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        e = r['_evaluation']
        groups[(e['catalog_id'], e['group_id'])].append(r)
    strata = collections.defaultdict(list)
    for (n, _), members in sorted(groups.items()):
        signature = tuple(sorted({(str(r.get('metadata', {}).get('domain', r['_evaluation'].get('domain', ''))), str(r['_evaluation'].get('track', ''))) for r in members}))
        strata[(n, signature)].append(members)
    return [strata[key] for key in sorted(strata)]


def resample(strata, rng):
    rows = []
    for groups in strata:
        for draw in range(len(groups)):
            members = groups[rng.randrange(len(groups))]
            for original in members:
                r = dict(original)
                r['_evaluation'] = dict(original['_evaluation'], group_id=f"{original['_evaluation']['group_id']}:bootstrap:{draw}")
                # Keep native metadata clusters and tracks unchanged. Only the
                # sampled catalog/group identity is duplicated for each draw.
                rows.append(r)
    return rows


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


_BOOTSTRAP = None


def initialize_worker(strata, baseline, candidate, seed):
    global _BOOTSTRAP
    _BOOTSTRAP = strata, baseline, candidate, seed


def paired_draw(i):
    strata, baseline, candidate, seed = _BOOTSTRAP
    sampled = resample(strata, random.Random(seed + i))
    b, ba = official_index(sampled, baseline)
    c, ca = official_index(sampled, candidate)
    return c - b, {a: ca[a] - ba[a] for a in ba}


def main():
    from decision_index.suite.io import Suite
    from decision_index.scoring.report import load_results
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--replicates', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20261008)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    ids = read_ids(root / 'run-ids.txt', manifest['run_ids_sha256'])
    suite = Suite(args.suite, '0.3'); suite.verify()
    rows = load_proxy(suite, ids).selected
    baseline = {k: v for k, v in load_results(args.baseline).items() if k in ids}
    candidate = {k: v for k, v in load_results(args.candidate).items() if k in ids}
    if set(baseline) != ids or set(candidate) != ids:
        raise ValueError('bootstrap requires complete manifest coverage in both runs')
    bscore, bareas = official_index(rows, baseline)
    cscore, careas = official_index(rows, candidate)
    strata = group_strata(rows)
    deltas, area_deltas = [], collections.defaultdict(list)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork'),
                             initializer=initialize_worker,
                             initargs=(strata, baseline, candidate, args.seed)) as pool:
        for i, (delta, differences) in enumerate(pool.map(paired_draw, range(args.replicates), chunksize=5)):
            deltas.append(delta)
            for a, difference in differences.items():
                area_deltas[a].append(difference)
            if (i + 1) % 25 == 0:
                print(json.dumps(dict(replicates=i+1, seconds=round(time.monotonic()-start, 1))), flush=True)
    output = dict(proxy=True, note='Proxy, not a leaderboard score; conditional interval over this frozen proxy, not full-suite sampling uncertainty.',
                  method='paired percentile bootstrap of complete catalog_id/group_id units with replacement within benchmark/domain/track strata; official scorer recomputed for each draw',
                  seed=args.seed, replicate_seed_rule='seed + zero-based replicate index', replicates=args.replicates, manifest_sha256=manifest['run_ids_sha256'],
                  baseline=bscore, candidate=cscore, difference=cscore-bscore,
                  difference_ci95=[quantile(deltas, .025), quantile(deltas, .975)],
                  areas={a: dict(difference=careas[a]-bareas[a], difference_ci95=[quantile(v,.025),quantile(v,.975)]) for a,v in area_deltas.items()})
    args.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
