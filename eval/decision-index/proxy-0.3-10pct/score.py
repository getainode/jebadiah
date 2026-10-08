#!/usr/bin/env python3
"""Apply the unmodified Decision Index 0.3 scorer to manifest rows only."""
import argparse
import collections
import json
from pathlib import Path
from sample import read_ids


class ProxySuite:
    def __init__(self, suite, rows):
        self.edition = suite.edition
        self.selected = rows

    def rows(self, apply_exclusions=False):
        return iter(self.selected)


def load_proxy(suite, ids):
    rows = [r for r in suite.rows(apply_exclusions=True) if r['_evaluation']['run_id'] in ids]
    found = {r['_evaluation']['run_id'] for r in rows}
    if found != ids:
        raise ValueError(f'{len(ids - found)} manifest IDs absent')
    groups = collections.defaultdict(set)
    for r in suite.rows(apply_exclusions=True):
        e = r['_evaluation']
        groups[(e['catalog_id'], e['group_id'])].add(e['run_id'])
    if any(m & ids and not m <= ids for m in groups.values()):
        raise ValueError('manifest splits a group')
    return ProxySuite(suite, rows)


def score(proxy, results_path, out, engine):
    from decision_index.pipeline import score_run_v02
    from decision_index.scoring.report import load_results
    results = load_results(results_path)
    expected = {r['_evaluation']['run_id'] for r in proxy.selected}
    Path(out).mkdir(parents=True, exist_ok=True)
    selected_results = {rid: result for rid, result in results.items() if rid in expected}
    result = score_run_v02(proxy, selected_results, engine, Path(out))
    present = expected & results.keys()
    groups = collections.defaultdict(list)
    for r in proxy.selected:
        e = r['_evaluation']
        groups[(e['catalog_id'], e['group_id'])].append(e['run_id'])
    result['proxy'] = True
    result['note'] = 'Frozen 10% per-benchmark proxy, not a leaderboard score.'
    result['expected_rows'] = len(expected)
    result['completed'] = len(present)
    result['complete'] = len(present) == len(expected)
    result['counts'] = dict(collections.Counter(results[rid]['status'] for rid in present))
    result['missing_rows'] = sorted(expected - present)
    result['outside_manifest_rows'] = len(results.keys() - expected)
    result['whole_case_failures'] = sum(any(results.get(rid, {}).get('status') != 'ok' for rid in members) for members in groups.values())
    result['total_groups'] = len(groups)
    Path(out, 'scores.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    from decision_index.suite.io import Suite
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True)
    p.add_argument('--ids', type=Path, default=Path(__file__).with_name('run-ids.txt'))
    p.add_argument('--sha256', default=Path(__file__).with_name('run-ids.sha256').read_text().split()[0])
    p.add_argument('--results', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--engine', required=True)
    args = p.parse_args()
    suite = Suite(args.suite, '0.3')
    suite.verify()
    proxy = load_proxy(suite, read_ids(args.ids, args.sha256))
    result = score(proxy, args.results, args.out, args.engine)
    print(json.dumps({k: result[k] for k in ('decision_index', 'areas', 'completed', 'expected_rows', 'counts', 'whole_case_failures', 'outside_manifest_rows')}))


if __name__ == '__main__':
    main()
