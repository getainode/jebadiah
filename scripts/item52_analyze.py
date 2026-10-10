"""CPU-only exact official scoring and four paired complete-group comparisons."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / 'eval/decision-index/proxy-0.3-10pct'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    for name in ('a3', 'replicate', 'rung1', 'rung2', 'rung2b'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PROXY))
    from sample import read_ids
    from score import load_proxy
    from decision_index.suite.io import Suite
    manifest = json.loads((PROXY / 'manifest.json').read_text())
    ids = read_ids(PROXY / 'run-ids.txt', manifest['run_ids_sha256'])
    suite = Suite(a.suite, '0.3')
    suite.verify()
    expected = {r['_evaluation']['run_id']: r['_evaluation']['payload_sha256']
                for r in load_proxy(suite, ids).selected}
    coverage = {}
    for name in ('a3', 'replicate', 'rung1', 'rung2', 'rung2b'):
        source = getattr(a, name)
        opener = gzip.open if source.suffix == '.gz' else open
        with opener(source, 'rt') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        actual = {r['run_id']: r.get('payload_sha256') for r in rows}
        if len(rows) != len(actual) or actual != expected:
            raise ValueError(f'{name}: duplicate rows or frozen proxy IDs/payload hashes differ')
        if any(r['status'] != 'ok' for r in rows):
            raise ValueError(f'{name}: incomplete successful coverage')
        coverage[name] = {'rows': len(rows), 'unique_ids': len(actual), 'all_ok': True,
                          'payload_hashes_match_frozen_suite': True}
        subprocess.run([sys.executable, str(PROXY / 'score.py'), '--suite', str(a.suite),
                        '--results', str(source), '--engine', 'item52-' + name,
                        '--out', str(a.out / name)], check=True)
    (a.out / 'coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')
    pairs = [('a3', 'replicate'), ('replicate', 'rung1'),
             ('replicate', 'rung2'), ('replicate', 'rung2b')]
    summary = {'control_run': True, 'proxy': True, 'private_only': True, 'comparisons': {}}
    for baseline, candidate in pairs:
        key = candidate + '-vs-' + baseline
        output = a.out / (key + '-bootstrap.json')
        subprocess.run([sys.executable, str(PROXY / 'bootstrap.py'), '--suite', str(a.suite),
                        '--baseline', str(getattr(a, baseline)), '--candidate', str(getattr(a, candidate)),
                        '--replicates', '2000', '--seed', '20261008', '--workers', str(a.workers),
                        '--out', str(output)], check=True)
        b = json.loads(output.read_text())
        summary['comparisons'][key] = {k: b[k] for k in
                                      ('baseline', 'candidate', 'difference', 'difference_ci95', 'areas')}
        # Persist each completed comparison; never serialize the internal tuple-key map.
        (a.out / 'item52-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('ITEM52_COMPLETED_COMPARISON', key, json.dumps(summary['comparisons'][key]), flush=True)
    indices = {name: json.loads((a.out / name / 'index.json').read_text())
               for name in ('a3', 'replicate', 'rung1', 'rung2', 'rung2b')}
    native = {name: json.loads((a.out / name / 'benchmark-summary.json').read_text())
              for name in indices}
    spec = json.loads((Path(__import__('decision_index').__file__).parent / 'data/index-0.3.json').read_text())
    benchmarks = {}
    for catalog, entry in spec['chance'].items():
        if entry.get('name') in ('CLINC150', 'CLadder', 'POP909'):
            benchmarks[entry['name']] = {
                name: {k: index['benchmarks'][catalog][k] for k in ('raw', 'skill', 'coverage')}
                for name, index in indices.items()}
            for name in indices:
                row = next(r for r in native[name]['benchmarks'] if r['catalog_id'] == int(catalog))
                point = benchmarks[entry['name']][name]
                point['raw'] = (row['detail']['cluster_macro_accuracy']
                                if entry['name'] == 'POP909' else row['score'])
                point['raw_metric'] = 'cluster macro accuracy' if entry['name'] == 'POP909' else row['metric']
    summary['benchmarks'] = benchmarks
    summary['areas'] = {name: {v['id']: 100 * v['skill'] for v in index['areas']}
                        for name, index in indices.items()}
    summary['replicate_loses_tools_vs_a3'] = summary['comparisons']['replicate-vs-a3']['areas']['tools']['difference'] < 0
    summary['replicate_loses_clinc_vs_a3'] = benchmarks['CLINC150']['replicate']['raw'] < benchmarks['CLINC150']['a3']['raw']
    (a.out / 'item52-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
