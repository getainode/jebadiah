# SPDX-License-Identifier: Apache-2.0
"""Apply the unchanged official proxy scorer and summarize both clean-baseline seeds."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / 'eval/decision-index/proxy-0.3-10pct'
SELECTED_SHA256 = '1f92d9723b24ca5483474bf324d0ed8e5d542770f89694a579244c45a59cb7fe'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    for seed in (17, 18):
        parser.add_argument(f'--proxy-s{seed}', type=Path, required=True)
        parser.add_argument(f'--heldout-s{seed}', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.kit))
    sys.path.insert(0, str(PROXY))
    from decision_index.suite.io import Suite
    from sample import read_ids
    from score import load_proxy, score
    from bootstrap import official_index
    assert hashlib.sha256((args.suite / 'selected-rows.jsonl.gz').read_bytes()).hexdigest() == SELECTED_SHA256
    suite = Suite(args.suite, '0.3')
    suite.verify()
    manifest = json.loads((PROXY / 'manifest.json').read_text())
    ids = read_ids(PROXY / 'run-ids.txt', manifest['run_ids_sha256'])
    proxy = load_proxy(suite, ids)
    expected = {r['_evaluation']['run_id']: r['_evaluation']['payload_sha256'] for r in proxy.selected}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'clean_baseline': True, 'candidate_keep_decision': False,
              'a3_reference': 46.82, 'a3_unrounded_two_seed_mean': 46.81940927298617,
              'manifest_sha256': hashlib.sha256((PROXY / 'manifest.json').read_bytes()).hexdigest(),
              'selected_sha256': SELECTED_SHA256, 'per_seed': {}}
    for seed in (17, 18):
        source = getattr(args, f'proxy_s{seed}')
        opener = gzip.open if source.suffix == '.gz' else open
        with opener(source, 'rt') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        actual = {r['run_id']: r.get('payload_sha256') for r in rows}
        if len(rows) != len(actual) or actual != expected or any(r['status'] != 'ok' for r in rows):
            raise ValueError('Duplicate, incomplete or mismatched frozen proxy results')
        results = {r['run_id']: r for r in rows}
        score(proxy, source, args.out / f's{seed}', f't2b0-s{seed}')
        headline, areas = official_index(proxy.selected, results)
        heldout = json.loads(getattr(args, f'heldout_s{seed}').read_text())
        assert heldout['seed'] == seed and heldout['questions'] == 3000
        assert heldout['truncated_prompts'] == 0
        families = {k: v for k, v in heldout['metrics'].items() if k != 'overall'}
        assert len(families) == 5 and all(v['n'] == 600 for v in families.values())
        report['per_seed'][str(seed)] = {
            'proxy': headline, 'areas': areas, 'proxy_records': len(rows),
            'all_proxy_records_ok': True, 'proxy_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'heldout_accuracy': heldout['metrics']['overall']['accuracy'],
            'heldout_family_macro_accuracy': heldout['family_macro_accuracy'],
            'heldout_per_family': families,
            'heldout_sha256': heldout['heldout_sha256'],
            'dataset_revision': heldout['dataset_revision']}
    report['proxy_mean'] = sum(v['proxy'] for v in report['per_seed'].values()) / 2
    report['proxy_difference_vs_46_82'] = report['proxy_mean'] - 46.82
    report['proxy_difference_vs_unrounded_A3'] = report['proxy_mean'] - report['a3_unrounded_two_seed_mean']
    report['heldout_family_mean'] = sum(v['heldout_family_macro_accuracy'] for v in report['per_seed'].values()) / 2
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
