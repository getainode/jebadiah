"""Score the paired frozen proxy and bootstrap through the unchanged official kit.

Inputs are local paths inside the item39 PRO-G40 scratch directory. This script
never performs inference, changes scoring, or publishes a model.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / 'eval/decision-index/proxy-0.3-10pct'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, results in [('published', a.baseline), ('a3', a.candidate)]:
        subprocess.run([sys.executable, str(PROXY/'score.py'), '--suite', str(a.suite),
                        '--results', str(results), '--engine', 'item39-27b-'+name,
                        '--out', str(a.out/name)], check=True)
    subprocess.run([sys.executable, str(PROXY/'bootstrap.py'), '--suite', str(a.suite),
                    '--baseline', str(a.baseline), '--candidate', str(a.candidate),
                    '--replicates', '2000', '--seed', '20261008', '--workers', str(a.workers),
                    '--out', str(a.out/'item39-paired-bootstrap.json')], check=True)
    b = json.loads((a.out/'item39-paired-bootstrap.json').read_text())
    lo, hi = b['difference_ci95']
    summary = {'published_proxy': b['baseline'], 'a3_proxy': b['candidate'],
               'difference': b['difference'], 'difference_ci95': [lo, hi],
               'area_deltas': {k: v['difference'] for k, v in b['areas'].items()},
               'statistical_keep': lo > 0, 'proxy': True, 'private_only': True,
               'note': 'A positive interval does not authorize shipping or submission.'}
    (a.out/'item39-score-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
