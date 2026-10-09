"""Launch the frozen 11,079-row proxy for a 27B A3 checkpoint or published baseline."""
import argparse
import os
import re
import shlex
import subprocess

RESULTS = 'jbrashear/jebadiah-9b-v2-1-index-results'
# Exact item31 runner, engine, source kit and manifest snapshot.
CODE_REVISION = '587f4f496b9cf1d83f0bc8e495379496f65afa2f'


def command(run, revision):
    model = ('frontier-infra/jebadiah-27b' if run == 'published' else 'frontier-infra/jebadiah-27b-v2-1-a3')
    name = f'item39-27b-{run}-{revision[:8]}-proxy-0.3-10pct'
    options = (f'--option model={model} --option revision={revision} '
               '--option prefix_cache=true --option max_batch=64 '
               '--option cuda_alloc_conf=expandable_segments:True --option batch_tokens=32768')
    patch = "from pathlib import Path; p=Path('/tmp/src/code/indexrun-job.sh'); s=p.read_text(); "
    for old in ('retry hf download "$RESULTS_REPO" code/decision-index-src.tar.gz',
                'retry hf download "$RESULTS_REPO" --include "code/proxy/*"'):
        new = old.replace('"$RESULTS_REPO"', '"$RESULTS_REPO" --revision ' + CODE_REVISION)
        patch += f'assert s.count({old!r}) == 1; s=s.replace({old!r}, {new!r}); '
    patch += 'p.write_text(s)'
    setup = f''' set -euo pipefail
export HF_HUB_DISABLE_XET=0
pip install -q 'huggingface_hub==1.33.0' hf_xet
hf download {RESULTS} --revision {CODE_REVISION} --include 'code/*' --repo-type dataset --local-dir /tmp/src
python -c {shlex.quote(patch)}
ENGINE=jebadiah_engine:JebadiahEngine RUN_NAME={name} ENGINE_OPTS={shlex.quote(options)} RESULTS_REPO={RESULTS} SUITE_DATASET=jbrashear/decision-index-suite-0.3 SYNC_SEC=120 ROWS_IN_SUITE='' LIMIT='' ATTEMPTS=1 bash /tmp/src/code/indexrun-job.sh 2>&1 | tee /workspace/item39-indexrun-job.log
'''
    return ['hf', 'jobs', 'run', '--detach', '--name', f'item39-{run}-proxy',
            '--flavor', 'rtx-pro-6000', '--timeout', '80m', '--secrets', 'HF_TOKEN',
            'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime', '--', 'bash', '-lc', setup]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', choices=['a3', 'published'])
    p.add_argument('model_revision')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', a.model_revision):
        p.error('Model revision must be an immutable SHA1')
    cmd = command(a.run, a.model_revision)
    if a.dry_run:
        print(shlex.join(cmd))
    else:
        if not os.environ.get('HF_TOKEN'):
            p.error('Bitwarden-sourced HF_TOKEN required')
        subprocess.run(cmd, check=True)
