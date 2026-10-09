"""Launch the frozen 11,079-row proxy for a private item44 checkpoint."""
import argparse
import os
import re
import shlex
import subprocess

RESULTS = 'jbrashear/jebadiah-9b-v2-1-index-results'
# Exact item31 runner, engine, source kit and manifest snapshot.
CODE_REVISION = '587f4f496b9cf1d83f0bc8e495379496f65afa2f'


def command(run, revision):
    model = f'frontier-infra/jebadiah-9b-v2-1-{run}'
    name = f'jebadiah-9b-v2-1-{run}-{revision[:8]}-proxy-0.3-10pct'
    options = (f'--option model={model} --option revision={revision} '
               '--option prefix_cache=true --option max_batch=64 '
               '--option cuda_alloc_conf=expandable_segments:True --option batch_tokens=32768')
    setup = f'''set -euo pipefail
export HF_HUB_DISABLE_XET=0
pip install -q 'huggingface_hub==1.33.0' hf_xet
hf download {RESULTS} --revision {CODE_REVISION} --include 'code/*' --repo-type dataset --local-dir /tmp/src
ENGINE=jebadiah_engine:JebadiahEngine RUN_NAME={name} ENGINE_OPTS={shlex.quote(options)} RESULTS_REPO={RESULTS} SUITE_DATASET=jbrashear/decision-index-suite-0.3 SYNC_SEC=120 ROWS_IN_SUITE='' LIMIT='' ATTEMPTS=1 bash /tmp/src/code/indexrun-job.sh 2>&1 | tee /workspace/item44-indexrun-job.log
'''
    return ['hf', 'jobs', 'run', '--detach', '--name', f'item44-{run}-proxy',
            '--flavor', 'rtx-pro-6000', '--timeout', '35m', '--secrets', 'HF_TOKEN',
            'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime', '--', 'bash', '-lc', setup]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', choices=['n2'])
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
            os.environ['HF_TOKEN'] = __import__('huggingface_hub').get_token() or ''
        if not os.environ['HF_TOKEN']:
            p.error('Logged-in HF session required')
        subprocess.run(cmd, check=True)
