"""Exact A3 control replicate, using the original runtime commit and launcher flags."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = '86a8203d792463536eb009a4d8da6c8681ffdd99'
DATA_REVISION = 'bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed'
MODEL = 'frontier-infra/jebadiah-9b-v2-1-a3-rep1'
RESULTS = 'jbrashear/jebadiah-9b-v2-1-index-results'
PROXY_CODE = '587f4f496b9cf1d83f0bc8e495379496f65afa2f'


def command(stage, revision):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Immutable code or model revision required')
    if stage == 'train':
        cfg = json.loads((ROOT / 'configs/item32-a3.json').read_text())
        flags = ['--sizes', '9b', '--dataset-id', cfg['dataset_repo'],
                 '--dataset-revision', DATA_REVISION, '--train-file', cfg['train_file'],
                 '--calib-file', cfg['calib_file'], '--manifest-file', cfg['manifest_file'],
                 '--model-repo', MODEL, '--checkpoint-repo', MODEL + '-checkpoints',
                 '--rank', '16', '--alpha', '32', '--max-seq-length', '2048',
                 '--epochs', '1', '--lr', '0.0001', '--microbatch', '8',
                 '--accumulation', '1', '--group-by-length', '--pad-to-multiple-of', '64',
                 '--checkpoint-min-tokens', '512', '--eval-steps', '200',
                 '--calib-eval-limit', '358', '--save-steps', '500', '--resume', 'none']
        setup = f'''set -euo pipefail
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git build-essential >/dev/null 2>&1
git clone -q https://github.com/getainode/jebadiah.git /workspace/item52-src
cd /workspace/item52-src
git checkout -q {ORIGINAL_CODE}
git show {revision}:scripts/item52_environment.py > /workspace/item52-environment.py
git show {revision}:scripts/item52_persist.py > /workspace/item52-persist.py
export JEB_ROOT=/workspace/item52-a3-rep1
bash scripts/v21.sh --help >/dev/null
"$JEB_ROOT/venv/bin/python" -m pip install -q hf_xet ninja packaging
MAX_JOBS=16 timeout 540 "$JEB_ROOT/venv/bin/python" -m pip install -q --no-build-isolation causal-conv1d
"$JEB_ROOT/venv/bin/python" /workspace/item52-environment.py
set +e
JEB_SKIP_INSTALL=1 bash scripts/v21.sh {shlex.join(flags)} 2>&1 | tee /workspace/item52-train.log
status=${{PIPESTATUS[0]}}
"$JEB_ROOT/venv/bin/python" /workspace/item52-persist.py
exit "$status"
'''
        image, minutes = 'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel', 135
    else:
        name = f'jebadiah-9b-v2-1-a3-rep1-{revision[:8]}-proxy-0.3-10pct'
        options = (f'--option model={MODEL} --option revision={revision} '
                   '--option prefix_cache=true --option max_batch=64 '
                   '--option cuda_alloc_conf=expandable_segments:True --option batch_tokens=32768')
        setup = f'''set -euo pipefail
export HF_HUB_DISABLE_XET=0
pip install -q 'huggingface_hub==1.33.0' hf_xet
hf download {RESULTS} --revision {PROXY_CODE} --include 'code/*' --repo-type dataset --local-dir /tmp/src
ENGINE=jebadiah_engine:JebadiahEngine RUN_NAME={name} ENGINE_OPTS={shlex.quote(options)} RESULTS_REPO={RESULTS} SUITE_DATASET=jbrashear/decision-index-suite-0.3 SYNC_SEC=120 ROWS_IN_SUITE='' LIMIT='' ATTEMPTS=1 bash /tmp/src/code/indexrun-job.sh 2>&1 | tee /workspace/item52-indexrun-job.log
'''
        image, minutes = 'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime', 35
    return ['hf', 'jobs', 'run', '--detach', '--name', f'item52-{stage}',
            '--flavor', 'rtx-pro-6000', '--timeout', f'{minutes}m',
            '--secrets', 'HF_TOKEN', image, '--', 'bash', '-lc', setup]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['train', 'proxy'])
    p.add_argument('revision')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cmd = command(a.stage, a.revision)
    if a.dry_run:
        print(shlex.join(cmd))
    else:
        from huggingface_hub import get_token
        os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN') or get_token() or ''
        if not os.environ['HF_TOKEN']:
            p.error('Authorized existing HF session required')
        subprocess.run(cmd, check=True)
