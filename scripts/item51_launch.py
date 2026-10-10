"""Launch one private item51 training run with a fixed allocation budget.

HF_TOKEN uses the explicitly authorized logged-in session. Rung 2 changes only matched knowledge slots.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_MINUTES = {'rung2b': 130}


def command(run, dataset_revision, code_revision):
    cfg = json.loads((ROOT / f'configs/item51-{run}.json').read_text())
    flags = ['--sizes', '9b', '--dataset-id', cfg['dataset_repo'],
             '--dataset-revision', dataset_revision, '--train-file', cfg['train_file'],
             '--calib-file', cfg['calib_file'], '--manifest-file', cfg['manifest_file'],
             '--model-repo', cfg['model_repo'], '--checkpoint-repo', cfg['checkpoint_repo'],
             '--rank', str(cfg['lora_rank']), '--alpha', str(cfg['lora_alpha']),
             '--max-seq-length', str(cfg['max_seq_length']), '--epochs', str(cfg['num_epochs']),
             '--lr', str(cfg['learning_rate']), '--lora-dropout', str(cfg['decide']['lora_dropout']), '--microbatch', str(cfg['batch_size']),
             '--accumulation', str(cfg['gradient_accumulation_steps']), '--group-by-length',
             '--pad-to-multiple-of', '64', '--checkpoint-min-tokens', '512',
             '--eval-steps', str(cfg['eval_steps']), '--calib-eval-limit', str(cfg['decide']['calib_eval_limit']),
             '--save-steps', '500', '--resume', 'none']
    setup = f'''set -euo pipefail
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git build-essential >/dev/null 2>&1
git clone -q https://github.com/getainode/jebadiah.git /workspace/item51-src
cd /workspace/item51-src
git checkout -q {code_revision}
export JEB_ROOT=/workspace/item51-{run}
export HF_HOME="$JEB_ROOT/hf" TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/v21.sh --help >/dev/null
"$JEB_ROOT/venv/bin/python" -m pip install -q hf_xet ninja packaging
MAX_JOBS=16 timeout 540 "$JEB_ROOT/venv/bin/python" -m pip install -q --no-build-isolation causal-conv1d
set +e
JEB_SKIP_INSTALL=1 "$JEB_ROOT/venv/bin/python" scripts/item51_train.py --root "$JEB_ROOT" {shlex.join(flags)} 2>&1 | tee /workspace/item51-train.log
status=${{PIPESTATUS[0]}}
"$JEB_ROOT/venv/bin/python" scripts/item51_persist.py --repo {cfg["checkpoint_repo"]}
exit "$status"
'''
    return ['hf', 'jobs', 'run', '--detach', '--name', f'item51-{run}', '--flavor', 'rtx-pro-6000',
            '--timeout', f'{TIMEOUT_MINUTES[run]}m', '--secrets', 'HF_TOKEN',
            'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel', '--', 'bash', '-lc', setup]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', choices=TIMEOUT_MINUTES)
    p.add_argument('dataset_revision')
    p.add_argument('--code-revision', default=None)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    revision = a.code_revision or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if not all(re.fullmatch(r'[0-9a-f]{40}', x) for x in (revision, a.dataset_revision)):
        p.error('Code and data must use immutable SHA1 revisions')
    cmd = command(a.run, a.dataset_revision, revision)
    if a.dry_run:
        print(shlex.join(cmd))
    else:
        if not os.environ.get('HF_TOKEN'):
            os.environ['HF_TOKEN'] = __import__('huggingface_hub').get_token() or ''
        if not os.environ['HF_TOKEN']:
            p.error('Logged-in HF session required')
        subprocess.run(cmd, check=True)
