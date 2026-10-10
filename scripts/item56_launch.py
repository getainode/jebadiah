# SPDX-License-Identifier: Apache-2.0
"""Pin the original A3 runtime, guarded seeds and RTX PRO 6000 allocation ceiling."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from item52_launch import command as control_command

ROOT=Path(__file__).resolve().parents[1]
# Both train/proxy seeds plus one four-model diagnostic job: 415 min = $19.0208.
CAP_MINUTES={'train':150,'proxy':35,'diagnostic':45}


def command(stage,revision,code_revision,seed=17,data_revision=None):
    if seed not in (17,18):raise ValueError('Only approved seeds')
    for value in (revision,code_revision,data_revision or revision):
        if not re.fullmatch('[0-9a-f]{40}',value):raise ValueError('Immutable revision required')
    model=f'frontier-infra/jebadiah-9b-v2-1-probe-b-s{seed}'
    cfg=json.loads((ROOT/'configs/item56-probe-b.json').read_text())
    if not cfg.get('dataset_revision'):raise ValueError('Scanned math and final probe revision required')
    if stage in ('train','proxy'):
        cmd=control_command(stage,code_revision if stage=='train' else revision,seed)
        setup=cmd[-1]
        old='frontier-infra/jebadiah-9b-v2-1-a3-rep1' if seed==17 else 'frontier-infra/jebadiah-9b-v2-1-a3-seed18'
        setup=setup.replace(old,model).replace('a3-rep1' if seed==17 else 'a3-seed18',f'probe-b-s{seed}')
        if stage=='train':
            assert data_revision==cfg['dataset_revision']
            setup=setup.replace('item32','item47').replace('bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed',data_revision)
            setup=setup.replace('a3/train.jsonl','probe-b/train.jsonl').replace('a3/calib.jsonl','probe-b/calib.jsonl').replace('a3/manifest.json','probe-b/manifest.json')
            setup=setup.replace(f'git show {code_revision}:scripts/item52_persist.py > /workspace/item52-persist.py',
                                f'git show {code_revision}:scripts/item56_persist.py > /workspace/item52-persist.py\n'
                                f'git show {code_revision}:scripts/item56_train.py > /workspace/item56-train.py')
            setup=setup.replace('JEB_SKIP_INSTALL=1 bash scripts/v21.sh',
                                'JEB_SKIP_INSTALL=1 "$JEB_ROOT/venv/bin/python" /workspace/item56-train.py --root "$JEB_ROOT"')
            setup=setup.replace('export ITEM52_MODEL_REPO=',f'export ITEM56_DATA_REVISION={data_revision} ITEM56_MODEL_REPO={model}\nexport ITEM52_MODEL_REPO=')
            setup=setup.replace('/workspace/item52-train.log','/workspace/item56-train.log')
        cmd[-1]=setup
        cmd[cmd.index('--name')+1]=f'item56-{stage}-s{seed}'
        cmd[cmd.index('--timeout')+1]=f'{CAP_MINUTES[stage]}m'
        return cmd
    raise ValueError('Unsupported stage')


def diagnostic_command(config,code_revision):
    if not re.fullmatch('[0-9a-f]{40}',code_revision):raise ValueError('Immutable code revision required')
    payload=shlex.quote(json.dumps(config,separators=(',',':')))
    setup=f"""set -euo pipefail
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git build-essential >/dev/null 2>&1
git clone -q https://github.com/getainode/jebadiah.git /workspace/item56-src
cd /workspace/item56-src
git checkout -q 86a8203d792463536eb009a4d8da6c8681ffdd99
export JEB_ROOT=/workspace/item56-diagnostic
for script in item56_diagnostic item51_diagnostic item54_diagnostic_report; do
  git show {code_revision}:scripts/$script.py > /workspace/$script.py
done
printf '%s' {payload} > /workspace/item56-diagnostic-config.json
bash scripts/v21.sh --help >/dev/null
"$JEB_ROOT/venv/bin/python" -m pip install -q hf_xet ninja packaging
MAX_JOBS=16 timeout 540 "$JEB_ROOT/venv/bin/python" -m pip install -q --no-build-isolation causal-conv1d
"$JEB_ROOT/venv/bin/python" /workspace/item56_diagnostic.py --config /workspace/item56-diagnostic-config.json
"""
    return ['hf','jobs','run','--detach','--name','item56-diagnostics','--flavor','rtx-pro-6000',
            '--timeout',str(CAP_MINUTES['diagnostic'])+'m','--secrets','HF_TOKEN',
            'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel','--','bash','-lc',setup]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['train','proxy']);p.add_argument('revision')
    p.add_argument('--code-revision',required=True);p.add_argument('--data-revision')
    p.add_argument('--seed',type=int,choices=[17,18],required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();cmd=command(a.stage,a.revision,a.code_revision,a.seed,a.data_revision)
    if a.dry_run:print(shlex.join(cmd))
    else:
        from huggingface_hub import get_token
        os.environ['HF_TOKEN']=os.environ.get('HF_TOKEN') or get_token() or ''
        if not os.environ['HF_TOKEN']:p.error('Existing authorized HF session required')
        subprocess.run(cmd,check=True)
