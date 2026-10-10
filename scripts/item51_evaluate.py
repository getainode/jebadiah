"""Launch the paired causal diagnostic or strict export recovery within the $12 cap."""
import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path
from huggingface_hub import get_token
ROOT=Path(__file__).resolve().parents[1]


def command(stage, revision, code_revision, rung2_revision):
    script = ('scripts/item51_diagnostic.py ' + revision + ' ' + rung2_revision if stage == 'diagnostic' else
              'scripts/item51_recover.py rung2b ' + revision + ' ' + __import__('json').loads((ROOT / 'configs/item51-rung2b.json').read_text())['dataset_revision'])
    setup=f'''set -euo pipefail
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git build-essential >/dev/null 2>&1
git clone -q https://github.com/getainode/jebadiah.git /workspace/item51-src
cd /workspace/item51-src
git checkout -q {code_revision}
export JEB_ROOT=/workspace/item51-{stage}
export HF_HOME="$JEB_ROOT/hf" TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/v21.sh --help >/dev/null
"$JEB_ROOT/venv/bin/python" -m pip install -q hf_xet ninja packaging
MAX_JOBS=16 timeout 540 "$JEB_ROOT/venv/bin/python" -m pip install -q --no-build-isolation causal-conv1d
"$JEB_ROOT/venv/bin/python" {script}
'''
    return ['hf','jobs','run','--detach','--name',f'item51-{stage}','--flavor','rtx-pro-6000',
            '--timeout','45m','--secrets','HF_TOKEN','pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel',
            '--','bash','-lc',setup]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['diagnostic','recovery'])
    p.add_argument('revision')
    p.add_argument('--rung2-revision', required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if not re.fullmatch('[0-9a-f]{40}',a.revision): p.error('Immutable revision required')
    cmd=command(a.stage,a.revision,code,a.rung2_revision)
    if a.dry_run: print(shlex.join(cmd))
    else:
        os.environ['HF_TOKEN']=os.environ.get('HF_TOKEN') or get_token() or ''
        if not os.environ['HF_TOKEN']: p.error('Authorized logged-in HF session required')
        subprocess.run(cmd,check=True)
