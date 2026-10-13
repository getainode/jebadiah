# SPDX-License-Identifier: Apache-2.0
"""Monitor paid ceilings, then collect, proxy, bootstrap and open the unmerged PR."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from t2b_monitor import poll,message
from t2b_jobs import TERMINAL,save
from t2b_collect import collect_train,collect_proxy
from t2b_finish import finish


def wait_stage(root,stage):
    misses=0
    while True:
        try:
            ledger=poll(root);misses=0
        except Exception as exc:
            # Preserve remote jobs and their timeout ceilings on contact loss. A blip
            # (Studio IPv6/route flaps) gets 40 polls, about 30 min, before we give up.
            misses+=1
            if misses<40:
                print(f'poll failed ({misses}/40), retrying: {type(exc).__name__}',flush=True);time.sleep(45);continue
            error=str(exc).replace(os.environ.get('HF_TOKEN','') or 'impossible-secret-placeholder','[REDACTED]')
            message(root,'question','T2b monitoring blocked',f'Monitoring error: {error}. Remote jobs retain hard timeouts; no replacement or extension launched. Please advise if connectivity cannot be restored.',f'monitor-error-{stage}-send.json')
            time.sleep(45)
            continue
        entries=[ledger['jobs'][f'{stage}-s{s}'] for s in (17,18)]
        if any(e['status'] in TERMINAL and e['status']!='COMPLETED' for e in entries):
            save(root/'blocked.json',{'stage':stage,'reason':'Paid job ended unsuccessfully; no retry authorized','jobs':{f'{stage}-s{s}':ledger['jobs'][f'{stage}-s{s}'] for s in (17,18)}})
            raise RuntimeError('Paid job failed; question sent; no automatic paid retry')
        if all(e['status']=='COMPLETED' for e in entries):return ledger
        time.sleep(45)


def run(root):
    repo=root/'repo';py=sys.executable
    subprocess.run([py,'-m','unittest','discover','-s','scripts','-p','test_t2b*launch.py'],cwd=repo,check=True)
    ledger=wait_stage(root,'train')
    for seed in (17,18):collect_train(root,seed)
    # Proxies start only once both train-plus-heldout jobs and evidence checks pass.
    for seed in (17,18):
        ledger=json.loads((root/'jobs.json').read_text());revision=ledger['jobs'][f'train-s{seed}']['model_revision']
        subprocess.run([py,str(repo/'scripts/t2b_jobs.py'),'--root',str(root),'proxy','--seed',str(seed),'--model-revision',revision],cwd=repo,check=True)
    wait_stage(root,'proxy')
    for seed in (17,18):collect_proxy(root,seed)
    analysis=root/'analysis';analysis.mkdir(exist_ok=True)
    kit=Path('/Volumes/PRO-G40/scratch/lead-ainode/t2b0/scorer/decision-index');suite='/Volumes/PRO-G40/scratch/lead-ainode/suite-0.3'
    env=os.environ.copy();env['PYTHONPATH']=str(kit);env['TMPDIR']=str(root/'tmp')
    for seed in (17,18):
        output=analysis/f's{seed}-bootstrap.json'
        with (root/f'bootstrap-s{seed}.log').open('w') as log:
            subprocess.run([py,str(repo/'eval/decision-index/proxy-0.3-10pct/bootstrap.py'),'--suite',suite,'--baseline',f'/Volumes/PRO-G40/scratch/lead-ainode/t2b0/artifacts/s{seed}/proxy/results.jsonl.gz','--candidate',str(root/'artifacts'/f's{seed}'/'proxy/results.jsonl.gz'),'--out',str(output),'--replicates','2000','--workers','16'],env=env,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    args=[py,str(repo/'scripts/t2b_analyze.py'),'--suite',suite,'--out',str(analysis),'--workers','16','--replicates','2000']
    for seed in (17,18):args+=['--baseline-s'+str(seed),f'/Volumes/PRO-G40/scratch/lead-ainode/t2b0/artifacts/s{seed}/proxy/results.jsonl.gz','--candidate-s'+str(seed),str(root/'artifacts'/f's{seed}'/'proxy/results.jsonl.gz')]
    with (root/'bootstrap-mean.log').open('w') as log:subprocess.run(args,env=env,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([py,'-m','unittest','discover','-s','scripts','-p','test_t2b*launch.py'],cwd=repo,check=True)
    finish(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    try:run(root)
    except Exception as exc:
        error=str(exc).replace(os.environ.get('HF_TOKEN','') or 'impossible-secret-placeholder','[REDACTED]')
        if not (root/'blocked.json').exists():message(root,'question','T2b pipeline blocked',error+'. No paid retry or timeout extension launched; existing job ceilings remain under USD 65.','pipeline-blocked-send.json')
        raise
