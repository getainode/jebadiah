# SPDX-License-Identifier: Apache-2.0
"""Inspect bounded paid jobs; send the authorized checkpoint or blocking message once."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess
from huggingface_hub import HfApi
from t2b_jobs import refresh,save,TERMINAL,RATE

RUN='run:run_35087d489558'


def message(root,kind,subject,body,marker):
    path=root/marker
    if path.exists(): return
    # Reserve before send. An ambiguous transport receipt must be reconciled, never resent.
    save(path,{'state':'SEND_RESERVED','subject':subject,'body':body,'type':kind})
    cp=subprocess.run(['orca','orchestration','send','--to',RUN,'--type',kind,'--subject',subject,'--body',body,'--json'],text=True,capture_output=True,check=True)
    path.write_text(cp.stdout)
    print('MESSAGE',kind,subject,flush=True)


def cost(ledger):
    now=datetime.now(timezone.utc)
    total=ledger.get('smoke_usd',0)
    for e in ledger['jobs'].values():
        ts=e.get('timestamps',{})
        start=ts.get('started_at');end=ts.get('finished_at')
        if not start: e['estimated_usd']=0;continue
        seconds=max(0,((datetime.fromisoformat(end) if end else now)-datetime.fromisoformat(start)).total_seconds())
        e['running_seconds']=seconds;e['estimated_usd']=seconds*RATE/3600
        total+=e['estimated_usd']
    ledger['estimated_total_usd']=total
    return total


def poll(root,logs=True):
    path=root/'jobs.json';ledger=json.loads(path.read_text());api=HfApi();refresh(api,ledger);total=cost(ledger)
    for key,e in ledger['jobs'].items():
        if e['status'] in TERMINAL and e['status']!='COMPLETED':
            message(root,'question','T2b paid job blocked',f"{key} job {e['id']} ended {e['status']}; {e.get('status_message')}. Cost so far USD {total:.4f}; all jobs bounded below USD 65. No retry launched. Please advise recovery under the existing cap.",f'blocked-{key}-send.json')
        if e.get('running_seconds',0)>570*60 and e['stage']=='train' and e['status']=='RUNNING':
            message(root,'question','T2b training approaching timeout',f"{key} job {e['id']} is past 9.5 hours with a hard 630-minute timeout. Cost so far USD {total:.4f}; four paid timeout ceilings USD 61.4167. No extension or retry authorized. Please advise if recovery is needed.",f'near-timeout-{key}-send.json')
        if logs and e['status']=='RUNNING':
            raw='\n'.join(api.fetch_job_logs(job_id=e['id'],tail=100))
            token=os.environ.get('HF_TOKEN','')
            if token:raw=raw.replace(token,'[REDACTED]')
            (root/f'{key}.log').write_text(raw)
            matches=re.findall(r'(\d+)\s*/\s*34345',raw)
            if matches:e['observed_optimizer_step']=max(e.get('observed_optimizer_step',0),max(map(int,matches)))
            # Complete Hub markers prove checkpoint persistence before reporting past it.
            if e['stage']=='train' and e.get('observed_optimizer_step',0)>500:
                files=api.list_repo_files(f"frontier-infra/jebadiah-9b-t2b-s{e['seed']}-checkpoints")
                steps=[int(m.group(1)) for f in files if (m:=re.fullmatch(r'checkpoints/checkpoint-(\d+)/complete.json',f))]
                e['latest_complete_checkpoint']=max(steps,default=0)
    save(path,ledger)
    seeds=[ledger['jobs'][f'train-s{s}'] for s in (17,18)]
    if all(e['status']=='RUNNING' and e.get('observed_optimizer_step',0)>500 and e.get('latest_complete_checkpoint',0)>=500 for e in seeds):
        desc='; '.join(f"seed {e['seed']}: job {e['id']}, step {e['observed_optimizer_step']}, complete checkpoint {e['latest_complete_checkpoint']}, timeout 630 min" for e in seeds)
        message(root,'status','T2b both seeds past first checkpoint',desc+f". Estimated compute so far USD {total:.4f} at USD 2.75/h, local smoke USD 0. Four-job timeout ceiling USD 61.4167 under hard USD 65. Data revision bc416bc8485780d73aa4c22fb4cd772232faf69e. No paid failures or retries.",'checkpoint-send.json')
    print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'cost_usd':round(total,4),'jobs':{k:{f:e.get(f) for f in ('id','status','observed_optimizer_step','latest_complete_checkpoint')} for k,e in ledger['jobs'].items()}}),flush=True)
    return ledger


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();poll(a.root.resolve())
