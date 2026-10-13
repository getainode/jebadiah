# SPDX-License-Identifier: Apache-2.0
"""One-attempt HF submissions with a durable aggregate reservation ledger."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from huggingface_hub import HfApi
from t2b_hf_client import configure_hf_client
configure_hf_client()
from t2b_launch import command, CAP_MINUTES, RATE, TOTAL_MAX_USD
HELDOUT_SHA256 = '000451d29bb99b67b9b3925fb1cef5859f723cc6f5fd7c01850ec15a59fc472b'

CODE = '8f5fd57d35ba51119fae8787d8875f666645f492'
TERMINAL = {'COMPLETED','ERROR','CANCELED','CANCELLED','TIMEOUT','STOPPED'}


def save(path, value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)


def refresh(api, ledger):
    for entry in ledger['jobs'].values():
        if not entry.get('id'): continue
        info=api.inspect_job(job_id=entry['id'])
        entry['status']=info.status.stage
        # Never serialize JobInfo: it includes secret environment fields.
        entry['status_message']=info.status.message
        entry['hf_durations']=asdict(info.durations)
        entry['timestamps']={k:str(v) if v is not None else None for k,v in asdict(info).items() if k in ('created_at','started_at','finished_at')}
        entry['checked_at']=datetime.now(timezone.utc).isoformat()
    return ledger


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    p.add_argument('stage',choices=['train','proxy','inspect']);p.add_argument('--seed',type=int,choices=[17,18]);p.add_argument('--model-revision')
    a=p.parse_args();root=a.root.resolve();path=root/'jobs.json'
    api=HfApi()
    ledger=json.loads(path.read_text()) if path.exists() else {'cap_usd':65,'rate_usd_hour':RATE,'smoke_usd':0,'smoke':'Local original pipeline setup-only; no paid GPU','jobs':{}}
    if a.stage=='inspect':
        refresh(api,ledger); save(path,ledger);print(json.dumps(ledger,indent=2));return
    assert a.seed is not None
    key=f'{a.stage}-s{a.seed}'; assert key not in ledger['jobs'],'Submission already reserved; reconcile, never automatically retry'
    data=json.loads((root/'dataset-upload.json').read_text()); assert api.dataset_info(data['dataset'],revision=data['revision']).private
    assert json.loads((root/'smoke/receipt.json').read_text())['status']=='passed'
    model=f'frontier-infra/jebadiah-9b-t2b-s{a.seed}'
    assert api.model_info(model).private and api.model_info(model+'-checkpoints').private
    revision=a.model_revision or CODE
    if a.stage=='proxy':
        assert a.model_revision and api.model_info(model,revision=revision).sha==revision
        assert all(ledger['jobs'][f'train-s{s}']['status']=='COMPLETED' for s in (17,18))
        assert api.file_exists(model,'merge_verification.json',revision=revision)
        assert api.file_exists(model+'-checkpoints','t2b-initial/t2b-heldout.json')
    reserve=CAP_MINUTES[a.stage]*RATE/60
    # Count terminal failures at their full ceilings, and reserve every remaining planned stage.
    future=sum(CAP_MINUTES[stage]*RATE/60 for stage in ('train','proxy') for seed in (17,18) if f'{stage}-s{seed}' not in ledger['jobs'] and f'{stage}-s{seed}'!=key)
    assert ledger['smoke_usd']+sum(e['allocation_ceiling_usd'] for e in ledger['jobs'].values())+reserve+future<65
    entry={'stage':a.stage,'seed':a.seed,'timeout_minutes':CAP_MINUTES[a.stage],'allocation_ceiling_usd':reserve,'status':'SUBMISSION_RESERVED','attempts':1,'reserved_at':datetime.now(timezone.utc).isoformat(),'dataset_revision':data['revision'],'code_base_revision':CODE,'model_revision':a.model_revision}
    ledger['jobs'][key]=entry; save(path,ledger)
    cmd=command(a.stage,revision,CODE,a.seed,data['revision'],HELDOUT_SHA256);i=cmd.index('--')
    info=api.run_job(image=cmd[i-1],command=cmd[i+1:],flavor='rtx-pro-6000',timeout=f'{CAP_MINUTES[a.stage]}m',secrets={'HF_TOKEN':os.environ['HF_TOKEN']},name=f't2b-{a.stage}-s{a.seed}',labels={'task':'t2b-train','seed':str(a.seed),'stage':a.stage})
    entry.update(id=info.id,url=info.url,status=info.status.stage,created_at=str(info.created_at));save(path,ledger)
    print(json.dumps({k:entry[k] for k in ('id','stage','seed','status','timeout_minutes','allocation_ceiling_usd')}))


if __name__=='__main__':main()
