# SPDX-License-Identifier: Apache-2.0
"""Collect private training/proxy evidence and verify the original A3 contract."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from t2b_jobs import save


def collect_train(root,seed):
    ledger=json.loads((root/'jobs.json').read_text());e=ledger['jobs'][f'train-s{seed}'];assert e['status']=='COMPLETED'
    api=HfApi();model=f'frontier-infra/jebadiah-9b-t2b-s{seed}';cp=model+'-checkpoints'
    mi=api.model_info(model);ci=api.model_info(cp);assert mi.private and ci.private
    out=root/'artifacts'/f's{seed}';out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(snapshot_download(cp,revision=ci.sha,allow_patterns='t2b-initial/*.json',local_dir=out/'checkpoints'))/'t2b-initial'
    served=Path(snapshot_download(model,revision=mi.sha,allow_patterns=['training_provenance.json','merge_verification.json','temperatures.json','prompt_contract.json'],local_dir=out/'served'))
    summary=json.loads((checkpoint/'runs/9b/train_summary.json').read_text())
    old=json.loads((root/'repo/results/runs/item52/train_summary.json').read_text());cfg=summary['config']
    ignored={'base_model','dataset_path','eval_dataset_path','output_dir','dataset_sha256','eval_dataset_sha256','checkpoint_repo','seed'}
    assert {k:v for k,v in cfg.items() if k not in ignored}=={k:v for k,v in old['config'].items() if k not in ignored},'A3 recipe drift'
    assert cfg['seed']==seed
    manifest=json.loads((root/'build/manifest.json').read_text())
    assert cfg['dataset_sha256']==manifest['files']['train.jsonl']['sha256']
    assert cfg['eval_dataset_sha256']==manifest['files']['calib.jsonl']['sha256']
    assert summary['train_examples']==274758 and summary['calib_examples']==356 and summary['steps']==34345
    for key in ('target_modules','lora_modules','trainable_params','prompt_source_commit','prompt_source_sha256','chat_template_sha256','single_token_labels','chat_template_kwargs','fp32_candidate_logits','score_targets','score_ordinal_adjacent'):
        assert summary[key]==old[key],key
    provenance=json.loads((served/'training_provenance.json').read_text())
    assert provenance['base_revision']=='c202236235762e1c871ad0ccb60c8ee5ba337b9a' and provenance['effective_batch_size']==8
    assert provenance['dataset_revision']==e['dataset_revision']
    gate=json.loads((served/'merge_verification.json').read_text())
    assert gate['questions']==260 and gate['max_delta_allowed']==0.05 and gate['max_probability_delta']<=0.05 and gate['clear_margin_flips']==0
    heldout=json.loads((checkpoint/'t2b-heldout.json').read_text())
    assert heldout['seed']==seed and heldout['questions']==3000 and heldout['truncated_prompts']==0
    assert heldout['dataset_revision']==provenance['dataset_revision'] and heldout['heldout_sha256']==manifest['heldout_sha256']
    families={k:v for k,v in heldout['metrics'].items() if k!='overall'}
    assert len(families)==5 and all(v['n']==600 for v in families.values())
    assert len(heldout['answers'])==len({(v['id'],v['qid']) for v in heldout['answers']})==3000
    assert sum(v['correct'] for v in families.values())==sum(v['correct'] for v in heldout['answers'])
    proof={'seed':seed,'model_revision':mi.sha,'checkpoint_revision':ci.sha,'recipe_matches_A3':True,'steps':summary['steps'],'train_examples':summary['train_examples'],'calib_examples':summary['calib_examples'],'truncated_training_prompts':summary['truncated_prompts'],'merge_gate':gate,'heldout_family_accuracy':heldout['family_macro_accuracy'],'heldout_per_family':families,'dataset_revision':provenance['dataset_revision']}
    save(out/'training-proof.json',proof);e.update(model_revision=mi.sha,checkpoint_revision=ci.sha,training_proof=str(out/'training-proof.json'));save(root/'jobs.json',ledger)
    print(json.dumps(proof,indent=2))


def collect_proxy(root,seed):
    ledger=json.loads((root/'jobs.json').read_text());e=ledger['jobs'][f'proxy-s{seed}'];assert e['status']=='COMPLETED'
    revision=ledger['jobs'][f'train-s{seed}']['model_revision'];run=f'jebadiah-9b-v2-1-t2b-s{seed}-{revision[:8]}-proxy-0.3-10pct'
    repo='jbrashear/jebadiah-9b-v2-1-index-results';api=HfApi();info=api.dataset_info(repo);assert info.private
    prefix='runs/'+run+'/';files=[f for f in api.list_repo_files(repo,repo_type='dataset',revision=info.sha) if f.startswith(prefix) and f.endswith(('.json','.jsonl.gz'))]
    assert prefix+'results.jsonl.gz' in files
    out=root/'artifacts'/f's{seed}'/'proxy';out.mkdir(parents=True,exist_ok=True);hashes={}
    for f in files:
        path=Path(hf_hub_download(repo,f,repo_type='dataset',revision=info.sha));dest=out/f[len(prefix):];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());hashes[f[len(prefix):]]=hashlib.sha256(dest.read_bytes()).hexdigest()
    save(out/'receipt.json',{'run_name':run,'results_revision':info.sha,'private':True,'files':hashes});e.update(run_name=run,results_revision=info.sha,raw_proxy_path=str(out/'results.jsonl.gz'));save(root/'jobs.json',ledger)
    print(json.dumps({'seed':seed,'run_name':run,'results_revision':info.sha}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,choices=[17,18],required=True);p.add_argument('stage',choices=['train','proxy']);a=p.parse_args()
    (collect_train if a.stage=='train' else collect_proxy)(a.root.resolve(),a.seed)
