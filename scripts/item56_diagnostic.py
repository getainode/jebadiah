# SPDX-License-Identifier: Apache-2.0
"""Four-source independent diagnostics, persisting each completed timing-only pass."""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import sys


def summarize(rows,records):
    expected={(r['id'],qid) for r in records for qid in r['questions']}
    lookup={r['id']:r for r in records};seen=set();groups=defaultdict(list)
    for answer in rows:
        key=(answer['id'],answer['qid'])
        if key not in expected or key in seen or answer['repeat']!=0:raise ValueError('Invalid diagnostic coverage')
        seen.add(key);record=lookup[answer['id']];qid=answer['qid'];question=record['questions'][qid]
        gold=str(record['label'][qid]).lower() if question['type']=='noul' else record['label'][qid]
        if answer['label']!=gold:raise ValueError('Diagnostic gold mismatch')
        correct=answer['pick']==gold
        groups['overall'].append(correct);groups['type:'+question['type']].append(correct)
        groups['label:'+str(gold)].append(correct)
        for field in ('module','module_family','difficulty','split','pattern_id'):
            if field in record.get('provenance',{}):groups[field+':'+str(record['provenance'][field])].append(correct)
    if seen!=expected:raise ValueError('Incomplete diagnostic')
    result={'metrics':{k:{'n':len(v),'correct':sum(v),'accuracy':sum(v)/len(v)} for k,v in sorted(groups.items())},'rows':rows}
    result['macro_label_accuracy']=sum(v['accuracy'] for k,v in result['metrics'].items() if k.startswith('label:'))/sum(k.startswith('label:') for k in groups)
    return result


def main():
    import torch
    from huggingface_hub import HfApi,snapshot_download
    sys.path.insert(0,str(Path.cwd()/'train'))
    from jebadiah_model import Scorer,load_base,load_tokenizer,read_temperatures
    import eval_jebadiah as evaluate
    from item51_diagnostic import summarize as causal_summary
    from item54_diagnostic_report import summarize as intent_summary
    from item55_diagnostic_report import summarize as math_summary
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());api=HfApi();dest='frontier-infra/jebadiah-9b-v2-1-probe-b-s17-checkpoints'
    assert api.model_info(dest).private
    data=Path(snapshot_download('frontier-infra/jebadiah-data-v2-1-item47',repo_type='dataset',revision=cfg['data_revision'],allow_patterns='probe-b/*'))/'probe-b'
    manifest=json.loads((data/'manifest.json').read_text())
    records={}
    for name,pin in cfg['sources'].items():
        file=data/pin['diagnostic']
        import hashlib
        assert hashlib.sha256(file.read_bytes()).hexdigest()==manifest['files'][file.name]['sha256']
        records[name]=[json.loads(line) for line in file.read_text().splitlines() if line.strip()]
    report={'data_revision':cfg['data_revision'],'model_revisions':cfg['models'],'temperature_policy':'Per-model original A3 fitting recipe; held-out diagnostics never fitted'}
    path=Path('/workspace/item56-diagnostics.json')
    for name,model in cfg['models'].items():
        assert api.model_info(model['repo'],revision=model['revision']).private
        folder=snapshot_download(model['repo'],revision=model['revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model'])
        scorer=Scorer(load_base(folder),load_tokenizer(folder),2048,temperatures=read_temperatures(folder))
        for source,rr in records.items():
            answers,extra=evaluate.run_set(scorer,rr,1,False,8,0)
            timing=extra['timing']
            if timing['questions_scored']!=sum(len(r['questions']) for r in rr) or timing['truncated_prompts']!=0:
                raise ValueError('Incomplete or truncated diagnostic')
            summarizer=causal_summary if source=='rung2b' else intent_summary if source=='rung4-add' else math_summary if source=='rung5-add' else summarize
            result=summarizer(answers,rr);result['run_metrics']=timing
            report.setdefault(name,{})[source]=result
            temp=path.with_name('item56-diagnostics.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(path)
            api.upload_file(repo_id=dest,path_or_fileobj=path,path_in_repo='item56-diagnostics.json',commit_message='Persist each complete independent diagnostic pass')
            print('ITEM56_DIAGNOSTIC_PASS',name,source,json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
        del scorer;gc.collect();torch.cuda.empty_cache()
        # Jobs are ephemeral; remove each downloaded served model before loading the next.
        import shutil
        cache=Path(folder).parent.parent
        shutil.rmtree(cache)


if __name__=='__main__':main()
