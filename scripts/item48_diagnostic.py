"""Paired held-out evidence diagnostic with pinned A3 temperatures and full menus."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
import eval_jebadiah as evaluate
from jebadiah_model import Scorer, load_base, load_tokenizer, read_temperatures
DATA = 'frontier-infra/jebadiah-data-v2-1-item47'
DATA_REV = '7b6b15f95cc9902e1ebd783656a3bd590350875d'
A3 = 'frontier-infra/jebadiah-9b-v2-1-a3'
A3_REV = '9e69926a007dd636e82d33485dcd48f9751c4248'
REPO = 'frontier-infra/jebadiah-9b-v2-1-rung1'


def summarize(rows, records):
    lookup = {r['id']: r for r in records}
    groups = defaultdict(list)
    flips = defaultdict(list)
    tp = fp = fn = 0
    for row in rows:
        r = lookup[row['id']]
        q = r['questions'][row['qid']]
        ok = row['pick'] == row['label']
        for name in ('overall', 'type:' + q['type'], 'rule:' + r['rule'],
                     'documents:' + str(len(r['world']['documents'])),
                     'polarity:' + str(r['world']['query']['negated']), 'status:' + r['evidence_status']):
            groups[name].append(ok)
        flips[(r['group_id'], q['type'])].append(ok)
        if q['type'] == 'noul':
            tp += row['pick'] == 'true' and row['label'] == 'true'
            fp += row['pick'] == 'true' and row['label'] != 'true'
            fn += row['pick'] != 'true' and row['label'] == 'true'
    accuracy = {k: {'n':len(v), 'correct':sum(v), 'accuracy':sum(v)/len(v)} for k,v in groups.items()}
    complete = {}
    for kind in ('choice', 'noul'):
        values = [all(v) for (_,t),v in flips.items() if t == kind]
        assert len(values) == 50 and all(len(v)==3 for (_,t),v in flips.items() if t==kind)
        complete[kind] = {'n':len(values),'correct':sum(values),'accuracy':sum(values)/len(values)}
    all_groups = defaultdict(list)
    for (group,_),values in flips.items(): all_groups[group].extend(values)
    complete['both_types'] = {'n':len(all_groups),'correct':sum(all(v) for v in all_groups.values()),
                             'accuracy':sum(all(v) for v in all_groups.values())/len(all_groups)}
    return {'accuracy':accuracy, 'macro_rule_accuracy':sum(v['accuracy'] for k,v in accuracy.items() if k.startswith('rule:'))/5,
            'boolean_support': {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn)},
            'complete_minimal_variant_groups':complete, 'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate_revision')
    a=p.parse_args()
    api=HfApi(); assert api.model_info(REPO).private
    source=hf_hub_download(DATA,'rung1/evidence-diagnostic.jsonl',repo_type='dataset',revision=DATA_REV)
    records=[json.loads(x) for x in Path(source).read_text().splitlines() if x.strip()]
    assert len(records)==300
    temperatures=read_temperatures(Path(hf_hub_download(A3,'temperatures.json',revision=A3_REV)).parent)
    result={'data_revision':DATA_REV,'candidate_revision':a.candidate_revision,'a3_revision':A3_REV,
            'temperatures':temperatures,'scoring':'eval_jebadiah.run_set; full menus; one repeat; no temperature fitting'}
    for name,repo,revision in [('a3',A3,A3_REV),('rung1',REPO,a.candidate_revision)]:
        path=snapshot_download(repo,revision=revision,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model'])
        assert read_temperatures(path)==temperatures
        scorer=Scorer(load_base(path),load_tokenizer(path),2048,temperatures=temperatures)
        rows,extra=evaluate.run_set(scorer,records,1,False,8,0)
        result[name]=summarize(rows,records); result[name]['run_metrics']=extra
        del scorer; gc.collect(); torch.cuda.empty_cache()
    result['macro_accuracy_difference']=result['rung1']['macro_rule_accuracy']-result['a3']['macro_rule_accuracy']
    result['diagnostic_improves']=result['macro_accuracy_difference']>0
    output=Path('/workspace/item48-evidence-diagnostic.json')
    output.write_text(json.dumps(result,indent=2)+'\n')
    api.upload_file(repo_id=REPO+'-checkpoints',path_or_fileobj=output,path_in_repo=output.name)
    print('ITEM48_DIAGNOSTIC',json.dumps({k:v for k,v in result.items() if k not in ('a3','rung1')}),flush=True)


if __name__=='__main__': main()
