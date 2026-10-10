# SPDX-License-Identifier: Apache-2.0
"""Offline intent diagnostic reports, using item48's timing-only pass wrapper.

No model loading, inference, training, downloads or external writes here.
Future authorized evaluation calls save_completed_pass after each run_set pass.
"""
from collections import defaultdict
import copy
import json
from pathlib import Path


def summarize(rows, records):
    lookup = {r['id']:r for r in records}
    if len(lookup) != len(records): raise ValueError('Duplicate diagnostic ID')
    expected = {(r['id'],qid) for r in records for qid in r['questions']}
    seen=set();groups=defaultdict(list);choice_intents=defaultdict(list)
    tp=fp=fn=tn=0
    for row in rows:
        key=(row['id'],row['qid'])
        if key not in expected or key in seen or row['repeat'] != 0:
            raise ValueError('Duplicate, unknown or repeated diagnostic answer')
        seen.add(key);r=lookup[row['id']];q=r['questions'][row['qid']]
        gold=str(r['label'][row['qid']]).lower() if q['type']=='noul' else r['label'][row['qid']]
        options=q['criteria'] if q['type']=='choice' else ('true','false')
        if row['label'] != gold or row['pick'] not in options:
            raise ValueError('Result disagrees with diagnostic wire contract')
        ok=row['pick']==gold;p=r['provenance']
        for group in ('overall','type:'+q['type'],'intent:'+p['intent'],'scenario:'+p['scenario']):
            groups[group].append(ok)
        if q['type']=='choice':choice_intents[p['intent']].append(ok)
        else:
            tp+=row['pick']=='true' and gold=='true';fp+=row['pick']=='true' and gold=='false'
            fn+=row['pick']=='false' and gold=='true';tn+=row['pick']=='false' and gold=='false'
    if seen != expected: raise ValueError('Incomplete diagnostic pass')
    recall=tp/(tp+fn) if tp+fn else None
    specificity=tn/(tn+fp) if tn+fp else None
    return {'accuracy':{k:{'n':len(v),'correct':sum(v),'accuracy':sum(v)/len(v)}
                        for k,v in sorted(groups.items())},
            'primary_metric':'choice macro intent accuracy',
            'choice_macro_intent_accuracy':sum(sum(v)/len(v) for v in choice_intents.values())/len(choice_intents) if choice_intents else None,
            'verification':{'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                            'precision':tp/(tp+fp) if tp+fp else None,'recall':recall,
                            'balanced_accuracy':(recall+specificity)/2 if recall is not None and specificity is not None else None},
            'OOS_evaluated':False,'rows':rows}


def save_completed_pass(path, report, name, rows, extra, records):
    """Discard run_set's tuple-keyed question map and persist each finished pass."""
    if name in report: raise ValueError('Pass already saved')
    timing=extra['timing']
    if timing['questions_scored'] != len(rows) or timing['truncated_prompts'] != 0:
        raise ValueError('Incomplete or truncated diagnostic pass')
    updated=copy.deepcopy(report)
    updated[name]=summarize(rows,records)
    updated[name]['run_metrics']=timing
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('item54-'+path.name+'.tmp')
    temporary.write_text(json.dumps(updated,indent=2)+'\n');temporary.replace(path)
    report.clear();report.update(updated)
    return report
