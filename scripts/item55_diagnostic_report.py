# SPDX-License-Identifier: Apache-2.0
"""Offline arithmetic report; item48 timing-only, completed-pass persistence."""
from collections import defaultdict
import copy
import json
from pathlib import Path


def summarize(rows, records):
    lookup = {r['id']:r for r in records}
    if len(lookup) != len(records): raise ValueError('Duplicate diagnostic ID')
    expected = {(r['id'],qid) for r in records for qid in r['questions']}
    seen=set();groups=defaultdict(list);modules=defaultdict(list)
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
        ok=row['pick']==gold;module=r['provenance']['module']
        for group in ('overall','type:'+q['type'],'module:'+module,'structure:'+r['family_id']):
            groups[group].append(ok)
        if q['type']=='choice':modules[module].append(ok)
        else:
            tp+=row['pick']=='true' and gold=='true';fp+=row['pick']=='true' and gold=='false'
            fn+=row['pick']=='false' and gold=='true';tn+=row['pick']=='false' and gold=='false'
    if seen != expected: raise ValueError('Incomplete diagnostic pass')
    recall=tp/(tp+fn) if tp+fn else None
    specificity=tn/(tn+fp) if tn+fp else None
    return {'accuracy':{k:{'n':len(v),'correct':sum(v),'accuracy':sum(v)/len(v)} for k,v in sorted(groups.items())},
            'primary_metric':'choice macro module accuracy',
            'choice_macro_module_accuracy':sum(sum(v)/len(v) for v in modules.values())/len(modules) if modules else None,
            'verification':{'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else None,
                            'recall':recall,'balanced_accuracy':(recall+specificity)/2 if recall is not None and specificity is not None else None},
            'rows':rows}


def save_completed_pass(path, report, name, rows, extra, records):
    """Save each finished pass, excluding run_set's tuple-keyed question map."""
    if name in report: raise ValueError('Pass already saved')
    timing=extra['timing']
    if timing['questions_scored'] != len(rows) or timing['truncated_prompts'] != 0:
        raise ValueError('Incomplete or truncated diagnostic pass')
    updated=copy.deepcopy(report);updated[name]=summarize(rows,records)
    updated[name]['run_metrics']=timing
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('item55-'+path.name+'.tmp')
    temporary.write_text(json.dumps(updated,indent=2)+'\n');temporary.replace(path)
    report.clear();report.update(updated);return report
