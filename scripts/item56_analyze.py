# SPDX-License-Identifier: Apache-2.0
"""Official scoring and paired complete-group bootstrap of two seed means."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import gzip
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROXY=ROOT/'eval/decision-index/proxy-0.3-10pct'
sys.path.insert(0,str(PROXY))
from bootstrap import official_index,group_strata,resample,quantile
_WORK=None


def initialize(strata,results,seed):
    global _WORK
    _WORK=(strata,results,seed)


def draw(i):
    strata,results,seed=_WORK
    sampled=resample(strata,random.Random(seed+i))
    scores={name:official_index(sampled,value) for name,value in results.items()}
    return mean_difference(scores)


def mean_difference(scores):
    baseline=sum(scores[n][0] for n in ('a3-s17','a3-s18'))/2
    candidate=sum(scores[n][0] for n in ('probe-s17','probe-s18'))/2
    areas={a:sum(scores[n][1][a] for n in ('probe-s17','probe-s18'))/2-
             sum(scores[n][1][a] for n in ('a3-s17','a3-s18'))/2 for a in scores['a3-s17'][1]}
    return candidate-baseline,areas


def main():
    from decision_index.suite.io import Suite
    from decision_index.scoring.report import load_results
    from sample import read_ids
    from score import load_proxy,score
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    for name in ('a3-s17','a3-s18','probe-s17','probe-s18'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--replicates',type=int,default=2000)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((PROXY/'manifest.json').read_text());ids=read_ids(PROXY/'run-ids.txt',manifest['run_ids_sha256'])
    suite=Suite(a.suite,'0.3');suite.verify();proxy=load_proxy(suite,ids)
    expected={r['_evaluation']['run_id']:r['_evaluation']['payload_sha256'] for r in proxy.selected}
    results={};indices={};native={};inputs={}
    for name in ('a3-s17','a3-s18','probe-s17','probe-s18'):
        path=getattr(a,name.replace('-','_'))
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt') as stream:
            raw=[json.loads(line) for line in stream if line.strip()]
        if len(raw)!=len(ids) or len({r['run_id'] for r in raw})!=len(raw):
            raise ValueError('Duplicate or missing raw proxy records: '+name)
        result=load_results(path)
        if set(result)!=ids or any(r['status']!='ok' or r.get('payload_sha256')!=expected[r['run_id']] for r in raw):
            raise ValueError('Incomplete or unbound frozen proxy: '+name)
        results[name]=result;inputs[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        score(proxy,path,a.out/name,'item56-'+name)
        indices[name]=json.loads((a.out/name/'index.json').read_text())
        native[name]=json.loads((a.out/name/'benchmark-summary.json').read_text())
    points={n:official_index(proxy.selected,r) for n,r in results.items()}
    delta,areas=mean_difference(points)
    replicate_file=a.out/'item56-bootstrap-draws.json'
    binding={'inputs':inputs,'replicates':a.replicates,'seed':20261008,'manifest_sha256':manifest['run_ids_sha256']}
    draws=[]
    if replicate_file.exists():
        saved=json.loads(replicate_file.read_text())
        if saved['binding']!=binding:raise ValueError('Existing bootstrap draws from different inputs')
        draws=saved['draws']
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork'),
                             initializer=initialize,initargs=(group_strata(proxy.selected),results,20261008)) as pool:
        for difference,area_diff in pool.map(draw,range(len(draws),a.replicates),chunksize=5):
            draws.append([difference,area_diff])
            if len(draws)%25==0 or len(draws)==a.replicates:
                temp=replicate_file.with_suffix('.tmp');temp.write_text(json.dumps({'binding':binding,'draws':draws})+'\n');temp.replace(replicate_file)
                print('ITEM56_BOOTSTRAP',len(draws),round(time.monotonic()-start,1),flush=True)
    ci=[quantile([d[0] for d in draws],q) for q in (.025,.975)]
    summary={'proxy':True,'private':True,'probe':True,'binding':binding,
             'method':'Paired percentile complete catalog/group draws within benchmark/domain/track strata; unchanged official scorer, mean of each two seeds on identical draws',
             'training_variance_ci':False,'baseline_mean':sum(points[n][0] for n in ('a3-s17','a3-s18'))/2,
             'candidate_mean':sum(points[n][0] for n in ('probe-s17','probe-s18'))/2,
             'difference':delta,'difference_ci95':ci,'per_seed':{n:{'headline':v[0],'areas':v[1]} for n,v in points.items()},
             'areas':{area:{'difference':v,'difference_ci95':[quantile([d[1][area] for d in draws],q) for q in (.025,.975)]} for area,v in areas.items()}}
    spec=json.loads((Path(__import__('decision_index').__file__).parent/'data/index-0.3.json').read_text())
    benchmarks={}
    for catalog,entry in spec['chance'].items():
        if any(term in entry.get('name','').lower() for term in ('clinc','cladder','anli','gsm8k','hover','bbh','big-bench hard')):
            benchmarks[entry['name']]={}
            for name,index in indices.items():
                record=next(r for r in native[name]['benchmarks'] if r['catalog_id']==int(catalog))
                benchmarks[entry['name']][name]={'raw':record['score'],'metric':record['metric'],
                                               'skill':index['benchmarks'][catalog]['skill'],'coverage':index['benchmarks'][catalog]['coverage']}
    summary['benchmarks']=benchmarks
    clinc=next(value for name,value in benchmarks.items() if 'clinc' in name.lower())
    loss=100*(sum(clinc[n]['raw'] for n in ('a3-s17','a3-s18'))/2-sum(clinc[n]['raw'] for n in ('probe-s17','probe-s18'))/2)
    summary['clinc_mean_macro_f1_loss_points']=loss
    summary['clinc_matched_seed_macro_f1_loss_points']={str(seed):100*(clinc['a3-s'+str(seed)]['raw']-clinc['probe-s'+str(seed)]['raw']) for seed in (17,18)}
    summary['clinc_single_seed_loss_over_five_points']={seed:value>5 for seed,value in summary['clinc_matched_seed_macro_f1_loss_points'].items()}
    summary['clinc_guard_passed']=loss<=2.5
    summary['probe_wins']=delta>0 and ci[0]>0 and loss<=2.5
    summary['decision']='split into single rungs for confirmation' if summary['probe_wins'] else 'drop probe'
    summary['no_model_promotion']=True
    (a.out/'item56-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
