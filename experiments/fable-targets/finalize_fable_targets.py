#!/usr/bin/env python3
"""After fable_targets.py: (1) write data-v2-fable/synth-<family>.jsonl with Fable 5.1 targets (label = argmax,
the v2 teacher target kept as target_v2_teachers); (2) COMPARE.md, Fable vs v2 teachers per family and type;
(3) two v3 dose pools in the training package, built from the SAME states as data-dose-c20:
    data-v3-c20f       c20's records with the synthetic choice targets swapped to Fable's (choice only)
    data-v3-c20f-n3k   the same plus the noul questions of those states, Fable-labelled, capped at 3,000
Then split_pool.py and lint_data.py --train on both. No em dashes."""
import glob, hashlib, json, os, collections, statistics, subprocess, sys
D = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get('JEB_ROOT', '/workspace/jeb')  # holds data-dose-c20/ and receives data-v3-*/
DATA = os.path.join(D, '..', '..', 'data')  # split_pool.py, lint_data.py
POOL_V2 = os.environ.get('JEB_POOL_V2', os.path.join(D, '..', '..', 'data', 'pool-v2'))
OUT = f'{D}/data-v2-fable'; os.makedirs(OUT, exist_ok=True)
fable = {json.loads(l)['id']: json.loads(l)['targets'] for l in open(f'{D}/answers.jsonl')}
stat = collections.defaultdict(lambda: {'n': 0, 'agree': 0, 'dp': []}); n_written = 0; missing = 0
for f in sorted(glob.glob(f'{POOL_V2}/data-v2/synth-*.jsonl')):
    fam = os.path.basename(f)[6:-6]; rows = []
    for l in open(f):
        r = json.loads(l)
        if r['id'] not in fable: missing += 1; continue
        r['target_v2_teachers'] = r['target']; r['target'] = fable[r['id']]
        r['label'] = {qid: (max(t, key=t.get) if r['questions'][qid]['type'] == 'choice' else (t['true'] >= 0.5)) for qid, t in r['target'].items()}
        r['license'] = 'Apache-2.0 (AINode; synthetic states by DeepSeek V4 Flash; gold = Fable 5.1 verbalized probabilities, 2026-09-24, one call per state via the Claude Code harness)'
        for qid, q in r['questions'].items():
            t2 = r['target_v2_teachers'][qid]; t1 = r['target'][qid]; k = (fam, q['type']); s = stat[k]; s['n'] += 1
            s['agree'] += int(max(t1, key=t1.get) == max(t2, key=t2.get))
            if q['type'] == 'noul': s['dp'].append(abs(t1['true'] - t2['true']))
        rows.append(r)
    with open(f'{OUT}/synth-{fam}.jsonl', 'w') as g:
        for r in rows: g.write(json.dumps(r, ensure_ascii=False) + '\n')
    n_written += len(rows)
lines = ['# Fable 5.1 vs the v2 teachers on the synthetic pool', '', f'{n_written:,} states re-labelled ({missing} without a Fable answer).', '',
         '| family | type | questions | same answer | mean abs diff in p(yes) |', '|---|---|---|---|---|']
tot = collections.defaultdict(lambda: [0, 0])
for (fam, t), s in sorted(stat.items()):
    tot[t][0] += s['n']; tot[t][1] += s['agree']
    lines.append(f"| {fam} | {t} | {s['n']:,} | {s['agree']/s['n']*100:.1f}% | {statistics.mean(s['dp']):.3f} |" if s['dp'] else f"| {fam} | {t} | {s['n']:,} | {s['agree']/s['n']*100:.1f}% | |")
lines += ['', '| type | questions | same answer |', '|---|---|---|'] + [f'| {t} | {n:,} | {a/n*100:.1f}% |' for t, (n, a) in tot.items()]
open(f'{D}/COMPARE.md', 'w').write('\n'.join(lines)); print('\n'.join(lines[:4] + lines[-4:]))
# v3 pools from the same states as c20
fab_rec = {}
for f in glob.glob(f'{OUT}/synth-*.jsonl'):
    for l in open(f): r = json.loads(l); fab_rec[r['id']] = r
def hk(s): return hashlib.sha256(('v3noul:' + s).encode()).hexdigest()
c20 = [json.loads(l) for l in open(f'{PKG}/data-dose-c20/pool.jsonl')]
os.makedirs(f'{PKG}/data-v3-c20f', exist_ok=True); os.makedirs(f'{PKG}/data-v3-c20f-n3k', exist_ok=True)
a = open(f'{PKG}/data-v3-c20f/pool.jsonl', 'w'); b = open(f'{PKG}/data-v3-c20f-n3k/pool.jsonl', 'w')
syn_ids = [r['id'] for r in c20 if r['set'].startswith('synth') and r['id'] in fab_rec]
budget = 3000; noul_added = 0; swapped = 0
for r in c20:
    if r['set'].startswith('synth') and r['id'] in fab_rec:
        fr = fab_rec[r['id']]; keep = list(r['questions'].keys())  # choice-only question set of c20
        r1 = dict(r); r1['target'] = {q: fr['target'][q] for q in keep}; r1['label'] = {q: fr['label'][q] for q in keep}
        r1['target_v2_teachers'] = {q: r['target'][q] for q in keep}; r1['license'] = fr['license']; swapped += 1
        a.write(json.dumps(r1, ensure_ascii=False) + '\n')
        r2 = dict(r1)
        nouls = [q for q, qq in fr['questions'].items() if qq['type'] == 'noul']
        if nouls and noul_added < budget and int(hk(r['id'])[:8], 16) % 2 == 0:  # about half the states carry their nouls
            take = nouls[:max(0, budget - noul_added)]; noul_added += len(take)
            r2['questions'] = dict(r1['questions'], **{q: fr['questions'][q] for q in take})
            r2['target'] = dict(r1['target'], **{q: fr['target'][q] for q in take}); r2['label'] = dict(r1['label'], **{q: fr['label'][q] for q in take})
        b.write(json.dumps(r2, ensure_ascii=False) + '\n')
    else:
        a.write(json.dumps(r, ensure_ascii=False) + '\n'); b.write(json.dumps(r, ensure_ascii=False) + '\n')
a.close(); b.close(); print(f'v3 pools: {swapped} synthetic records swapped to Fable targets; {noul_added} Fable-labelled noul questions added in c20f-n3k')
for d in ('data-v3-c20f', 'data-v3-c20f-n3k'):
    subprocess.run([sys.executable, os.path.join(DATA, 'split_pool.py'), f'{d}/pool.jsonl', f'{d}/train.jsonl', f'{d}/calib.jsonl'], cwd=PKG, check=True, capture_output=True)
    out = subprocess.run([sys.executable, os.path.join(DATA, 'lint_data.py'), '--train', f'{d}/train.jsonl', f'{d}/calib.jsonl'], cwd=PKG, capture_output=True, text=True)
    print(d, 'lint rc', out.returncode, out.stdout.strip().splitlines()[-1][:120] if out.stdout.strip() else out.stderr[-200:])
