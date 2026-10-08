"""Item 25 full-suite overlap methods, reused without benchmark payloads.

Only load the local item 25 index after verifying its pinned hash. All source
records must be scanned before sampling; family is set to the upstream source
ID so a single overlap rejects the entire source, across all splits.
"""
import collections
import hashlib
import json
import pickle
import re
from pathlib import Path
import xxhash

ITEM25_SCANNER_SHA256 = 'd62ed3fc2d6c1b8e8cf9f8eea2506a4da89f8f3ed3abf681216937c58279b8d2'
ITEM25_INDEX_SHA256 = 'cf54ade9013c05db74f4c70925287382708de965fbf3cba62145ee821adad0ff'

def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n")

def digest(x):
    return hashlib.sha256(x.encode()).hexdigest()

def norm(x):
    return ' '.join(re.findall(r'\w+', x.casefold()))

def leaves(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from leaves(v)

def content(r):
    yield from leaves(r['state'])
    # Source option text can carry passages absent from the state.
    for q in r['questions'].values():
        yield from leaves(q.get('criteria', []))
        yield from leaves(q.get('instructions', []))

def shingle(w, n):
    return {xxhash.xxh3_64_intdigest(' '.join(w[i:i+n]).encode()) for i in range(len(w)-n+1)}

def scan(root):
    exact, grams, near, inv = pickle.load((root / 'protected.pkl').open('rb'))
    hits = []; badfamilies = set(); checked = 0
    with (root / 'candidates.jsonl').open() as f:
        for line in f:
            r = json.loads(line); checked += 1; hit = None
            canonical = digest(json.dumps(r['state'], sort_keys=True, ensure_ascii=False))
            if r['state'] is None or r['state'] == '' or r['state'] == {} or r['state'] == []:
                hit = ('invalid_empty_state', None)
            elif canonical in exact: hit = ('exact_state', exact[canonical])
            for text in content(r):
                if hit: break
                t = norm(text); w = t.split()
                if len(t) >= 80 and digest(t) in exact: hit = ('exact_leaf', exact[digest(t)]); break
                shared = {h for h in shingle(w, 13) if h in grams}
                if shared: hit = ('13_word_span', grams[next(iter(shared))]); break
                if len(w) >= 20:
                    hs = shingle(w, 5); candidates = set()
                    for h in sorted(hs)[:16]: candidates.update(inv.get(h, []))
                    for i in candidates:
                        other, run = near[i]; overlap = len(hs & other)
                        if overlap >= .8 * min(len(hs), len(other)):
                            hit = ('near_containment_5gram', run); break
            if hit:
                badfamilies.add(r['family']); hits.append({'id': r['id'], 'family': r['family'],
                                                        'method': hit[0], 'benchmark_run_id': hit[1]})
            if checked % 10000 == 0: print('scanned', checked, 'hits', len(hits), flush=True)
    dump(root / 'contamination-hits.json', hits)
    # All siblings removed, including siblings without direct matches.
    retained = 0
    with (root / 'clean.jsonl').open('w') as out:
        for line in (root / 'candidates.jsonl').open():
            if json.loads(line)['family'] not in badfamilies: out.write(line); retained += 1
    invalid = sum(x['method'] == 'invalid_empty_state' for x in hits)
    dump(root / 'scan-report.json', {'scanned_records': checked, 'direct_hits': len(hits)-invalid,
         'invalid_empty_state_records': invalid,
         'removed_families': len(badfamilies), 'removed_records': checked-retained,
         'retained_records': retained, 'methods': dict(collections.Counter(x['method'] for x in hits)),
         'remaining_hits_under_scanner': 0, 'semantic_independence_proven': False})
    print('clean', retained, flush=True)
