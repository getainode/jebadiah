#!/usr/bin/env python3
"""Re-label the synthetic pool (pool-v2/data-v2/synth-*.jsonl) with Fable 5.1 through the Claude Code
harness (subscription, no API key). One call per state, all of its questions at once. N concurrent
calls spread round-robin over the logged-in profiles; a profile that reports a rate limit cools down
for 5 minutes. Answers are cached in answers.jsonl (resume by record id). No em dashes."""
import glob, json, os, subprocess, sys, threading, time, queue, itertools, re
HERE = os.path.dirname(os.path.abspath(__file__))
POOL_V2 = os.environ.get('JEB_POOL_V2', os.path.join(HERE, '..', '..', 'data', 'pool-v2'))
SRC = sorted(glob.glob(os.path.join(POOL_V2, 'data-v2', 'synth-*.jsonl')))
CLAUDE = os.environ.get('CLAUDE_BIN', 'claude'); MODEL = 'claude-fable-5-1'
# logged-in Claude Code config directories to rotate over, colon separated; empty = the default login
PROFILES = [p for p in os.environ.get('CLAUDE_PROFILES', '').split(':') if p] or ['']
CONC = int(os.environ.get('RELABEL_CONC', '12')); TIMEOUT = 300; COOL = 300
INSTR = ("You are labelling decisions for a training set. Below is one JSON object with a \"state\" (the situation to judge) "
         "and \"questions\": a dict of question id to {type, instructions, criteria}. For a \"noul\" question the criteria are "
         "\"true\" and \"false\"; for a \"choice\" question the criteria are the options. For EVERY question return your calibrated "
         "probability for EVERY criterion key (they must sum to 1). Answer with one JSON object mapping each question id to an object "
         "mapping each criterion key to a number, and nothing else: no prose, no code fences. Do not use tools, do not search, "
         "do not read files; judge from the text given.\n\n")
lock = threading.Lock(); cool_until = {p: 0.0 for p in PROFILES}; rr = itertools.count()
def pick_profile():
    now = time.time()
    for _ in range(len(PROFILES)):
        p = PROFILES[next(rr) % len(PROFILES)]
        if cool_until[p] <= now: return p
    return None
def parse(text, rec):
    t = text.strip()
    if t.startswith('```'): t = re.sub(r'^```[a-z]*\n|\n```$', '', t.strip())
    i, j = t.find('{'), t.rfind('}')
    d = json.loads(t[i:j+1])
    out = {}
    for qid, q in rec['questions'].items():
        keys = list(q['criteria'].keys()); a = d.get(qid)
        if not isinstance(a, dict): raise ValueError(f'missing {qid}')
        vals = {k: float(a.get(k, 0.0)) for k in keys}
        if any(v < 0 for v in vals.values()): raise ValueError(f'negative {qid}')
        s = sum(vals.values())
        if s <= 0: raise ValueError(f'zero sum {qid}')
        out[qid] = {k: round(v / s, 6) for k, v in vals.items()}
    return out
def call(rec):
    prompt = INSTR + json.dumps({'state': rec['state'], 'questions': rec['questions']}, ensure_ascii=False)
    for attempt in range(4):
        prof = pick_profile()
        if prof is None: time.sleep(20); continue
        e = dict(os.environ)
        if prof: e['CLAUDE_CONFIG_DIR'] = prof
        for k in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY'): e.pop(k, None)
        t0 = time.time()
        try:
            p = subprocess.run([CLAUDE, '-p', '--model', MODEL, '--output-format', 'json', '--disallowedTools', '*'],
                               input=prompt, capture_output=True, text=True, timeout=TIMEOUT, cwd=HERE, env=e)
            d = json.loads(p.stdout); text = d.get('result') or ''
            if d.get('is_error') or not text: raise RuntimeError(f'is_error {str(d.get("result"))[:200]}')
            targets = parse(text, rec)
            return targets, {'profile': f'profile-{PROFILES.index(prof)}', 'secs': round(time.time() - t0, 1), 'attempt': attempt,
                             'usage': {k: v for k, v in (d.get('usage') or {}).items() if isinstance(v, (int, float))}}
        except Exception as ex:  # noqa: BLE001
            msg = str(ex)[:300]
            if re.search(r'limit|overloaded|529|429', msg, re.I):
                with lock: cool_until[prof] = time.time() + COOL
            log({'id': rec['id'], 'profile': f'profile-{PROFILES.index(prof)}', 'attempt': attempt, 'error': msg, 'secs': round(time.time() - t0, 1)})
            time.sleep(5 * (attempt + 1))
    return None, None
def log(row):
    with lock:
        with open(os.path.join(HERE, 'calls.jsonl'), 'a') as f: f.write(json.dumps(row) + '\n')
def main():
    recs = [json.loads(l) for f in SRC for l in open(f)]
    done = set()
    ap = os.path.join(HERE, 'answers.jsonl')
    if os.path.exists(ap):
        for l in open(ap): done.add(json.loads(l)['id'])
    todo = [r for r in recs if r['id'] not in done]
    print(f'{time.strftime("%H:%M:%S")} records {len(recs)} done {len(done)} todo {len(todo)} conc {CONC}', flush=True)
    q = queue.Queue()
    for r in todo: q.put(r)
    n_ok = [0]; n_fail = [0]; t_start = time.time()
    def worker():
        while True:
            try: rec = q.get_nowait()
            except queue.Empty: return
            targets, meta = call(rec)
            with lock:
                if targets is None:
                    n_fail[0] += 1
                    with open(os.path.join(HERE, 'failed.txt'), 'a') as f: f.write(rec['id'] + '\n')
                else:
                    n_ok[0] += 1
                    with open(ap, 'a') as f: f.write(json.dumps({'id': rec['id'], 'subset': rec['subset'], 'targets': targets, 'meta': meta}) + '\n')
                total = n_ok[0] + n_fail[0]
                if total % 50 == 0 or total == len(todo):
                    el = time.time() - t_start; rate = total / el * 3600
                    print(f'{time.strftime("%H:%M:%S")} progress {total}/{len(todo)} ok {n_ok[0]} failed {n_fail[0]} rate {rate:.0f}/h eta {((len(todo)-total)/max(rate,1))*60:.0f} min', flush=True)
    th = [threading.Thread(target=worker, daemon=True) for _ in range(CONC)]
    for t in th: t.start()
    for t in th: t.join()
    print(f'{time.strftime("%H:%M:%S")} DONE ok {n_ok[0]} failed {n_fail[0]} in {(time.time()-t_start)/60:.0f} min', flush=True)
if __name__ == '__main__': main()
