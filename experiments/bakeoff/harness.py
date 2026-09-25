"""Second pass: ask frontier teachers through their logged-in CLI harnesses (subscription, no API keys).

Every vendor key variable is removed from the child environment. Batches of 10 questions per call, fixed for
every teacher (seeded shuffle of questions.jsonl, then cut into tens). Inside a prompt the questions carry local
ids q1..q10 so the dataset id (which names the source and split) is not shown to the teacher.
Cache: answers-h-<teacher>.jsonl, one line per question id (ok ones are skipped on rerun).
Call log: calls-h-<teacher>.jsonl, one line per harness call (wall seconds, ok, missing ids, error).

  python harness.py --teachers fable,opus --deadline 520
  python harness.py --smoke gemflash      # one question, prints the raw output
"""
import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(tempfile.gettempdir(), "bakeoff-h")   # neutral cwd: no project instructions, no repo
os.makedirs(WORK, exist_ok=True)
KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY", "GROK_API_KEY",
        "CODEX_API_KEY", "ANTHROPIC_AUTH_TOKEN"]
CLAUDE = os.environ.get("CLAUDE_BIN", "claude")
# Logged-in Claude Code config directories to rotate over, colon separated (CLAUDE_CONFIG_DIR values).
# Empty means the default login only.
PROFILES = [p for p in os.environ.get("CLAUDE_PROFILES", "").split(":") if p] or [""]

TEACHERS = {
    # name: (harness, model)
    "fable": ("claude", "claude-fable-5-1"),
    "opus": ("claude", "claude-opus-5-5"),
    "gpt6astra": ("codex", "gpt-6-astra"),
    "gpt6luna": ("codex", "gpt-6-luna"),
    "grok": ("grok", "grok-4.7"),
    "gemflash": ("gemini", "gemini-3.8-flash"),
    "gempro": ("gemini", "gemini-3.1-pro-preview"),
}
CONC = {"claude": 3, "codex": 3, "grok": 3, "gemini": 3}   # per teacher
TIMEOUT = 180
TIMEOUT_H = {"grok": 300}   # grok-4.7 reasons for ~70 s per batch alone, longer under concurrency

INSTR = ("You are labelling decisions. Below is a JSON list of yes/no questions. Each has an \"id\", a \"state\" "
         "(the text to judge), \"instructions\" (the question) and \"criteria\" (\"true\" and \"false\" describe the two "
         "answers). For each question return your calibrated probability that the \"true\" criterion holds. Answer "
         "with a JSON array of {\"id\": ..., \"true\": p} objects, one per question, and nothing else: no prose, no "
         "code fences. Do not use tools, do not search, do not read files; answer from the text given.\n\n")

SCHEMA = {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "true": {"type": "number"}},
                                     "required": ["id", "true"], "additionalProperties": False}}
# codex --output-schema and grok --json-schema want an object at the root
OBJ_SCHEMA = {"type": "object", "properties": {"answers": SCHEMA}, "required": ["answers"], "additionalProperties": False}


def env():
    e = {k: v for k, v in os.environ.items() if k not in KEYS}
    return e


def load_questions():
    qs = [json.loads(l) for l in open(os.path.join(HERE, "questions.jsonl"))]
    order = list(range(len(qs)))
    random.Random(20260924).shuffle(order)
    qs = [qs[i] for i in order]
    return [qs[i:i + 10] for i in range(0, len(qs), 10)]


def prompt_for(batch):
    items = []
    for j, r in enumerate(batch, 1):
        (_, q), = r["questions"].items()
        items.append({"id": f"q{j}", "state": r["state"], "instructions": q["instructions"], "criteria": q["criteria"]})
    return INSTR + json.dumps(items, ensure_ascii=False, indent=1)


def parse_answers(text):
    """dict local_id -> p from the first JSON array (or {"answers": [...]}) in text."""
    if not text:
        return {}
    arr = None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    for cand in (s, ):
        try:
            v = json.loads(cand)
            arr = v.get("answers") if isinstance(v, dict) else v
        except Exception:  # noqa: BLE001
            pass
    if arr is None:
        m = re.search(r"\[\s*\{.*\}\s*\]", s, re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                arr = None
    out = {}
    for d in arr or []:
        try:
            p = float(d["true"])
            if 0.0 <= p <= 1.0:
                out[str(d["id"]).strip()] = p
        except Exception:  # noqa: BLE001
            continue
    return out


_rr = itertools.count()


def run_cmd(harness, model, prompt):
    """returns (text, meta). text is the model's final message."""
    e = env()
    meta = {}
    if harness == "claude":
        prof = PROFILES[next(_rr) % len(PROFILES)]
        if prof:
            e["CLAUDE_CONFIG_DIR"] = prof
        meta["profile"] = f"profile-{PROFILES.index(prof)}"
        cmd = [CLAUDE, "-p", "--model", model, "--output-format", "json", "--disallowedTools", "*"]
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=TIMEOUT, cwd=WORK, env=e)
        try:
            d = json.loads(p.stdout)
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"rc={p.returncode} out={p.stdout[-300:]} err={p.stderr[-300:]}")
        meta.update({k: d.get(k) for k in ("is_error", "num_turns", "total_cost_usd", "duration_ms")})
        meta["usage"] = {k: v for k, v in (d.get("usage") or {}).items() if isinstance(v, (int, float))}
        if d.get("is_error"):
            raise RuntimeError(f"is_error: {str(d.get('result'))[:300]}")
        return d.get("result") or "", meta
    if harness == "codex":
        with tempfile.TemporaryDirectory() as td:
            sch, out = os.path.join(td, "schema.json"), os.path.join(td, "last.txt")
            json.dump(OBJ_SCHEMA, open(sch, "w"))
            cmd = ["codex", "exec", "-m", model, "--skip-git-repo-check", "-s", "read-only", "--ephemeral",
                   "--color", "never", "--output-schema", sch, "-o", out, "-"]
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=TIMEOUT, cwd=WORK, env=e)
            text = open(out).read() if os.path.exists(out) else ""
            m = re.search(r"tokens used\s*\n?\s*([\d,]+)", p.stderr + p.stdout)
            if m:
                meta["tokens_used"] = int(m.group(1).replace(",", ""))
            if not text:
                raise RuntimeError(f"rc={p.returncode} err={(p.stderr or p.stdout)[-400:]}")
            return text, meta
    if harness == "grok":
        cmd = ["grok", "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(OBJ_SCHEMA), "-m", model,
               "--disable-web-search", "--always-approve", "--tools", "", "--no-subagents", "--cwd", WORK]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_H["grok"], cwd=WORK, env=e, stdin=subprocess.DEVNULL)
        try:
            d = json.loads(p.stdout)
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"rc={p.returncode} out={p.stdout[-300:]} err={p.stderr[-300:]}")
        meta["keys"] = sorted(d.keys()) if isinstance(d, dict) else None
        if isinstance(d, dict):
            for k in ("structured_output", "structuredOutput", "result", "text", "response", "content"):
                if d.get(k):
                    v = d[k]
                    return (v if isinstance(v, str) else json.dumps(v)), meta
        return json.dumps(d), meta
    if harness == "gemini":
        cmd = ["gemini", "-m", model, "-o", "json", prompt]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, cwd=WORK, env=e, stdin=subprocess.DEVNULL)
        try:
            d = json.loads(p.stdout[p.stdout.index("{"):])
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"rc={p.returncode} out={p.stdout[-300:]} err={p.stderr[-400:]}")
        if d.get("error"):
            raise RuntimeError(f"error: {json.dumps(d['error'])[:300]}")
        st = d.get("stats") or {}
        meta["models"] = list((st.get("models") or {}).keys())
        meta["tools"] = (st.get("tools") or {}).get("totalCalls")
        return d.get("response") or "", meta
    raise ValueError(harness)


class Runner:
    def __init__(self, teachers):
        self.batches = load_questions()
        self.lock = threading.Lock()
        self.done, self.fh, self.ch = {}, {}, {}
        for t in teachers:
            path = os.path.join(HERE, f"answers-h-{t}.jsonl")
            self.done[t] = set()
            if os.path.exists(path):
                for l in open(path):
                    self.done[t].add(json.loads(l)["id"])
            self.fh[t] = open(path, "a")
            self.ch[t] = open(os.path.join(HERE, f"calls-h-{t}.jsonl"), "a")

    def call(self, t, batch, attempt):
        harness, model = TEACHERS[t]
        t0 = time.time()
        rec = {"teacher": t, "harness": harness, "model": model, "ids": [r["id"] for r in batch], "attempt": attempt,
               "start": t0}
        try:
            text, meta = run_cmd(harness, model, prompt_for(batch))
            got = parse_answers(text)
            rec.update(meta)
            rec["text"] = text[:3000]
        except subprocess.TimeoutExpired:
            got, rec["error"] = {}, "timeout"
        except Exception as ex:  # noqa: BLE001
            got, rec["error"] = {}, str(ex)[:600]
        rec["secs"] = round(time.time() - t0, 2)
        res = {}
        for j, r in enumerate(batch, 1):
            if f"q{j}" in got:
                res[r["id"]] = got[f"q{j}"]
        rec["n_ok"], rec["n_missing"] = len(res), len(batch) - len(res)
        rec["ok"] = rec["n_missing"] == 0
        with self.lock:
            self.ch[t].write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.ch[t].flush()
            for r in batch:
                if r["id"] in res and r["id"] not in self.done[t]:
                    self.fh[t].write(json.dumps({"id": r["id"], "p": res[r["id"]], "teacher": t, "model": model,
                                                 "harness": harness, "attempt": attempt}) + "\n")
                    self.done[t].add(r["id"])
            self.fh[t].flush()
        return rec

    def do_batch(self, t, batch, deadline):
        for attempt in range(3):          # first ask + up to 2 re-asks for the missing ids
            todo = [r for r in batch if r["id"] not in self.done[t]]
            if not todo or time.time() > deadline:
                return
            rec = self.call(t, todo, attempt)
            if rec.get("error") and ("limit" in rec["error"].lower() or "quota" in rec["error"].lower()):
                time.sleep(30)

    def run(self, teachers, deadline_s, max_batches=0):
        deadline = time.time() + deadline_s
        pools = {}
        futs = []
        for t in teachers:
            h = TEACHERS[t][0]
            pools[t] = ThreadPoolExecutor(4 if t == "grok" else CONC[h])
            todo = [b for b in self.batches if any(r["id"] not in self.done[t] for r in b)]
            if max_batches:
                todo = todo[:max_batches]
            print(f"{t}: {len(self.done[t])} cached, {len(todo)} batches to ask", flush=True)
            for b in todo:
                futs.append(pools[t].submit(self.do_batch, t, b, deadline))
        for f in futs:
            f.result()
        for p in pools.values():
            p.shutdown()
        print("END " + " ".join(f"{t}:{len(self.done[t])}" for t in teachers), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", default="")
    ap.add_argument("--deadline", type=float, default=500)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--smoke", default="")
    a = ap.parse_args()
    if a.smoke:
        h, m = TEACHERS[a.smoke]
        b = load_questions()[0][:1]
        t0 = time.time()
        text, meta = run_cmd(h, m, prompt_for(b))
        print(json.dumps({"secs": round(time.time() - t0, 1), "meta": meta, "text": text[:800],
                          "parsed": parse_answers(text)}, indent=1))
        sys.exit(0)
    ts = a.teachers.split(",")
    Runner(ts).run(ts, a.deadline, a.max_batches)
