"""Jebadiah v2 synthetic pool generator: AUTHOR a state plus typed questions, JUDGE every question
with both teachers' own probabilities through /v1/decide, keep what the teachers can answer.

  author  chat completions on the strong teacher (temperature 0.9), one state and 3 to 5 choice or
          noul questions per call, seeded with random attributes; the output carries no answer and
          is linted for label-like fields before anything else happens to it.
  judge   POST /v1/decide against DeepSeek V4 Flash and Qwen3.8 27B; each question's options are
          the exact strings AINode's /v1/systemone would send (translate_one, verbatim copy), so
          the teachers read the same prompt Jebadiah is trained on. Gold = the mean of the two
          distributions, unsharpened. Dropped: argmaxes disagree AND both tops < 0.6; mean top < 0.4.

Resumable: every accepted record and every drop event is appended to state/<family>.jsonl and
state/<family>.events.jsonl as it happens; a restart rebuilds all counters and the dedupe index
from those files. Progress goes to generate.log once a minute. When every family reaches its
target, finalize.py builds data-v2/ and appends the DONE section to REPORT.md.

Never prints or logs the teacher key (JEB_TEACHER_KEY).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))  # jebadiah_prompt, ainode_prompt_verbatim
sys.path.insert(0, os.path.join(HERE, ".."))  # split_pool, lint_data
sys.path.insert(0, HERE)
from ainode_prompt_verbatim import translate_one  # noqa: E402
from jebadiah_prompt import LABEL_KEYS  # noqa: E402
from families import FAMILIES, LENGTHS, seed_attrs  # noqa: E402

# An OpenAI-compatible AINode endpoint serving both teachers (chat completions and /v1/decide).
# JEB_TEACHER_BASE_URL is e.g. https://<your-ainode-host>/v1; JEB_TEACHER_KEY is its API key, read from
# the environment only and never logged.
BASE = os.environ.get("JEB_TEACHER_BASE_URL", "http://127.0.0.1:8000/v1")
TEACHERS = {"deepseek": "fraserprice/DeepSeek-V4-Flash-DSpark", "qwen": "unsloth/Qwen3.8-27B-NVFP4"}
AUTHOR_MODEL = "deepseek"
SET = "synth-v2"
SOURCE = "AINode synthetic 2026-09-22, teachers DeepSeek V4 Flash + Qwen3.8 27B via /v1/decide"
LICENSE = "Apache-2.0 (AINode)"

ANSWERISH_KEYS = set(LABEL_KEYS) | {"correct_option", "correct_answer", "expected_answer", "ground_truth",
                                    "is_correct", "solution", "correct_options", "right_answer"}
# Inside the STATE, these label-like names are also ordinary field names (an SLA target, an invoice
# reference, a PR's labels, an expected delivery). They are allowed there unless the field's value is
# one of a question's option names; everything else in ANSWERISH_KEYS rejects the state outright.
STATE_ORDINARY_KEYS = {"target", "targets", "reference", "expected", "label", "labels"}
# An option literally named like a label key cannot sit under `questions` (lint_data.py rejects it);
# the moderation action "label" is the one that occurs, so it is renamed, and any other such option
# drops its question.
OPTION_RENAMES = {"label": "apply a label", "labels": "apply labels"}
GENERIC_TOKENS = {"date", "time", "name", "type", "status", "level", "code", "number", "value", "text", "info",
                  "data", "list", "current", "note", "notes", "the", "and", "for", "id"}
QID_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")

AGREE_MIN = 0.6    # a question both teachers disagree on and neither is sure of is dropped
TOP_MIN = 0.4      # a question whose averaged top probability is below this is dropped


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ----------------------------------------------------------------------------- text utilities

def flat_text(obj) -> str:
    if isinstance(obj, dict):
        return " ".join(f"{k} {flat_text(v)}" for k, v in obj.items())
    if isinstance(obj, list):
        return " ".join(flat_text(v) for v in obj)
    return str(obj)


def norm_words(obj) -> list[str]:
    return re.findall(r"[a-z0-9]+", flat_text(obj).lower())


def state_hash(state) -> str:
    return hashlib.sha256(" ".join(norm_words(state)).encode()).hexdigest()


def shingles(state, n: int = 5) -> frozenset:
    w = norm_words(state)
    if len(w) < n:
        return frozenset([" ".join(w)])
    return frozenset(hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_keys(obj, keys: set, path="") -> list[str]:
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.strip().lower() in keys:
                hits.append(f"{path}.{k}")
            hits += find_keys(v, keys, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_keys(v, keys, f"{path}[{i}]")
    return hits


def field_values(obj, key=None):
    """(field name, scalar value) for every field NOT inside a list: list items are enumerations of
    candidates (open tickets, tools, attendees) and naturally carry option names."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from field_values(v, str(k))
    elif not isinstance(obj, list) and key is not None:
        yield key, obj


def all_values(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from all_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_values(v)
    else:
        yield obj


# ----------------------------------------------------------------------------- author prompt

AUTHOR_SYSTEM = (
    "You write realistic synthetic data for training a small decision model. Everything you write is "
    "invented: fictional people, companies, addresses, numbers and events. Never copy a real ticket, email, "
    "post, benchmark or dataset item. You write the situation and the questions only, never the answers."
)


def author_prompt(family: str, a: dict, n_choice: int, n_noul: int, n_true: int) -> str:
    fam = FAMILIES[family]
    menu = "\n".join(f"- {q}" for q in fam["questions"])
    extra = ""
    if "lang_pair" in a:
        extra = f"\n- Language pair: {a['lang_pair']} (the source text in the first language, the candidate translation in the second)."
    noul_line = ""
    if n_noul:
        noul_line = (f"\nWrite the situation so that, for a careful reader, {n_true} of the {n_noul} yes/no question(s) "
                     f"come out true and {n_noul - n_true} come out false. Do not say which anywhere in the output.")
    return f"""Family: {family}
The state is {fam['state']}.

Seed attributes for this one (use them, do not mention them as such):
- Setting or industry: {a['industry']}
- Person writing or involved: {a['persona']}
- Tone: {a['tone']}
- Length: {LENGTHS[a['length']]}
- Edge condition to build in: {a['edge']}
- Dates around {a['year']}-{a['month']:02d}{extra}

Question ideas for this family (pick, adapt, or write others in the same spirit):
{menu}

Write exactly {n_choice} choice question(s) and {n_noul} noul (yes/no) question(s) about this state.{noul_line}

Rules:
1. Every question must be answerable from the state alone by a careful reader, with one option clearly best. Borderline is fine when the state gives enough to decide.
2. The state must not contain any field that gives away an answer (no "priority" field if you ask for the priority, no field that states the outcome, category or verdict). Do not use the key names label, labels, answer, target, gold, expected, reference, correct or truth anywhere.
3. choice: "criteria" maps 2 to 12 short, distinct option names (1 to 6 words) to a one-line description of when that option applies. Options must be plausible alternatives, not jokes. Set "ordered": true only when the options are a natural scale (severity levels, P1 to P4, counts), else false.
4. noul: "criteria" is {{"true": "<one line: what true means>", "false": "<one line: what false means>"}}.
5. "instructions" is the question itself, one or two sentences, and never hints at the answer.
6. Question ids are short snake_case names such as routing_queue or needs_human.
7. Use plain hyphens; never an em dash or an en dash.
8. Everything a question is about (the summary, the translation, the reply, the log lines, the listed options' sources) goes in the state; a question never carries material of its own beyond naming what it asks about.
9. Questions talk about the thing by its natural name (the ticket, the email, the invoice, the reply), never "the state", and never mention the seed attributes or the edge condition.

Output one JSON object and nothing else:
{{"state": {{...the state as a JSON object...}},
 "questions": {{"<id>": {{"type": "choice", "instructions": "...", "ordered": false, "criteria": {{"<option>": "<description>", ...}}}},
               "<id>": {{"type": "noul", "instructions": "...", "criteria": {{"true": "...", "false": "..."}}}}}}}}"""


DASHES = {"\u2014": " - ", "\u2013": "-", "\u2015": " - ", "\u2012": "-"}


def sanitize(obj):
    """Em and en dashes become plain hyphens (the pool has no em dash anywhere), before judging,
    so the teachers read exactly the text that is stored."""
    if isinstance(obj, str):
        for k, v in DASHES.items():
            obj = obj.replace(k, v)
        return obj
    if isinstance(obj, dict):
        return {sanitize(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def parse_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            return json.loads(t[i:j + 1])
        raise


def clean_authored(obj, rng: random.Random) -> tuple:
    """(state, questions, orders) or raises ValueError(reason). Linting of the author's raw output."""
    if not isinstance(obj, dict):
        raise ValueError("not_object")
    state, qs = obj.get("state"), obj.get("questions")
    if isinstance(qs, dict):
        for q in qs.values():
            if isinstance(q, dict) and q.get("type") == "choice" and isinstance(q.get("criteria"), dict):
                crit = q["criteria"]
                q["criteria"] = {OPTION_RENAMES.get(str(k).strip().lower(), k): v for k, v in crit.items()}
                if find_keys(q["criteria"], ANSWERISH_KEYS):
                    q["type"] = "dropped"   # skipped below like any other malformed question
    hits = find_keys({k: v for k, v in obj.items() if k != "state"}, ANSWERISH_KEYS) + \
        find_keys(state, ANSWERISH_KEYS - STATE_ORDINARY_KEYS, ".state")
    if hits:
        raise ValueError("label_like_field", hits[:3])
    if not isinstance(state, dict) or len(state) == 0 or len(flat_text(state)) < 60:
        raise ValueError("bad_state")
    if not isinstance(qs, dict) or not qs:
        raise ValueError("no_questions")
    out, ordered = {}, {}
    for i, (qid, q) in enumerate(qs.items()):
        if not isinstance(q, dict):
            continue
        qid = qid.strip().lower() if isinstance(qid, str) else ""
        if not QID_RE.match(qid) or qid in ANSWERISH_KEYS or qid in out:
            qid = f"q{i + 1}"
        t = q.get("type")
        ins = q.get("instructions")
        crit = q.get("criteria")
        if t not in ("choice", "noul") or not isinstance(ins, str) or not ins.strip():
            continue
        if re.search(r"correct answer|the answer is|\(answer", ins, re.I):
            raise ValueError("answer_in_instructions")
        if t == "choice":
            if not isinstance(crit, dict):
                continue
            crit = {str(k).strip(): (str(v).strip() if v is not None else None) for k, v in crit.items() if str(k).strip()}
            if not 2 <= len(crit) <= 12 or any(len(k) > 90 for k in crit):
                continue
            if any(k.lower() in ANSWERISH_KEYS for k in crit):
                continue
            keys = list(crit)
            if not q.get("ordered"):
                rng.shuffle(keys)   # the author tends to list its intended answer first
            crit = {k: crit[k] for k in keys}
            ordered[qid] = bool(q.get("ordered"))
        else:
            if not isinstance(crit, dict) or set(crit) != {"true", "false"} \
                    or not all(isinstance(v, str) and v.strip() for v in crit.values()):
                continue
            crit = {"true": crit["true"].strip(), "false": crit["false"].strip()}
        wq = {"type": t, "instructions": ins.strip(), "criteria": crit}
        try:
            translate_one(qid, wq)   # AINode's own validation of the wire shape
        except Exception:  # noqa: BLE001
            continue
        # a state field that is named like the question and holds one of its option names verbatim
        if t == "choice":
            toks = {x for x in qid.split("_") if len(x) >= 3} - GENERIC_TOKENS
            names = {k.lower() for k in crit}
            for key, v in field_values(state):
                ktoks = set(key.lower().replace("-", "_").split("_"))
                if isinstance(v, str) and v.strip().lower() in names and (toks & ktoks or key.lower() in STATE_ORDINARY_KEYS):
                    raise ValueError("state_leaks_answer", [f"{key}={v[:40]} under {qid}"])
        out[qid] = wq
    if len(out) < 2:
        raise ValueError("too_few_valid_questions")
    return state, dict(list(out.items())[:5]), ordered


# ----------------------------------------------------------------------------- fleet client

class TeacherDown(Exception):
    pass


class Fleet:
    def __init__(self, args, log):
        self.key = os.environ.get("JEB_TEACHER_KEY", "")
        self.log = log
        self.sem = {"deepseek": asyncio.Semaphore(args.ds_inflight), "qwen": asyncio.Semaphore(args.qw_inflight),
                    "author": asyncio.Semaphore(args.author_inflight)}
        self.inflight = collections.Counter()
        self.calls = collections.Counter()
        self.errors = collections.Counter()
        self.rate_limited = 0
        self.consec_fail = collections.Counter()
        self.down_until = {"deepseek": 0.0, "qwen": 0.0}
        self.session = None

    async def start(self):
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=420, sock_connect=15))

    async def close(self):
        if self.session:
            await self.session.close()

    def is_down(self, teacher: str) -> bool:
        return time.time() < self.down_until[teacher]

    async def post(self, teacher: str, path: str, body: dict, tries: int = 6, slot: str = "") -> dict:
        """slot: which in-flight budget the call uses (default the teacher's judge budget)."""
        slot = slot or teacher
        delay = 2.0
        last = ""
        for attempt in range(tries):
            async with self.sem[slot]:
                self.inflight[slot] += 1
                try:
                    self.calls[teacher] += 1
                    async with self.session.post(BASE + path, json=body) as r:
                        status = r.status
                        retry_after = r.headers.get("Retry-After")
                        text = await r.text()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    status, text, retry_after = -1, f"{type(e).__name__}", None
                finally:
                    self.inflight[slot] -= 1
            if status == 200:
                self.consec_fail[teacher] = 0
                return json.loads(text)
            last = f"{status}: {text[:200]}"
            if status == 429:
                self.rate_limited += 1
                wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else delay
                await asyncio.sleep(min(90.0, wait) + random.random())
                delay = min(90.0, delay * 2)
                continue
            self.errors[f"{teacher}:{status}"] += 1
            if 400 <= status < 500:
                raise ValueError(f"http_{status}")
            self.consec_fail[teacher] += 1
            if self.consec_fail[teacher] >= 8 and not self.is_down(teacher):
                self.down_until[teacher] = time.time() + 300
                self.log(f"TEACHER DOWN {teacher}: {self.consec_fail[teacher]} consecutive failures ({last[:120]}); "
                         f"continuing without it for 5 minutes, then probing again")
            await asyncio.sleep(delay + random.random())
            delay = min(90.0, delay * 2)
        raise TeacherDown(f"{teacher} {path}: {last[:160]}")

    async def author(self, messages: list, max_tokens: int) -> tuple[str, dict]:
        body = {"model": TEACHERS[AUTHOR_MODEL], "messages": messages, "temperature": 0.9, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
                "response_format": {"type": "json_object"}}
        d = await self.post(AUTHOR_MODEL, "/chat/completions", body, tries=4, slot="author")
        return d["choices"][0]["message"]["content"] or "", d.get("usage") or {}

    async def decide(self, teacher: str, state: dict, questions: dict, per_question: bool) -> dict:
        """{qid: {name: prob}} for this teacher, names in the question's own keys."""
        tr = {qid: translate_one(qid, q) for qid, q in questions.items()}
        wire = {qid: {"question": t.question, "options": list(t.options)} for qid, t in tr.items()}
        groups = [[qid] for qid in wire] if per_question else [list(wire)]
        out = {}
        for g in groups:
            d = await self.post(teacher, "/decide", {"model": TEACHERS[teacher], "state": state,
                                                     "questions": {qid: wire[qid] for qid in g}})
            for qid in g:
                dec = d["decisions"][qid]
                dist = dec.get("distribution")
                if not dist:
                    out[qid] = None
                    continue
                names = list(tr[qid].names)
                out[qid] = {n: float(dist.get(o, 0.0)) for n, o in zip(names, tr[qid].options)}
        return out


# ----------------------------------------------------------------------------- per-family book-keeping

class Family:
    def __init__(self, name: str, state_dir: str, target: int):
        self.name = name
        self.target = target
        self.path = os.path.join(state_dir, f"{name}.jsonl")
        self.events_path = os.path.join(state_dir, f"{name}.events.jsonl")
        self.records = 0
        self.q = collections.Counter()       # kept questions by type
        self.noul = collections.Counter()    # kept noul labels: true / false
        self.drops = collections.Counter()   # reason -> count (state-level reasons count states)
        self.authored = 0
        self.judged_both = 0
        self.agree = 0
        self.top_sum = 0.0
        self.pending = 0
        self.label_pos = collections.Counter()  # position of the choice label among unordered options

    def load(self, dedupe):
        if os.path.exists(self.path):
            good = []
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        good.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass       # a line cut by a kill mid-write
            with open(self.path, "w", encoding="utf-8") as f:
                for r in good:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in good:
                self.count_record(r)
                dedupe.add(r["state"])
        if os.path.exists(self.events_path):
            with open(self.events_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        self.count_event(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def count_record(self, r):
        self.records += 1
        for qid, q in r["questions"].items():
            self.q[q["type"]] += 1
            if q["type"] == "noul":
                self.noul["true" if r["label"][qid] else "false"] += 1
            else:
                prov = r["provenance"]["questions"][qid]
                if not prov.get("ordered"):
                    self.label_pos[list(q["criteria"]).index(r["label"][qid])] += 1
            self.top_sum += max(r["target"][qid].values())

    def count_event(self, e):
        k = e["kind"]
        if k == "authored":
            self.authored += 1
        elif k == "drop":
            self.drops[e["reason"]] += e.get("n", 1)
        elif k == "judged":
            self.judged_both += e.get("both", 0)
            self.agree += e.get("agree", 0)

    def event(self, e):
        e = {"t": now(), **e}
        self.count_event(e)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def append(self, r):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.count_record(r)

    def effective(self) -> int:
        """Kept questions after finalize.py trims the majority noul label to 60 percent."""
        t, f = self.noul["true"], self.noul["false"]
        lo, hi = min(t, f), max(t, f)
        return self.q["choice"] + lo + min(hi, int(lo * 1.5))

    def true_share(self) -> float:
        n = self.noul["true"] + self.noul["false"]
        return self.noul["true"] / n if n else 0.5

    def done(self) -> bool:
        return self.effective() >= self.target

    def progress(self) -> float:
        return (self.effective() + 3.0 * self.pending) / self.target


class Dedupe:
    def __init__(self):
        self.hashes = set()
        self.sh = []

    def check(self, state) -> str | None:
        if state_hash(state) in self.hashes:
            return "duplicate_hash"
        s = shingles(state)
        for other in self.sh:
            if jaccard(s, other) > 0.8:
                return "near_duplicate_5gram"
        return None

    def add(self, state):
        self.hashes.add(state_hash(state))
        self.sh.append(shingles(state))


# ----------------------------------------------------------------------------- the run

class Run:
    def __init__(self, args):
        self.args = args
        self.state_dir = args.state_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self.logf = open(args.log, "a", encoding="utf-8")
        self.fleet = Fleet(args, self.log)
        self.rng = random.Random(args.seed + int(time.time()))
        names = [f for f in FAMILIES if not args.families or f in args.families.split(",")]
        self.fams = {n: Family(n, self.state_dir, args.target) for n in names}
        self.dedupe = Dedupe()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=args.queue)
        self.stop = asyncio.Event()
        self.started = time.time()
        self.kept_log = collections.deque()   # (time, questions) for the rolling rate
        self.authored_this_run = 0
        self.author_tokens = 0
        self.author_secs = 0.0
        self.author_calls = 0

    def log(self, msg: str):
        line = f"{now()} {msg}"
        self.logf.write(line + "\n")
        self.logf.flush()
        if self.args.echo:
            print(line, flush=True)

    def pick_family(self) -> Family | None:
        open_ = [f for f in self.fams.values() if not f.done()]
        if not open_:
            return None
        return min(open_, key=lambda f: (f.progress(), self.rng.random()))

    def all_done(self) -> bool:
        return all(f.done() for f in self.fams.values())

    async def author_worker(self, wid: int):
        while not self.stop.is_set():
            if self.args.max_states and self.authored_this_run >= self.args.max_states:
                return
            fam = self.pick_family()
            if fam is None:
                return
            if self.fleet.is_down(AUTHOR_MODEL):
                await asyncio.sleep(20)
                continue
            fam.pending += 1
            self.authored_this_run += 1
            handed_off = False
            try:
                a = seed_attrs(fam.name, self.rng)
                n = a["n_questions"]
                n_noul = self.rng.choice([1, 2]) if n >= 4 else 1
                if fam.name.startswith("jde_") or fam.name in ("claim_evidence", "summary_faithfulness", "translation_adequacy"):
                    n_noul = min(n - 1, n_noul + 1)
                n_choice = n - n_noul
                ts = fam.true_share()
                p_true = 0.5 if 0.45 <= ts <= 0.55 else (0.8 if ts < 0.45 else 0.2)
                n_true = sum(self.rng.random() < p_true for _ in range(n_noul))
                msgs = [{"role": "system", "content": AUTHOR_SYSTEM},
                        {"role": "user", "content": author_prompt(fam.name, a, n_choice, n_noul, n_true)}]
                try:
                    t0 = time.time()
                    text, usage = await self.fleet.author(msgs, self.args.author_max_tokens)
                    self.author_secs += time.time() - t0
                    self.author_calls += 1
                    self.author_tokens += int(usage.get("completion_tokens") or 0)
                except TeacherDown as e:
                    fam.event({"kind": "drop", "reason": "author_call_failed"})
                    self.log(f"author failed: {e}")
                    continue
                except ValueError as e:
                    fam.event({"kind": "drop", "reason": f"author_{e}"})
                    continue
                fam.event({"kind": "authored"})
                try:
                    obj = sanitize(parse_json(text))
                except Exception:  # noqa: BLE001
                    fam.event({"kind": "drop", "reason": "author_bad_json"})
                    continue
                try:
                    state, qs, ordered = clean_authored(obj, self.rng)
                except ValueError as e:
                    fam.event({"kind": "drop", "reason": f"lint_{e.args[0]}",
                               **({"detail": e.args[1]} if len(e.args) > 1 else {})})
                    continue
                dup = self.dedupe.check(state)
                if dup:
                    fam.event({"kind": "drop", "reason": dup})
                    continue
                self.dedupe.add(state)   # reserve it now so a parallel twin is caught too
                await self.queue.put((fam, state, qs, ordered, a, n_true, n_noul))
                handed_off = True
            except Exception as e:  # noqa: BLE001
                self.log(f"author worker {wid} error {type(e).__name__}: {e}")
                await asyncio.sleep(5)
            finally:
                if not handed_off:
                    fam.pending -= 1

    async def judge_one(self, teacher, state, qs):
        if self.fleet.is_down(teacher):
            return None
        try:
            return await self.fleet.decide(teacher, state, qs, per_question=(teacher == "qwen" and self.args.qwen_per_question))
        except (TeacherDown, ValueError, KeyError) as e:
            self.log(f"judge {teacher} failed: {str(e)[:160]}")
            return None

    async def judge_worker(self, wid: int):
        while True:
            item = await self.queue.get()
            if item is None:
                return
            fam, state, qs, ordered, attrs, n_true, n_noul = item
            try:
                await self.judge_state(fam, state, qs, ordered, attrs, n_true, n_noul)
            except Exception as e:  # noqa: BLE001
                self.log(f"judge worker {wid} error {type(e).__name__}: {e}")
                fam.event({"kind": "drop", "reason": "judge_error", "n": len(qs)})
            finally:
                fam.pending -= 1
                self.queue.task_done()

    async def judge_state(self, fam, state, qs, ordered, attrs, n_true, n_noul):
        res = {}
        for _ in range(4):   # both teachers down: wait for one rather than drop the state
            ds, qw = await asyncio.gather(self.judge_one("deepseek", state, qs), self.judge_one("qwen", state, qs))
            if ds is not None or qw is not None:
                break
            await asyncio.sleep(60)
        res = {"deepseek": ds or {}, "qwen": qw or {}}
        if ds is None and qw is None:
            fam.event({"kind": "drop", "reason": "no_teacher_answered", "n": len(qs)})
            return
        kept_q, label, target, prov = {}, {}, {}, {}
        both = agree = 0
        for qid, q in qs.items():
            dists = {t: res[t].get(qid) for t in TEACHERS if res[t].get(qid)}
            if not dists:
                fam.event({"kind": "drop", "reason": "no_distribution"})
                continue
            keys = list(q["criteria"]) if q["type"] == "choice" else ["true", "false"]
            avg = {k: sum(d.get(k, 0.0) for d in dists.values()) / len(dists) for k in keys}
            s = sum(avg.values())
            if s <= 0:
                fam.event({"kind": "drop", "reason": "no_distribution"})
                continue
            avg = {k: round(v / s, 6) for k, v in avg.items()}
            tops = {t: max(d, key=d.get) for t, d in dists.items()}
            if len(dists) == 2:
                both += 1
                agree += len(set(tops.values())) == 1
                if len(set(tops.values())) == 2 and all(max(d.values()) < AGREE_MIN for d in dists.values()):
                    fam.event({"kind": "drop", "reason": "teachers_disagree_both_unsure", "type": q["type"]})
                    continue
            best = max(keys, key=lambda k: avg[k])
            if avg[best] < TOP_MIN:
                fam.event({"kind": "drop", "reason": "top_below_0.4", "type": q["type"]})
                continue
            kept_q[qid] = q
            label[qid] = (best == "true") if q["type"] == "noul" else best
            target[qid] = avg
            prov[qid] = {"teachers": {t: {k: round(v, 6) for k, v in d.items()} for t, d in dists.items()},
                         "argmax": tops, **({"ordered": ordered.get(qid, False)} if q["type"] == "choice" else {})}
        fam.event({"kind": "judged", "both": both, "agree": agree})
        if not kept_q:
            fam.event({"kind": "drop", "reason": "state_all_questions_dropped"})
            return
        sid = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        rec = {
            "id": f"{SET}:{fam.name}:{sid}",
            "set": SET,
            "subset": fam.name,
            "source": SOURCE,
            "license": LICENSE,
            "family": f"{fam.name}:{sid}",
            "state": state,
            "questions": kept_q,
            "label": label,
            "target": target,
            "provenance": {
                "generated_at": now(),
                "author": {"model": TEACHERS[AUTHOR_MODEL], "temperature": 0.9, "seed": attrs,
                           "noul_true_requested": n_true, "noul_requested": n_noul},
                "judge": {"route": "/v1/decide", "models": {t: TEACHERS[t] for t in TEACHERS},
                          "answered": sorted(t for t in TEACHERS if res[t]),
                          "gold": "mean of the answering teachers' distributions, unsharpened"},
                "questions": prov,
            },
        }
        fam.append(rec)
        self.kept_log.append((time.time(), len(kept_q)))

    def progress_line(self) -> str:
        el = (time.time() - self.started) / 60
        kept = sum(n for _, n in self.kept_log)
        cutoff = time.time() - 600
        recent = sum(n for t, n in self.kept_log if t >= cutoff)
        win = min(10.0, max(el, 1e-6))
        rate = recent / win
        eff = sum(min(f.effective(), f.target) for f in self.fams.values())
        tot = sum(f.target for f in self.fams.values())
        left = tot - eff
        eta = left / rate / 60 if rate > 0 else float("inf")
        return (f"PROGRESS elapsed {el:.1f} min | kept this run {kept} q | rate {rate:.1f} q/min (last {win:.0f} min) | "
                f"effective {eff}/{tot} | eta {eta:.1f} h | authored this run {self.authored_this_run} | "
                f"author {self.author_tokens / max(1, self.author_calls):.0f} tok/call, {self.author_secs / max(1, self.author_calls):.0f} s/call, "
                f"{self.author_tokens / max(1.0, time.time() - self.started):.0f} tok/s | queue {self.queue.qsize()} | inflight {dict(self.fleet.inflight)} | calls {dict(self.fleet.calls)} | "
                f"errors {dict(self.fleet.errors)} | 429s {self.fleet.rate_limited} | "
                f"down {[t for t in TEACHERS if self.fleet.is_down(t)]}")

    def write_progress(self):
        out = {"t": now(), "elapsed_min": round((time.time() - self.started) / 60, 1), "line": self.progress_line(),
               "families": {}}
        for f in self.fams.values():
            out["families"][f.name] = {
                "records": f.records, "questions": dict(f.q), "effective": f.effective(), "target": f.target,
                "noul": dict(f.noul), "authored": f.authored, "drops": dict(f.drops),
                "agreement": round(f.agree / f.judged_both, 3) if f.judged_both else None,
                "mean_top": round(f.top_sum / max(1, sum(f.q.values())), 3), "done": f.done()}
        tmp = os.path.join(self.state_dir, "progress.json.tmp")
        json.dump(out, open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(self.state_dir, "progress.json"))

    async def reporter(self):
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=60)
            except asyncio.TimeoutError:
                pass
            now_t = time.time()
            while self.kept_log and self.kept_log[0][0] < now_t - 3600 * 24:
                self.kept_log.popleft()
            self.log(self.progress_line())
            for f in self.fams.values():
                self.log(f"  {f.name:24s} eff {f.effective():4d}/{f.target} rec {f.records:4d} "
                         f"c {f.q['choice']:4d} n {f.q['noul']:4d} (T{f.noul['true']}/F{f.noul['false']}) "
                         f"auth {f.authored:4d} pend {f.pending:2d} drops {sum(f.drops.values()):4d}"
                         f"{' DONE' if f.done() else ''}")
            self.write_progress()

    async def main(self):
        self.log(f"START pid {os.getpid()} target {self.args.target}/family over {len(self.fams)} families, "
                 f"author in flight {self.args.author_inflight}, deepseek judge in flight {self.args.ds_inflight}, qwen in flight {self.args.qw_inflight}, "
                 f"qwen per question {self.args.qwen_per_question}, state {self.state_dir}")
        for f in self.fams.values():
            f.load(self.dedupe)
        self.log(f"RESUME {sum(f.records for f in self.fams.values())} records, "
                 f"{sum(sum(f.q.values()) for f in self.fams.values())} questions already kept")
        await self.fleet.start()
        # the models must be serving; nothing here loads or unloads one
        async with self.fleet.session.get(BASE + "/models") as r:
            ids = {m["id"] for m in (await r.json()).get("data", [])}
        for t, m in TEACHERS.items():
            if m not in ids:
                self.fleet.down_until[t] = time.time() + 10 ** 9
                self.log(f"TEACHER MISSING {t} ({m}) not in /v1/models: continuing with the other teacher only")
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop.set)
        rep = asyncio.create_task(self.reporter())
        judges = [asyncio.create_task(self.judge_worker(i)) for i in range(self.args.judges)]
        authors = [asyncio.create_task(self.author_worker(i)) for i in range(self.args.authors)]
        stopper = asyncio.create_task(self.stop.wait())
        await asyncio.wait([asyncio.gather(*authors), stopper], return_when=asyncio.FIRST_COMPLETED)
        if self.stop.is_set():
            self.log("STOP requested: finishing nothing further, in-flight states are abandoned (restart resumes)")
            for t in authors + judges:
                t.cancel()
        else:
            await self.queue.join()
            for _ in judges:
                await self.queue.put(None)
            await asyncio.gather(*judges)
        self.stop.set()
        await asyncio.gather(rep, return_exceptions=True)
        self.write_progress()
        self.log(self.progress_line())
        await self.fleet.close()
        complete = self.all_done()
        self.log(f"END complete={complete}")
        if complete and self.args.finalize:
            self.log("FINALIZE running finalize.py")
            p = subprocess.run([sys.executable, os.path.join(HERE, "finalize.py"), "--report"],
                               cwd=HERE, capture_output=True, text=True)
            with open(os.path.join(HERE, "finalize.log"), "w") as f:
                f.write(p.stdout + p.stderr)
            self.log(f"FINALIZE exit {p.returncode} (finalize.log)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=610, help="kept questions per family (after noul trimming)")
    ap.add_argument("--families", default="", help="comma list (default all)")
    ap.add_argument("--state-dir", default=os.path.join(HERE, "state"))
    ap.add_argument("--log", default=os.path.join(HERE, "generate.log"))
    ap.add_argument("--author-inflight", type=int, default=5, help="DeepSeek author calls in flight")
    ap.add_argument("--ds-inflight", type=int, default=2, help="DeepSeek judge calls in flight")
    ap.add_argument("--qw-inflight", type=int, default=1, help="Qwen judge calls in flight (we capped the Qwen host at 2)")
    ap.add_argument("--qwen-per-question", type=int, default=1)
    ap.add_argument("--authors", type=int, default=7)
    ap.add_argument("--judges", type=int, default=8)
    ap.add_argument("--queue", type=int, default=10)
    ap.add_argument("--author-max-tokens", type=int, default=3000)
    ap.add_argument("--max-states", type=int, default=0, help="stop after authoring this many states (smoke test)")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--no-finalize", dest="finalize", action="store_false")
    ap.add_argument("--echo", action="store_true", help="also print log lines")
    args = ap.parse_args()
    assert args.author_inflight + args.ds_inflight + args.qw_inflight <= 8, "the brief caps the whole run at 8 requests in flight"
    pidf = os.path.join(HERE, "generate.pid")
    if os.path.exists(pidf) and args.state_dir == os.path.join(HERE, "state"):
        try:
            old = int(open(pidf).read().strip())
            os.kill(old, 0)
            sys.exit(f"generate.py already running as pid {old}")
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    if args.state_dir == os.path.join(HERE, "state"):
        open(pidf, "w").write(str(os.getpid()))
    asyncio.run(Run(args).main())


if __name__ == "__main__":
    main()
