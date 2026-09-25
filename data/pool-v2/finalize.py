"""Turn the generator's checkpoints into the v2 pool.

  1. read state/<family>.jsonl, re-lint every record with the trainer's own linter (20-option cap)
  2. noul balance: trim the majority label in any family outside 40/60, least confident first
  3. eval overlap: drop any state whose 5-gram containment against any reported eval set's state
     (the 20 test sets convert_data.py rebuilt) is 0.5 or more, or whose normalised hash matches one
  4. write data-v2/synth-<family>.jsonl, pool-v2.jsonl = v1 pool + synthetic records,
     train.jsonl / calib.jsonl with split_pool.py, manifest.json, SAMPLE.md (200 questions)
  5. lint_data.py --train on every file written
  --report appends the DONE section to REPORT.md.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))  # jebadiah_prompt, ainode_prompt_verbatim
sys.path.insert(0, os.path.join(HERE, ".."))  # split_pool, lint_data
sys.path.insert(0, HERE)
from lint_data import lint_record  # noqa: E402
from generate import FAMILIES, TEACHERS, norm_words, state_hash  # noqa: E402

STATE = os.path.join(HERE, "state")
OUT = os.path.join(HERE, "data-v2")
V1_DIR = os.environ.get("JEB_V1_DIR", os.path.join(HERE, "v1-rebuild", "data-v1"))  # convert_data.py --out
PY = sys.executable


def read_jsonl(path):
    with open(path, encoding="utf-8", newline="\n") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grams(obj, n=5):
    w = norm_words(obj)
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def drop_question(r, qid):
    for k in ("questions", "label", "target"):
        r[k].pop(qid, None)
    r["provenance"]["questions"].pop(qid, None)


def read_checkpoint(path):
    """A checkpoint may be read while the generator appends: skip a half-written last line."""
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="\n") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def events(family):
    return read_checkpoint(os.path.join(STATE, f"{family}.events.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="append the DONE section to REPORT.md")
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ------------------------------------------------ eval-set index (5-gram containment)
    test_files = sorted(glob.glob(os.path.join(V1_DIR, "test", "*.jsonl")))
    inv = collections.defaultdict(set)
    test_hash = {}
    test_ids = []
    for tf in test_files:
        for r in read_jsonl(tf):
            i = len(test_ids)
            test_ids.append(f"{os.path.basename(tf)[:-6]}:{r.get('id')}")
            test_hash[state_hash(r["state"])] = test_ids[i]
            for g in grams(r["state"]):
                inv[g].add(i)

    def eval_overlap(state):
        h = state_hash(state)
        if h in test_hash:
            return 1.0, test_hash[h]
        gs = grams(state)
        hits = collections.Counter()
        for g in gs:
            for i in inv.get(g, ()):
                hits[i] += 1
        if not hits:
            return 0.0, None
        i, n = hits.most_common(1)[0]
        return n / max(1, len(gs)), test_ids[i]

    fam_stats, all_synth = {}, []
    rng = random.Random(20260922)
    max_overlap = (0.0, None)
    for fam in FAMILIES:
        path = os.path.join(STATE, f"{fam}.jsonl")
        recs = read_checkpoint(path)
        ev = events(fam)
        drops = collections.Counter()
        for e in ev:
            if e["kind"] == "drop":
                drops[e["reason"]] += e.get("n", 1)
        both = sum(e.get("both", 0) for e in ev if e["kind"] == "judged")
        agree = sum(e.get("agree", 0) for e in ev if e["kind"] == "judged")
        authored = sum(1 for e in ev if e["kind"] == "authored")
        # lint
        kept = []
        for r in recs:
            probs = []
            lint_record(r, r["id"], probs, 20)
            if probs:
                drops["final_lint"] += len(r["questions"])
                continue
            ov, which = eval_overlap(r["state"])
            if ov > max_overlap[0]:
                max_overlap = (ov, which, r["id"])
            if ov >= 0.5:
                drops["eval_overlap"] += len(r["questions"])
                continue
            kept.append(r)
        # noul balance, trimming the majority label's least confident questions first
        nouls = [(r, qid) for r in kept for qid, q in r["questions"].items() if q["type"] == "noul"]
        t = [x for x in nouls if x[0]["label"][x[1]] is True]
        f = [x for x in nouls if x[0]["label"][x[1]] is False]
        maj, mino = (t, f) if len(t) > len(f) else (f, t)
        allowed = int(len(mino) * 1.5)
        if len(maj) > allowed:
            maj.sort(key=lambda x: max(x[0]["target"][x[1]].values()))
            for r, qid in maj[:len(maj) - allowed]:
                drop_question(r, qid)
                drops["noul_balance_trim"] += 1
        kept = [r for r in kept if r["questions"]]
        types = collections.Counter(q["type"] for r in kept for q in r["questions"].values())
        noul = collections.Counter(("true" if r["label"][qid] else "false") for r in kept
                                   for qid, q in r["questions"].items() if q["type"] == "noul")
        tops = [max(r["target"][qid].values()) for r in kept for qid in r["questions"]]
        pos = collections.Counter()
        nopt = collections.Counter()
        single = 0
        for r in kept:
            if len(r["provenance"]["judge"]["answered"]) < 2:
                single += len(r["questions"])
            for qid, q in r["questions"].items():
                if q["type"] == "choice":
                    nopt[len(q["criteria"])] += 1
                    if not r["provenance"]["questions"][qid].get("ordered"):
                        pos[list(q["criteria"]).index(r["label"][qid])] += 1
        write_jsonl(os.path.join(OUT, f"synth-{fam}.jsonl"), kept)
        all_synth += kept
        fam_stats[fam] = {
            "states_authored": authored, "states_kept": len(kept),
            "questions": sum(types.values()), "by_type": dict(types),
            "noul_labels": dict(noul),
            "noul_true_share": round(noul["true"] / max(1, sum(noul.values())), 3),
            "drops": dict(sorted(drops.items())),
            "teacher_agreement": round(agree / both, 3) if both else None,
            "questions_judged_by_both": both,
            "questions_single_teacher": single,
            "mean_top_probability": round(sum(tops) / max(1, len(tops)), 3),
            "choice_label_position_unordered": {str(k): pos[k] for k in sorted(pos)},
            "choice_option_counts": {str(k): nopt[k] for k in sorted(nopt)},
        }

    # ------------------------------------------------ merged pool, split
    v1_pool = os.path.join(V1_DIR, "pool.jsonl")
    v1 = list(read_jsonl(v1_pool))
    pool_path = os.path.join(OUT, "pool-v2.jsonl")
    write_jsonl(pool_path, v1 + all_synth)
    split = subprocess.run([PY, os.path.join(HERE, "..", "split_pool.py"), pool_path,
                            os.path.join(OUT, "train.jsonl"), os.path.join(OUT, "calib.jsonl")],
                           capture_output=True, text=True)
    split_ok = split.returncode == 0
    # family-disjoint check across the split
    fam_sides = collections.defaultdict(set)
    for side in ("train", "calib"):
        for r in read_jsonl(os.path.join(OUT, f"{side}.jsonl")):
            fam_sides[f"{r['set']}:{r.get('family')}"].add(side)
    straddle = sum(1 for s in fam_sides.values() if len(s) > 1)

    def counts(rows):
        types = collections.Counter(q["type"] for r in rows for q in r["questions"].values())
        return {"records": len(rows), "questions": sum(types.values()), "types": dict(types),
                "score_share": round(types.get("score", 0) / max(1, sum(types.values())), 4)}

    pool_counts = counts(v1 + all_synth)
    synth_counts = counts(all_synth)
    lint_files = [pool_path, os.path.join(OUT, "train.jsonl"), os.path.join(OUT, "calib.jsonl")] + \
        [os.path.join(OUT, f"synth-{f}.jsonl") for f in FAMILIES]
    lint = subprocess.run([PY, os.path.join(HERE, "..", "lint_data.py"), "--train"] + lint_files,
                          capture_output=True, text=True, cwd=os.path.join(HERE, "..", "..", "train"))
    v1_manifest = json.load(open(os.path.join(V1_DIR, "manifest.json")))
    manifest = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pool-v2.jsonl": {"sha256": sha256_file(pool_path), **pool_counts,
                          "sets": dict(collections.Counter(r["set"] for r in v1 + all_synth)),
                          "by_family": {**{f"{s}": n for s, n in collections.Counter(
                              r["subset"] for r in all_synth).items()}}},
        "v1_part": {**counts(v1), "sha256": sha256_file(v1_pool),
                    "how": "rebuilt on this Mac with the package's train/convert_data.py (all stages, no --limit, "
                           "JEB_SCORE_CAP default 0.45); no v1 pool.jsonl existed under package/results/",
                    "convert_manifest_failures": v1_manifest.get("failures")},
        "synth_part": {**synth_counts, "set": "synth-v2",
                       "source": "AINode synthetic 2026-09-22, teachers DeepSeek V4 Flash + Qwen3.8 27B via /v1/decide",
                       "license": "Apache-2.0 (AINode)", "teachers": TEACHERS,
                       "by_family": fam_stats},
        "split": {"ok": split_ok, "output": json.loads(split.stdout) if split_ok else split.stderr[-2000:],
                  "families_on_both_sides": straddle},
        "eval_overlap": {"test_sets": [os.path.basename(t) for t in test_files], "test_states": len(test_ids),
                         "rule": "drop a synthetic state with 5-gram containment >= 0.5 in any test state, or a normalised hash match",
                         "max_containment_seen": {"value": round(max_overlap[0], 3),
                                                  "test_item": max_overlap[1] if len(max_overlap) > 1 else None,
                                                  "synth_record": max_overlap[2] if len(max_overlap) > 2 else None},
                         "dropped_questions": sum(s["drops"].get("eval_overlap", 0) for s in fam_stats.values())},
        "lint": {"exit": lint.returncode, "output": lint.stdout.strip().splitlines()},
    }
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    write_sample(all_synth, args.sample, rng)
    print(json.dumps({"pool": pool_counts, "synth": synth_counts, "lint_exit": lint.returncode,
                      "split_ok": split_ok, "straddle": straddle}, indent=1))
    print(lint.stdout)
    if args.report:
        append_done(manifest)


def fmt_dist(d):
    return ", ".join(f"{k}: {v:.3f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))


def write_sample(rows, n, rng):
    by_fam = collections.defaultdict(list)
    for r in rows:
        for qid in r["questions"]:
            by_fam[r["subset"]].append((r, qid))
    fams = sorted(by_fam)
    picks = []
    per = max(1, n // max(1, len(fams)))
    for f in fams:
        items = by_fam[f][:]
        rng.shuffle(items)
        picks += items[:per]
    rest = [x for f in fams for x in by_fam[f] if x not in picks]
    rng.shuffle(rest)
    picks += rest[:max(0, n - len(picks))]
    picks.sort(key=lambda x: (x[0]["subset"], x[0]["id"], x[1]))
    L = ["# Jebadiah v2 synthetic pool: audit sample", "",
         f"{len(picks)} questions drawn at random, about {per} per family (seed 20260922). For each: the state, the "
         "question, the options, both teachers' distributions from /v1/decide, the averaged gold and its argmax "
         "(the label). States longer than 1,500 characters are cut here only, not in the pool.", ""]
    last = None
    for i, (r, qid) in enumerate(picks, 1):
        if r["subset"] != last:
            L += [f"## {r['subset']}", ""]
            last = r["subset"]
        q = r["questions"][qid]
        st = json.dumps(r["state"], ensure_ascii=False, indent=1)
        if len(st) > 1500:
            st = st[:1500] + "\n... (cut)"
        prov = r["provenance"]["questions"][qid]
        L += [f"### {i}. {r['id']} / {qid} ({q['type']})", "", "```json", st, "```", "",
              f"**Question:** {q['instructions']}", ""]
        for k, v in q["criteria"].items():
            L.append(f"- `{k}`: {v}")
        L.append("")
        for t, d in prov["teachers"].items():
            L.append(f"- {t}: {fmt_dist(d)}")
        lab = r["label"][qid]
        L += [f"- **gold (mean):** {fmt_dist(r['target'][qid])}", f"- **label:** `{str(lab).lower() if isinstance(lab, bool) else lab}`", ""]
    with open(os.path.join(HERE, "SAMPLE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def append_done(m):
    p, s = m["pool-v2.jsonl"], m["synth_part"]
    L = ["", "## DONE", "", f"Finalized {m['built']} (finalize.py, run automatically when the generator reached every family's target).", "",
         "| | records | questions | choice | noul | score | score share |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, c in (("v1 part (rebuilt)", m["v1_part"]), ("synth-v2 (new)", s), ("pool-v2.jsonl", p)):
        t = c["types"]
        L.append(f"| {name} | {c['records']:,} | {c['questions']:,} | {t.get('choice', 0):,} | {t.get('noul', 0):,} | "
                 f"{t.get('score', 0):,} | {100 * c['score_share']:.1f} percent |")
    L += ["", "| family | states kept / authored | choice | noul (T/F) | drops by reason | agreement | mean top |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for f, st in s["by_family"].items():
        d = "; ".join(f"{k} {v}" for k, v in st["drops"].items()) or "none"
        nl = st["noul_labels"]
        L.append(f"| {f} | {st['states_kept']} / {st['states_authored']} | {st['by_type'].get('choice', 0)} | "
                 f"{st['by_type'].get('noul', 0)} ({nl.get('true', 0)}/{nl.get('false', 0)}) | {d} | "
                 f"{st['teacher_agreement']} | {st['mean_top_probability']} |")
    sp = m["split"]["output"]
    L += ["", f"Split (split_pool.py, 95/5 family hash): train {sp['train']['questions']:,} questions, calib "
          f"{sp['calib']['questions']:,}; families on both sides: {m['split']['families_on_both_sides']}." if m["split"]["ok"]
          else "Split FAILED, see data-v2/manifest.json.",
          f"Eval overlap: {m['eval_overlap']['test_states']:,} states in {len(m['eval_overlap']['test_sets'])} reported test sets; "
          f"highest 5-gram containment of any kept synthetic state {m['eval_overlap']['max_containment_seen']['value']}; "
          f"{m['eval_overlap']['dropped_questions']} questions dropped for overlap.",
          f"Lint (lint_data.py --train, every file): exit {m['lint']['exit']}.", ""]
    L += ["```"] + m["lint"]["output"] + ["```", "", "Audit sample: SAMPLE.md. Full numbers: data-v2/manifest.json.", ""]
    with open(os.path.join(HERE, "REPORT.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
