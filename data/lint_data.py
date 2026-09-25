"""Training-data linter. Fails (exit 1) on any file where:
  - a question object, or anything under `questions`, carries a label-like key
    (label, labels, expected, passingAnswer, answer, target, gold, reference, ...)
  - a record has no `label` map, or a question with no label, or a label that does not fit the
    question (choice: not an option key; noul: not a bool; score: not a level index)
  - a question is malformed (unknown type, empty instructions, bad criteria shape)
  - a state is empty
Usage: python lint_data.py [--train] FILE [FILE ...]   (--train also caps choice questions at 20 options)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from jebadiah_prompt import LABEL_KEYS, wire_keys as answer_keys

MAX_OPTIONS_TRAIN = 20  # AINode's /v1/decide reads top_logprobs=20; /v1/systemone refuses more


def walk_keys(obj, path, problems):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in LABEL_KEYS:
                problems.append(f"{path}: forbidden key {k!r} inside the question payload")
            walk_keys(v, f"{path}.{k}", problems)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_keys(v, f"{path}[{i}]", problems)


def lint_record(r: dict, where: str, problems: list, max_options: int = 255) -> None:
    qs = r.get("questions")
    if not isinstance(qs, dict) or not qs:
        problems.append(f"{where}: no questions")
        return
    for qid, q in qs.items():
        # the question id is the caller's name for the question and is never shown to the model;
        # the label-key rule applies to the question object and everything inside it
        walk_keys(q, f"{where}.questions.{qid}", problems)
    labels = r.get("label")
    if not isinstance(labels, dict):
        problems.append(f"{where}: no record-level label map")
        labels = {}
    st = r.get("state")
    if st is None or (isinstance(st, (str, list, dict)) and len(st) == 0):
        problems.append(f"{where}: empty state")
    for qid, q in qs.items():
        p = f"{where}.questions.{qid}"
        if not isinstance(q, dict) or q.get("type") not in ("choice", "noul", "score"):
            problems.append(f"{p}: bad type {q.get('type') if isinstance(q, dict) else q!r}")
            continue
        extra = set(q) - {"type", "instructions", "criteria"}
        if extra:
            problems.append(f"{p}: unexpected keys {sorted(extra)}")
        ins = q.get("instructions")
        if isinstance(ins, str):
            if not ins.strip():
                problems.append(f"{p}: empty instructions")
        elif not isinstance(ins, (dict, list)) or len(ins) == 0:
            problems.append(f"{p}: instructions must be a non-empty string, object or list")
        crit = q.get("criteria")
        t = q["type"]
        if t == "choice":
            if isinstance(crit, dict):
                keys = list(crit)
            elif isinstance(crit, list):
                keys = [str(c) for c in crit]
            else:
                problems.append(f"{p}: choice criteria must be an object or a list")
                continue
            if len(keys) < 1 or len(set(keys)) != len(keys):
                problems.append(f"{p}: choice needs distinct option keys")
        elif t == "score":
            if not isinstance(crit, list) or len(crit) < 2:
                problems.append(f"{p}: score criteria must be an ordered list of at least two levels")
                continue
        elif t == "noul" and crit is not None:
            if not isinstance(crit, dict) or set(crit) - {"true", "false"}:
                problems.append(f"{p}: noul criteria may only carry true/false descriptions")
        if qid not in labels:
            problems.append(f"{p}: no label")
            continue
        lab = labels[qid]
        try:
            keys = answer_keys(q)
        except Exception as e:
            problems.append(f"{p}: {e}")
            continue
        if t == "choice" and len(keys) > max_options:
            problems.append(f"{p}: {len(keys)} options is more than the {max_options} the served route can carry")
        tgt = (r.get("target") or {}).get(qid)
        if tgt is not None:
            if not isinstance(tgt, dict) or set(tgt) != set(keys) or any(not isinstance(v, (int, float)) or v < 0 for v in tgt.values()) \
                    or abs(sum(tgt.values()) - 1.0) > 1e-3:
                problems.append(f"{p}: target must be a distribution over exactly the option keys")
            elif max(keys, key=lambda k: tgt[k]) != (("true" if lab else "false") if t == "noul" else str(lab)):
                problems.append(f"{p}: target argmax disagrees with the label")
        if t == "choice" and (not isinstance(lab, str) or lab not in keys):
            problems.append(f"{p}: label {lab!r} is not one of the option keys")
        if t == "noul" and not isinstance(lab, bool):
            problems.append(f"{p}: noul label must be a JSON boolean, got {lab!r}")
        if t == "score" and (isinstance(lab, bool) or not isinstance(lab, int) or not 0 <= lab < len(keys)):
            problems.append(f"{p}: score label must be a level index in [0, {len(keys)}), got {lab!r}")


def lint_file(path: str, max_options: int = 255) -> list:
    problems = []
    n = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{path}:{i}: not JSON ({e})")
                continue
            lint_record(r, f"{path}:{i}", problems, max_options)
    return n, problems


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    max_options = 255
    if argv[0] == "--train":
        max_options = MAX_OPTIONS_TRAIN
        argv = argv[1:]
    bad = 0
    for path in argv:
        n, problems = lint_file(path, max_options)
        if problems:
            bad += 1
            print(f"FAIL {path}: {n} records, {len(problems)} problems")
            for p in problems[:20]:
                print("  ", p)
            if len(problems) > 20:
                print(f"   ... {len(problems) - 20} more")
        else:
            print(f"ok   {path}: {n} records, no label-like key under questions, every label fits")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
