"""Nonce robustness test for one adapter (or the bare base) on one eval set: the check Jev's
creator describes. A model is robust when similar inputs give similar outputs, so every question
is scored once clean and then K more times with a random nonce (a fresh uuid4) planted in the
request, and the picks and probabilities are compared with the clean pass. eval_jebadiah's
flip_rate only compares identical repeats; this is the stronger test, and it runs on every adapter
the sweep produces (see nonce/REPORT.md for the commands).

Two variants, reported separately (both by default; --variants picks):
  state   the uuid is added to the state as an extra top-level field "request_id". The state's
          own fields are never touched. A state that is not a JSON object, or one that already
          carries a request_id, is skipped for this variant and counted under `skipped`.
  instr   the same uuid is appended to the question's instructions as a trailing line
          "ref: <uuid>"; the state stays clean.
  both    the two at once (opt-in: --variants state,instr,both).

Model loading, prompt rendering and the fp32 candidate-logit read are eval_jebadiah's own
(jebadiah_model.load_base / load_adapter / Scorer, eval_jebadiah.pick / label_key), and the clean
pass scores the questions in the order eval_jebadiah's repeat 0 does, so the clean numbers are a
plain eval's. On the first 8 questions the clean picks are checked against eval_jebadiah.run_set
and the run stops on a mismatch (--no-check skips that, --check-questions changes the count).

Per set, per variant, overall and split by question type (choice / score / noul):
  pick_agreement           share of nonce passes whose pick equals the clean pick
  any_flip_rate            share of questions where at least one nonce pass changed the pick
  p_max_abs_change_*       mean and p95 of |p_max(nonce) - p_max(clean)| over the nonce passes
  prob_tv_*                mean and p95 of the total variation distance, L1/2, between the nonce
                           and the clean probability vector (same keys, same order)
  accuracy_clean           clean-pass accuracy against the set's labels (pick == label, the
                           eval_jebadiah rule, ties included)
  accuracy_nonce           the same pooled over every nonce pass, plus one number per pass
  flips_right_to_wrong,    how many questions a flip took from correct to wrong and back
  flips_wrong_to_right
  score_expected_abs_change_mean   score questions only: |E[level](nonce) - E[level](clean)|

Writes one JSON (--out): {meta, overall: {variant: ...}, by_type: {type: {variant: ...}},
questions: [...]}. Prints one summary line per variant and NONCE_DONE at the end.

CUDA when available, CPU otherwise (slow, but the same numbers). --toy swaps the base weights for
a random 2-layer model built from the base's config (the tokenizer is still the base's), which
checks the plumbing end to end in seconds; its picks mean nothing.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
import uuid

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))

import eval_jebadiah  # noqa: E402
from eval_jebadiah import label_key, pick  # noqa: E402
from jebadiah_model import (  # noqa: E402
    FP32_CANDIDATE_LOGITS, Scorer, load_adapter, load_base, load_tokenizer, model_class, read_temperatures,
)

VARIANTS = ("state", "instr", "both")
NONCE_FIELD = "request_id"
SOURCE_FILES = ("nonce_eval.py", "eval_jebadiah.py", "jebadiah_model.py", "jebadiah_prompt.py", "ainode_prompt_verbatim.py")


# ---------------------------------------------------------------- nonces and their injection

def make_nonce(rng: random.Random) -> str:
    """A version-4 UUID drawn from the run's seeded generator, so a run is reproducible from
    --seed while every nonce still has the shape of a real request id."""
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def state_with_nonce(state, nonce: str):
    """The state plus a top-level request_id. None when the state cannot take one (it is not a
    JSON object) or already has one; existing fields are never rewritten."""
    if not isinstance(state, dict) or NONCE_FIELD in state:
        return None
    out = dict(state)
    out[NONCE_FIELD] = nonce
    return out


def question_with_nonce(q: dict, nonce: str) -> dict:
    """The question with "ref: <uuid>" as a trailing line of its instructions."""
    text = str(q.get("instructions", "")).rstrip()
    return {**q, "instructions": f"{text}\nref: {nonce}"}


def perturbed(variant: str, state, q: dict, nonce: str):
    """(state, question) for one variant, or None when the variant cannot apply to this state."""
    if variant == "instr":
        return state, question_with_nonce(q, nonce)
    st = state_with_nonce(state, nonce)
    if st is None:
        return None
    if variant == "state":
        return st, q
    return st, question_with_nonce(q, nonce)


# ---------------------------------------------------------------- the model

def toy_model(base: str, revision: str | None, device: str, dtype, seed: int):
    """A random 2-layer model from the base's config: one linear-attention and one full-attention
    layer when the architecture has both, narrow hidden sizes, the base's vocabulary (the label
    token ids must be the real ones). Only fields the config actually has are changed."""
    import copy
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(base, revision=revision)
    text = copy.deepcopy(cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg)
    text.num_hidden_layers = 2
    if getattr(text, "layer_types", None):
        kinds = list(dict.fromkeys(text.layer_types))
        text.layer_types = (kinds + kinds)[:2] if len(kinds) > 1 else [kinds[0], kinds[0]]
    small = {"hidden_size": 256, "intermediate_size": 512, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 64, "linear_num_key_heads": 4, "linear_num_value_heads": 8, "linear_key_head_dim": 32,
             "linear_value_head_dim": 32, "use_cache": False}
    for k, v in small.items():
        if hasattr(text, k):
            setattr(text, k, v)
    torch.manual_seed(seed)
    cls = model_class(base, revision)
    model = AutoModelForCausalLM.from_config(text) if cls is AutoModelForCausalLM else cls(text)
    model = model.to(device=device, dtype=dtype)
    model.config.use_cache = False
    return model


def load_model(args, device: str, dtype):
    if args.toy:
        model = toy_model(args.base, args.revision, device, dtype, args.seed)
    else:
        model = load_base(args.base, args.revision, attn_implementation=args.attn, dtype=dtype, device=device)
    adapter = None if args.adapter.strip().lower() in ("", "none") else args.adapter
    if adapter:
        model = load_adapter(model, adapter)
    temps = read_temperatures(adapter) if (adapter and args.temperatures == "adapter") else {}
    return model, adapter, temps


# ---------------------------------------------------------------- scoring

def load_records(path: str, limit: int) -> list[dict]:
    with open(path, encoding="utf-8", newline="\n") as f:
        records = [json.loads(l) for l in f if l.strip()]
    return records[:limit] if limit else records


def questions_of(records: list[dict]) -> list[dict]:
    """One entry per question, in record order then question order (eval_jebadiah's order)."""
    out = []
    for r in records:
        for qid, q in r["questions"].items():
            out.append({"rid": r["id"], "qid": qid, "state": r["state"], "q": q, "type": q["type"],
                        "label": label_key(q, r["label"][qid])})
    return out


def prompt_len(scorer: Scorer, state, q: dict) -> int:
    return len(scorer.tok.encode(scorer.render(state, q).prompt, add_special_tokens=False))


def score_items(scorer: Scorer, items: list[tuple], batch: int) -> list[dict]:
    """items: (key, state, question). Renders and scores them in the given order, `batch` at a
    time, through Scorer.render and Scorer.score_rendered (eval_jebadiah's path). Returns one
    result per item: keys, probs (rounded as eval_jebadiah rounds), pick, p_max, truncated."""
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        rendered = [(scorer.render(st, q), q["type"]) for _, st, q in chunk]
        probs = scorer.score_rendered(rendered)
        for (key, st, q), (rd, _), p in zip(chunk, rendered, probs):
            # the pick is taken on the unrounded probabilities, as eval_jebadiah takes it
            out.append({"key": key, "keys": rd.keys, "probs": [round(x, 6) for x in p], "pick": pick(q, rd.keys, p),
                        "p_max": round(max(p), 6), "truncated": bool(rd.truncated), "label_scheme": rd.label_scheme})
    return out


def run_clean(scorer: Scorer, qs: list[dict], batch: int, lengths: dict) -> dict:
    """The clean pass in eval_jebadiah.run_set's repeat-0 order: questions sorted by rendered
    length (stable), batches of `batch` in that order. Same batches, same numbers."""
    order = sorted(range(len(qs)), key=lambda i: lengths[i])
    items = [(i, qs[i]["state"], qs[i]["q"]) for i in order]
    return {r["key"]: r for r in score_items(scorer, items, batch)}


def run_nonce(scorer: Scorer, qs: list[dict], nonces: list[list[str]], variants: list[str], batch: int,
              lengths: dict) -> tuple[dict, dict]:
    """Every (question, variant, pass) that applies, batched by the question's clean length so
    padding stays small. Returns {(i, variant, k): result} and the per-variant skip counts."""
    items, skipped = [], collections.Counter()
    for i in sorted(range(len(qs)), key=lambda i: lengths[i]):
        for k, nonce in enumerate(nonces[i]):
            for v in variants:
                pq = perturbed(v, qs[i]["state"], qs[i]["q"], nonce)
                if pq is None:
                    skipped[v] += 1
                    continue
                items.append(((i, v, k), pq[0], pq[1]))
    return {r["key"]: r for r in score_items(scorer, items, batch)}, dict(skipped)


# ---------------------------------------------------------------- metrics

def p95(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(0.95 * len(s)))]


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def tv_distance(a: list[float], b: list[float]) -> float:
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def expected_level(keys: list[str], probs: list[float]) -> float:
    return sum(int(k) * p for k, p in zip(keys, probs))


def summarize(qrecs: list[dict], variant: str, repeats: int) -> dict:
    """The variant's metrics over the questions it applied to."""
    rows = [(q, q["variants"][variant]) for q in qrecs if variant in q["variants"]]
    n = len(rows)
    if n == 0:
        return {"n_questions": 0, "n_passes": 0}
    agree, flips, dp, tv, correct_nonce = [], 0, [], [], []
    by_pass = [[] for _ in range(repeats)]
    r2w = w2r = 0
    score_dx = []
    for q, v in rows:
        clean_ok = q["clean"]["pick"] == q["label"]
        flipped = False
        for k, pk in enumerate(v["picks"]):
            same = pk == q["clean"]["pick"]
            agree.append(1.0 if same else 0.0)
            flipped = flipped or not same
            dp.append(abs(v["p_max"][k] - q["clean"]["p_max"]))
            tv.append(v["tv"][k])
            ok = pk == q["label"]
            correct_nonce.append(1.0 if ok else 0.0)
            by_pass[k].append(1.0 if ok else 0.0)
            if clean_ok and not ok:
                r2w += 1
            if ok and not clean_ok:
                w2r += 1
            if q["type"] == "score":
                score_dx.append(abs(v["expected"][k] - q["clean"]["expected"]))
        flips += int(flipped)
    out = {
        "n_questions": n, "n_passes": len(agree),
        "pick_agreement": mean(agree), "any_flip_rate": flips / n,
        "p_max_abs_change_mean": mean(dp), "p_max_abs_change_p95": p95(dp),
        "prob_tv_mean": mean(tv), "prob_tv_p95": p95(tv),
        "accuracy_clean": sum(q["clean"]["pick"] == q["label"] for q, _ in rows) / n,
        "accuracy_nonce": mean(correct_nonce),
        "accuracy_nonce_by_pass": [mean(b) for b in by_pass],
        "flips_right_to_wrong": r2w, "flips_wrong_to_right": w2r,
    }
    if score_dx:
        out["score_expected_abs_change_mean"] = mean(score_dx)
    return out


# ---------------------------------------------------------------- the plain-eval cross-check

def check_against_eval(scorer: Scorer, records: list[dict], qs: list[dict], clean: dict, batch: int, n_check: int) -> dict:
    """eval_jebadiah.run_set on the records holding the first n_check questions, one repeat, no
    latency sample; every pick must equal the clean pass's. Returns what was compared."""
    if not torch.cuda.is_available():
        # run_set calls torch.cuda.synchronize() unconditionally; on a CPU box that raises
        torch.cuda.synchronize = lambda *a, **k: None
    want = {(q["rid"], q["qid"]): i for i, q in enumerate(qs[:n_check])}
    rids = {rid for rid, _ in want}
    subset = [r for r in records if r["id"] in rids]
    rows, _ = eval_jebadiah.run_set(scorer, subset, 1, False, batch, 0)
    compared, mismatches, max_dp = [], [], 0.0
    for row in rows:
        i = want.get((row["id"], row["qid"]))
        if i is None:
            continue
        c = clean[i]
        dp = max(abs(a - b) for a, b in zip(row["probs"], c["probs"])) if row["keys"] == c["keys"] else float("inf")
        max_dp = max(max_dp, dp)
        compared.append({"id": row["id"], "qid": row["qid"], "eval_pick": row["pick"], "clean_pick": c["pick"],
                         "max_abs_prob_diff": round(dp, 6)})
        if row["pick"] != c["pick"] or row["keys"] != c["keys"]:
            mismatches.append({"id": row["id"], "qid": row["qid"], "eval": (row["keys"], row["probs"]),
                               "clean": (c["keys"], c["probs"])})
    result = {"questions": len(compared), "max_abs_prob_diff": round(max_dp, 6), "mismatches": mismatches,
              "compared": compared}
    if mismatches:
        raise AssertionError("clean pass differs from eval_jebadiah.run_set on %d of %d questions: %s"
                             % (len(mismatches), len(compared), json.dumps(mismatches)[:2000]))
    return result


# ---------------------------------------------------------------- metadata

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version_of(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return None


def gpu_name() -> str | None:
    try:
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="HF id or path of the base model")
    ap.add_argument("--revision", default=None, help="base model revision")
    ap.add_argument("--adapter", default="none", help="adapter dir, or 'none' for the base alone")
    ap.add_argument("--set", required=True, help="one eval JSONL file in the shape eval_jebadiah reads")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--repeats", type=int, default=3, help="nonce passes per question (K)")
    ap.add_argument("--limit", type=int, default=0, help="records to score (0 = all), as eval_jebadiah --limit")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--seed", type=int, default=20260922, help="seeds the nonces (and the toy weights)")
    ap.add_argument("--variants", default="state,instr", help="comma list of state, instr, both")
    ap.add_argument("--temperatures", choices=["none", "adapter"], default="adapter",
                    help="apply the temperatures saved with the adapter (default, as eval_jebadiah) or none")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="bf16 is eval_jebadiah's")
    ap.add_argument("--memory-cap-gb", type=float, default=0, help="cap the CUDA allocator (0 = none), as eval_jebadiah")
    ap.add_argument("--toy", action="store_true",
                    help="random 2-layer model from the base's config (plumbing check; --attn is not applied)")
    ap.add_argument("--no-check", action="store_true", help="skip the cross-check against eval_jebadiah.run_set")
    ap.add_argument("--check-questions", type=int, default=8)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)
    args.variant_list = [v for v in args.variants.split(",") if v]
    bad = [v for v in args.variant_list if v not in VARIANTS]
    if bad or not args.variant_list:
        ap.error(f"--variants must name some of {', '.join(VARIANTS)} (got {args.variants!r})")
    if args.repeats < 1:
        ap.error("--repeats must be at least 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.manual_seed(args.seed)
    if args.memory_cap_gb and device == "cuda":
        # a shared box: the allocator raises instead of growing past the cap (eval_jebadiah's rule)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.95, args.memory_cap_gb * 1e9 / total), 0)
    t_start = time.time()

    tok = load_tokenizer(args.base, args.revision)
    model, adapter, temps = load_model(args, device, dtype)
    scorer = Scorer(model, tok, args.max_tokens, temperatures=temps, device=device)
    t_loaded = time.time()

    records = load_records(args.set, args.limit)
    qs = questions_of(records)
    if not qs:
        raise SystemExit(f"no questions in {args.set}")
    rng = random.Random(args.seed)
    nonces = [[make_nonce(rng) for _ in range(args.repeats)] for _ in qs]
    lengths = {i: prompt_len(scorer, q["state"], q["q"]) for i, q in enumerate(qs)}

    clean = run_clean(scorer, qs, args.batch, lengths)
    t_clean = time.time()
    check = None
    if not args.no_check and args.check_questions > 0:
        check = check_against_eval(scorer, records, qs, clean, args.batch, min(args.check_questions, len(qs)))
    t_check = time.time()
    nonce_res, skipped = run_nonce(scorer, qs, nonces, args.variant_list, args.batch, lengths)
    t_nonce = time.time()

    qrecs = []
    for i, q in enumerate(qs):
        c = clean[i]
        rec = {"id": q["rid"], "qid": q["qid"], "type": q["type"], "label": q["label"], "keys": c["keys"],
               "label_scheme": c["label_scheme"], "nonces": nonces[i],
               "clean": {"pick": c["pick"], "p_max": c["p_max"], "probs": c["probs"], "correct": c["pick"] == q["label"],
                         "truncated": c["truncated"]},
               "variants": {}}
        if q["type"] == "score":
            rec["clean"]["expected"] = round(expected_level(c["keys"], c["probs"]), 6)
        for v in args.variant_list:
            passes = [nonce_res.get((i, v, k)) for k in range(args.repeats)]
            if any(p is None for p in passes):
                continue
            block = {"picks": [p["pick"] for p in passes], "p_max": [p["p_max"] for p in passes],
                     "tv": [round(tv_distance(p["probs"], c["probs"]), 6) for p in passes],
                     "correct": [p["pick"] == q["label"] for p in passes],
                     "flipped": any(p["pick"] != c["pick"] for p in passes),
                     "truncated": sum(int(p["truncated"]) for p in passes),
                     "probs": [p["probs"] for p in passes]}
            if q["type"] == "score":
                block["expected"] = [round(expected_level(p["keys"], p["probs"]), 6) for p in passes]
            rec["variants"][v] = block
        qrecs.append(rec)

    overall = {v: summarize(qrecs, v, args.repeats) for v in args.variant_list}
    for v in args.variant_list:
        overall[v]["skipped"] = skipped.get(v, 0) // args.repeats
    types = sorted({q["type"] for q in qrecs})
    by_type = {t: {v: summarize([q for q in qrecs if q["type"] == t], v, args.repeats) for v in args.variant_list}
               for t in types}

    set_name = os.path.basename(args.set).replace(".jsonl", "")
    here = os.path.dirname(os.path.abspath(__file__))
    meta = {
        "schema": 1, "stamp": time.strftime("%Y%m%d-%H%M%S"), "source": "train/nonce_eval.py",
        "label": args.label or f"{os.path.basename(adapter.rstrip('/')) if adapter else args.base} on {set_name}, nonce test",
        "base": args.base, "revision": args.revision, "adapter": adapter, "temperatures": temps,
        "attn": args.attn, "device": device, "gpu": gpu_name(), "node": platform.node(), "dtype": args.dtype,
        "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": version_of("transformers"),
        "peft": version_of("peft"), "fp32_candidate_logits": FP32_CANDIDATE_LOGITS, "toy": args.toy,
        "seed": args.seed, "repeats": args.repeats, "variants": args.variant_list, "batch": args.batch,
        "limit": args.limit, "max_tokens": args.max_tokens, "nonce_field": NONCE_FIELD,
        "set": {"id": set_name, "file": os.path.abspath(args.set), "sha256": sha256_file(args.set),
                "records": len(records), "questions": len(qs),
                "types": dict(collections.Counter(q["type"] for q in qs)),
                "license": records[0].get("license") if records else None},
        "sources": {f: sha256_file(os.path.join(here, f)) for f in SOURCE_FILES if os.path.exists(os.path.join(here, f))},
        "protocol": {
            "state": f"a fresh uuid4 per (question, pass) added to the state as top-level field {NONCE_FIELD!r}; "
                     "states that are not objects or already carry the field are skipped (counted)",
            "instr": "the same uuid appended to the question's instructions as a trailing line 'ref: <uuid>'",
            "both": "state and instr at once",
            "pick": "argmax with eval_jebadiah's tie rules (choice: first listed; score: lower level)",
            "prob_tv": "0.5 * L1 distance between the nonce and clean probability vectors",
            "p95": "nearest-rank at index int(0.95 * n) of the sorted values (eval_jebadiah's latency p95)",
            "accuracy": "pick == label, labels read the way eval_jebadiah reads them",
            "clean_order": "questions sorted by rendered length (stable), batches in that order: eval_jebadiah repeat 0",
        },
        "clean_check": check,
        "truncated_prompts": {"clean": sum(int(q["clean"]["truncated"]) for q in qrecs),
                              **{v: sum(q["variants"][v]["truncated"] for q in qrecs if v in q["variants"]) for v in args.variant_list}},
        "timing": {"load_s": round(t_loaded - t_start, 1), "clean_s": round(t_clean - t_loaded, 1),
                   "check_s": round(t_check - t_clean, 1), "nonce_s": round(t_nonce - t_check, 1),
                   "prompts_scored": len(qs) + len(nonce_res) + (check["questions"] if check else 0)},
    }
    out = {"meta": meta, "overall": overall, "by_type": by_type, "questions": qrecs}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, args.out)

    for v in args.variant_list:
        o = overall[v]
        line = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in o.items()
                if k in ("n_questions", "n_passes", "pick_agreement", "any_flip_rate", "p_max_abs_change_mean",
                         "p_max_abs_change_p95", "prob_tv_mean", "accuracy_clean", "accuracy_nonce", "skipped")}
        print(json.dumps({set_name: {v: line}}), flush=True)
    if len(types) > 1:
        print(json.dumps({"by_type": {t: {v: {k: (round(x, 4) if isinstance(x, float) else x)
                                              for k, x in by_type[t][v].items()
                                              if k in ("n_questions", "pick_agreement", "any_flip_rate", "accuracy_clean", "accuracy_nonce")}
                                          for v in args.variant_list} for t in types}}), flush=True)
    if check:
        print(json.dumps({"clean_check": {"questions": check["questions"], "max_abs_prob_diff": check["max_abs_prob_diff"]}}), flush=True)
    print("NONCE_DONE", flush=True)


if __name__ == "__main__":
    main()
