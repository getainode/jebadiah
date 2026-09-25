"""Evaluate a base model, or base plus adapter, on the test sets through the one logit read.

Per set it reports, every number with its contract:
  accuracy            share of questions whose pick equals the label (pick = argmax; choice ties
                      to the option listed first, score ties to the lower level), over repeat 0
  majority_floor      accuracy of always answering the set's commonest label
  decision_score_acc  (accuracy - floor) / (1 - floor): the share of the room between the floor
                      and 100 percent that the model took (0 at the floor, 100 perfect)
  decision_score_jevals  100 * (1 - L_model / L_prior), Jevals' definition: mean per-item
                      multiclass Brier (choice, noul) or ranked probability score (score) against
                      the label prior that answers with the set's base rates
  brier               multiclass Brier, mean over questions of sum_k (p_k - y_k)^2 (Jevals, Kev,
                      Nimble); brier_one_term is (1 - p_label)^2 (the AINode bench definition)
  ece                 expected calibration error on the pick's probability, 10 equal-width bins
                      (Jevals' binning), in [0, 1]; reported before and after temperature
  nll                 mean -log p(label)
  flip_rate           share of questions whose pick differs between any two of the 5 repeats of
                      the identical request; for the jevals sets repeats 2 to 4 use Jevals'
                      option orders and order_flip_rate is reported separately
  latency_ms_median   wall clock per question at batch 1 (a sample of the set), and
  qps_batch8          questions per second at batch size 8 over the whole set, on this box

Writes one JSON per set in the AINode bench-record shape with a top-level `decide` block.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import platform
import statistics
import subprocess
import time

import torch

from jebadiah_model import FP32_CANDIDATE_LOGITS, Scorer, load_adapter, load_base, load_tokenizer, read_temperatures
from jebadiah_prompt import wire_keys as answer_keys


# ---------------------------------------------------------------- Jevals' option-order shuffle

def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def mulberry32(seed: int):
    a = seed & 0xFFFFFFFF

    def imul(x, y):
        return ((x & 0xFFFFFFFF) * (y & 0xFFFFFFFF)) & 0xFFFFFFFF

    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = imul(t ^ (t >> 15), t | 1) & 0xFFFFFFFF
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rnd


def shuffled(xs: list, seed: int) -> list:
    a = list(xs)
    r = mulberry32(seed)
    for i in range(len(a) - 1, 0, -1):
        j = math.floor(r() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


def jevals_order(item_id: str, order_seed: int, options: list) -> list:
    return shuffled(options, fnv1a(f"{item_id}:{order_seed}"))


# ---------------------------------------------------------------- metrics

def label_key(q: dict, label) -> str:
    if q["type"] == "noul":
        return "true" if label else "false"
    return str(label)


def pick(q: dict, keys: list[str], probs: list[float]) -> str:
    """Argmax with the Jevals tie rules: choice -> first listed; score -> lower level."""
    if q["type"] == "score":
        order = sorted(range(len(keys)), key=lambda i: int(keys[i]))
        best = order[0]
        for i in order:
            if probs[i] > probs[best]:
                best = i
        return keys[best]
    best = 0
    for i in range(1, len(keys)):
        if probs[i] > probs[best]:
            best = i
    return keys[best]


def ece_10(confidences: list[float], correct: list[bool]) -> float:
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for c, ok in zip(confidences, correct):
        b = min(9, int(round(100 * c)) // 10)
        bins[b][0] += 1
        bins[b][1] += c
        bins[b][2] += float(ok)
    n = len(confidences)
    return sum((cnt / n) * abs(acc / cnt - conf / cnt) for cnt, conf, acc in bins if cnt) if n else float("nan")


def reliability(confidences: list[float], correct: list[bool]) -> list[dict]:
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for c, ok in zip(confidences, correct):
        b = min(9, int(round(100 * c)) // 10)
        bins[b][0] += 1
        bins[b][1] += c
        bins[b][2] += float(ok)
    return [{"lo": i / 10, "hi": (i + 1) / 10, "count": cnt,
             "confidence": (conf / cnt) if cnt else None, "accuracy": (acc / cnt) if cnt else None}
            for i, (cnt, conf, acc) in enumerate(bins)]


def item_loss(q: dict, keys: list[str], probs: list[float], label: str) -> float:
    """Multiclass Brier for choice and noul; ranked probability score for score."""
    if q["type"] == "score":
        order = sorted(range(len(keys)), key=lambda i: int(keys[i]))
        k = len(order)
        cp = cy = 0.0
        rps = 0.0
        for i in order[:-1]:
            cp += probs[i]
            cy += 1.0 if keys[i] == label else 0.0
            # cumulative label: 1 once we have passed the label level
            rps += (cp - cy) ** 2
        # cy above counts the label only at its level; the cumulative indicator is 1 from the
        # label level onward, which the loop realises because cy stays 1 after that level
        return rps / (k - 1)
    return sum((p - (1.0 if k == label else 0.0)) ** 2 for k, p in zip(keys, probs))


def untempered(probs: list[float], t: float) -> list[float]:
    """The probabilities the same logits give at temperature 1: p_fitted = softmax(z/T), so
    softmax(z) = normalize(p_fitted ** T)."""
    w = [max(p, 1e-12) ** t for p in probs]
    s = sum(w)
    return [x / s for x in w]


def summarize_set(rows: list[dict], q_of: dict, temperatures: dict | None = None) -> dict:
    """rows: one per question per repeat: {id, qid, repeat, keys, probs, pick, label}. When
    `temperatures` is given the rows carry fitted probabilities and the *_raw numbers are
    recovered from them at temperature 1 (the pick never changes)."""
    by_id = collections.defaultdict(dict)
    for r in rows:
        by_id[(r["id"], r["qid"])][r["repeat"]] = r
    r0 = [reps[0] for reps in by_id.values() if 0 in reps]
    n = len(r0)
    correct = [r["pick"] == r["label"] for r in r0]
    accuracy = sum(correct) / n
    labels = collections.Counter(r["label"] for r in r0)
    floor = labels.most_common(1)[0][1] / n
    ds_acc = (accuracy - floor) / (1 - floor) if floor < 1 else float("nan")
    confs = [max(r["probs"]) for r in r0]
    brier = statistics.mean(sum((p - (1.0 if k == r["label"] else 0.0)) ** 2 for k, p in zip(r["keys"], r["probs"])) for r in r0)
    brier_one = statistics.mean((1.0 - r["probs"][r["keys"].index(r["label"])]) ** 2 for r in r0)
    nll = statistics.mean(-math.log(max(r["probs"][r["keys"].index(r["label"])], 1e-12)) for r in r0)
    # Jevals Decision Score: mean per-item loss (averaged over repeats) against the label prior
    # of the evaluated items, per question type
    prior = {}
    for r in r0:
        t = q_of[(r["id"], r["qid"])]["type"]
        prior.setdefault(t, collections.Counter())[r["label"]] += 1
    l_model = l_prior = 0.0
    for (rid, qid), reps in by_id.items():
        q = q_of[(rid, qid)]
        rs = list(reps.values())
        l_model += statistics.mean(item_loss(q, r["keys"], r["probs"], r["label"]) for r in rs)
        keys = rs[0]["keys"]
        cnt = prior[q["type"]]
        tot = sum(cnt.values())
        pri = [cnt.get(k, 0) / tot for k in keys]
        l_prior += item_loss(q, keys, pri, rs[0]["label"])
    l_model /= len(by_id)
    l_prior /= len(by_id)
    ds_jevals = 100 * (1 - l_model / l_prior) if l_prior > 0 else float("nan")
    # flips
    flips = order_flips = 0
    n_multi = 0
    for reps in by_id.values():
        picks = [reps[k]["pick"] for k in sorted(reps)]
        if len(picks) > 1:
            n_multi += 1
            same_order = [reps[k]["pick"] for k in sorted(reps) if reps[k].get("order_seed", 0) == 0]
            if len(set(same_order)) > 1:
                flips += 1
            if len(set(picks)) > 1:
                order_flips += 1
    score_mae = None
    sc = [r for r in r0 if q_of[(r["id"], r["qid"])]["type"] == "score"]
    if sc:
        score_mae = statistics.mean(abs(sum(int(k) * p for k, p in zip(r["keys"], r["probs"])) - int(r["label"])) for r in sc)
    types = collections.Counter(q_of[(r["id"], r["qid"])]["type"] for r in r0)
    raw = {}
    if temperatures:
        rp = [untempered(r["probs"], float(temperatures.get(q_of[(r["id"], r["qid"])]["type"], 1.0))) for r in r0]
        raw = {"ece_raw": ece_10([max(p) for p in rp], correct),
               "brier_raw": statistics.mean(sum((x - (1.0 if k == r["label"] else 0.0)) ** 2 for k, x in zip(r["keys"], p)) for r, p in zip(r0, rp)),
               "nll_raw": statistics.mean(-math.log(max(p[r["keys"].index(r["label"])], 1e-12)) for r, p in zip(r0, rp)),
               "bins_raw": reliability([max(p) for p in rp], correct)}
    return {
        "n": n, "types": dict(types), "accuracy": accuracy, "majority_floor": floor, **raw,
        "decision_score_acc": ds_acc, "decision_score_jevals": ds_jevals,
        "brier": brier, "brier_one_term": brier_one, "nll": nll,
        "ece": ece_10(confs, correct), "bins": reliability(confs, correct),
        "score_mae": score_mae,
        "repeats": max(len(v) for v in by_id.values()),
        "flip_rate": (flips / n_multi) if n_multi else None,
        "order_flip_rate": (order_flips / n_multi) if n_multi else None,
    }


# ---------------------------------------------------------------- running a set

def run_set(scorer: Scorer, records: list[dict], repeats: int, jevals_orders: bool, batch_size: int,
            latency_sample: int) -> tuple[list[dict], dict]:
    q_of = {}
    items = []  # (record id, qid, state, q, label, repeat, order_seed)
    for r in records:
        for qid, q in r["questions"].items():
            q_of[(r["id"], qid)] = q
            for rep in range(repeats):
                seed = 0
                if jevals_orders and q["type"] == "choice":
                    seed = [0, 0, 1, 2, 3][rep] if rep < 5 else rep - 1
                items.append((r["id"], qid, r["state"], q, r["label"][qid], rep, seed))
    # each repeat scores the questions in a different order, so batch neighbours (and padding)
    # differ between repeats the way they do on a live server; a pick that survives that is stable
    import random
    order_rng = random.Random(20260922)
    per_rep = {}
    for it in items:
        per_rep.setdefault(it[5], []).append(it)
    # length-bucketed: sort each repeat by rendered length so a batch pads little, then shuffle
    # inside buckets of 64 for repeats > 0 so batch neighbours still differ between repeats
    def plen(it):
        return len(scorer.tok.encode(scorer.render(it[2], it[3]).prompt, add_special_tokens=False))
    lengths = {}
    ordered = []
    for rep in sorted(per_rep):
        lst = per_rep[rep]
        for it in lst:
            key = (it[0], it[1])
            if key not in lengths:
                lengths[key] = plen(it)
        lst.sort(key=lambda it: lengths[(it[0], it[1])])
        if rep > 0:
            for i in range(0, len(lst), 64):
                chunk = lst[i:i + 64]
                order_rng.shuffle(chunk)
                lst[i:i + 64] = chunk
        ordered += lst
    items = ordered
    rows = []
    truncated = 0
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        rendered = []
        for rid, qid, state, q, label, rep, seed in chunk:
            order = None
            if jevals_orders and q["type"] == "choice" and seed:
                item_id = rid.split(":", 1)[1]
                order = jevals_order(item_id, seed, answer_keys(q))
            rd = scorer.render(state, q, order)
            truncated += int(rd.truncated)
            rendered.append((rd, q["type"]))
        probs = scorer.score_rendered(rendered)
        for (rid, qid, state, q, label, rep, seed), (rd, _), p in zip(chunk, rendered, probs):
            rows.append({"id": rid, "qid": qid, "repeat": rep, "order_seed": seed, "keys": rd.keys,
                         "probs": [round(x, 6) for x in p], "pick": pick(q, rd.keys, p),
                         "label": label_key(q, label), "label_scheme": rd.label_scheme})
    torch.cuda.synchronize()
    batch_s = time.time() - t0
    rows.sort(key=lambda r: (r["id"], r["qid"], r["repeat"]))
    # single-question latency on a sample
    lat = []
    for rid, qid, state, q, label, rep, seed in items[:latency_sample]:
        torch.cuda.synchronize()
        t = time.time()
        scorer.score_rendered([(scorer.render(state, q), q["type"])])
        torch.cuda.synchronize()
        lat.append((time.time() - t) * 1000)
    timing = {"questions_scored": len(items), "batch_size": batch_size, "wall_s": round(batch_s, 2),
              "qps_batch": round(len(items) / batch_s, 2),
              "latency_ms_median_batch1": round(statistics.median(lat), 1) if lat else None,
              "latency_ms_p95_batch1": round(sorted(lat)[min(len(lat) - 1, int(0.95 * len(lat)))], 1) if lat else None,
              "latency_sample": len(lat), "truncated_prompts": truncated}
    return rows, {"q_of": q_of, "timing": timing}


def box_info() -> dict:
    try:
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = None
    return {"node": platform.node(), "gpu": gpu, "torch": torch.__version__,
            "cuda": torch.version.cuda}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-revision")
    ap.add_argument("--adapter", help="adapter dir; omit for the base model alone")
    ap.add_argument("--temperatures", choices=["none", "adapter"], default="adapter",
                    help="apply the temperatures saved with the adapter (default) or none")
    ap.add_argument("--sets", nargs="+", required=True, help="test JSONL files")
    ap.add_argument("--out", required=True, help="directory for the per-set records")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--repeats-for", default="", help="per-set override, e.g. 'jevals-=5,nimble-public=2'")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--latency-sample", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--label", default="")
    ap.add_argument("--limit", type=int, help="records per set (smoke tests)")
    ap.add_argument("--memory-cap-gb", type=float, default=0, help="cap the CUDA allocator (0 = none)")
    ap.add_argument("--attn", default="sdpa", help="attention implementation for the base model (sdpa, flash_attention_2, eager)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # keep the caching allocator inside the memory ceiling on this shared box: it raises instead
    # of growing past the cap, and the cache is dropped between sets
    if args.memory_cap_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.95, args.memory_cap_gb * 1e9 / total), 0)
    tok = load_tokenizer(args.base, args.base_revision)
    model = load_base(args.base, args.base_revision, attn_implementation=args.attn)
    if args.adapter:
        model = load_adapter(model, args.adapter)
    temps = read_temperatures(args.adapter) if (args.adapter and args.temperatures == "adapter") else {}
    scorer = Scorer(model, tok, args.max_tokens, temperatures=temps)
    model_id = (os.path.basename(args.adapter.rstrip("/")) if args.adapter else args.base)
    summary = {}
    for path in args.sets:
        set_name = os.path.basename(path).replace(".jsonl", "")
        records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if args.limit:
            records = records[:args.limit]
        jev = set_name.startswith("jevals-")
        repeats = args.repeats
        for spec in filter(None, args.repeats_for.split(",")):
            prefix, n = spec.split("=")
            if set_name.startswith(prefix):
                repeats = int(n)
        rows, extra = run_set(scorer, records, repeats, jev, args.batch_size, args.latency_sample)
        stats = summarize_set(rows, extra["q_of"], temps or None)
        # per-subset breakdown (repeat 0 accuracy and floor)
        subsets = collections.defaultdict(list)
        sub_of = {r["id"]: r.get("subset", "") for r in records}
        for r in rows:
            if r["repeat"] == 0:
                subsets[sub_of[r["id"]]].append(r)
        per_subset = {}
        for s, rs in sorted(subsets.items()):
            c = collections.Counter(r["label"] for r in rs)
            acc = sum(r["pick"] == r["label"] for r in rs) / len(rs)
            per_subset[s] = {"n": len(rs), "accuracy": acc, "majority_floor": c.most_common(1)[0][1] / len(rs)}
        record = {
            "schema": 1, "stamp": time.strftime("%Y%m%d-%H%M%S"),
            "label": args.label or f"{model_id} on {set_name}",
            "model": {"id": model_id, "base": args.base, "base_revision": args.base_revision,
                      "adapter": args.adapter, "temperatures": temps},
            "placement": {"node": box_info()["node"], **box_info()},
            "settings": {"sets": [set_name], "repeats": repeats, "batch_size": args.batch_size,
                         "max_tokens": args.max_tokens, "jevals_option_orders": jev,
                         "items_file": path, "items": len(records)},
            "decide": {
                "backend": "jebadiah-logit-read", "fp32_candidate_logits": FP32_CANDIDATE_LOGITS, "attn": args.attn,
                "item_set": {"id": set_name, "file": path,
                                                               "count": len(records),
                                                               "license": records[0].get("license") if records else None},
                "protocol": {"bins": 10, "confidence": "the probability of the pick (for noul the larger of P(true), P(false))",
                             "brier": "multiclass sum_k (p_k - y_k)^2; brier_one_term is (1 - p_label)^2",
                             "decision_score_acc": "(accuracy - majority_floor) / (1 - majority_floor)",
                             "decision_score_jevals": "100 * (1 - L_model / L_prior), L = mean per-item Brier (choice, noul) or RPS (score) averaged over repeats, prior = the set's label base rates per type",
                             "flip_rate": "share of questions whose pick differs across identical repeats (order_seed 0)",
                             "order_flip_rate": "share of questions whose pick is not the same across all repeats (Jevals orders on the jevals sets)"},
                "overall": stats, "subsets": per_subset, "timing": extra["timing"],
                "rows": rows,
            },
            "notes": ([f"{sum(1 for r in rows if r['label_scheme'] == 'extended') // max(repeats, 1)} questions used the extended "
                       "single-token label alphabet (more options than AINode's option_label range); the served route cannot carry them"]
                      if any(r["label_scheme"] == "extended" for r in rows) else []),
            "source": "train/eval_jebadiah.py",
        }
        out_path = os.path.join(args.out, f"{set_name}.json")
        json.dump(record, open(out_path, "w"), indent=1)
        torch.cuda.empty_cache()
        summary[set_name] = {k: stats.get(k) for k in ("n", "accuracy", "majority_floor", "decision_score_acc",
                                                      "decision_score_jevals", "brier", "ece", "ece_raw", "nll", "flip_rate",
                                                      "order_flip_rate", "score_mae")}
        summary[set_name]["qps_batch"] = extra["timing"]["qps_batch"]
        summary[set_name]["latency_ms_median_batch1"] = extra["timing"]["latency_ms_median_batch1"]
        print(json.dumps({set_name: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary[set_name].items()}}), flush=True)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
