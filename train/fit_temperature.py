"""Fit one temperature per question type on the held-out calib slice by minimising NLL, and
report the ECE before and after. Saves {"temperatures": {type: T}, ...} next to the adapter,
which the scorer and the server read at load time.

The calib slice is the 5 percent of the training pool split off by split_pool.py; it trained
nothing and selected nothing. A temperature never changes a pick, only how sure it sounds.

v1: --target chooses what the NLL is taken against.
  hard   the argmax label (v0's fit). On a model trained towards soft or ordinal targets this
         sharpens: v0's score T came out 0.39 and made HelpSteer2 confidently wrong.
  train  the same target distribution the run trained towards (the source's soft target where
         it gives one, the ordinal kernel for score questions when the run used it), i.e. the
         held-out training loss. Both fits are always recorded; --target says which is applied.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_jebadiah import ece_10, label_key, pick  # noqa: E402
from jebadiah_model import Scorer, load_adapter, load_base, load_tokenizer  # noqa: E402
from train_jebadiah import make_target  # noqa: E402


def fit_one(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float]:
    """Grid then golden-section search on log T in [0.05, 20]. `target` is a distribution per
    row (one-hot for a hard label). Returns (T, nll_before, nll_after)."""
    def nll(t):
        z = (logits / t).masked_fill(~mask, float("-inf"))
        logp = torch.log_softmax(z, dim=-1).masked_fill(target == 0, 0.0)
        return -(target * logp).sum(dim=-1).mean().item()
    lo, hi = math.log(0.05), math.log(20.0)
    grid = [lo + (hi - lo) * i / 200 for i in range(201)]
    best = min(grid, key=lambda g: nll(math.exp(g)))
    a, b = best - (hi - lo) / 200, best + (hi - lo) / 200
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = b - phi * (b - a), a + phi * (b - a)
        if nll(math.exp(c)) < nll(math.exp(d)):
            b = d
        else:
            a = c
    t = math.exp((a + b) / 2)
    return t, nll(1.0), nll(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-revision")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--target", choices=["hard", "train"], default="hard", help="which fit is applied (both are recorded)")
    ap.add_argument("--score-targets", choices=["source", "ordinal"], default="source", help="the run's decide.score_targets")
    ap.add_argument("--score-adjacent", type=float, default=0.2, help="the run's decide.score_ordinal_adjacent")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    tok = load_tokenizer(args.base, args.base_revision)
    model = load_adapter(load_base(args.base, args.base_revision, attn_implementation=args.attn), args.adapter)
    scorer = Scorer(model, tok, args.max_tokens)
    records = [json.loads(l) for l in open(args.calib, encoding="utf-8") if l.strip()]
    items = [(r["state"], q, r["label"][qid], (r.get("target") or {}).get(qid)) for r in records for qid, q in r["questions"].items()]

    # raw candidate logits per type (log of the T=1 probabilities is the same thing up to a
    # constant, which the softmax removes)
    per_type = {}
    for i in range(0, len(items), args.batch_size):
        chunk = items[i:i + args.batch_size]
        rendered = [(scorer.render(s, q), q["type"]) for s, q, _, _ in chunk]
        probs = scorer.score_rendered(rendered)
        for (s, q, label, target), (rd, _), p in zip(chunk, rendered, probs):
            d = per_type.setdefault(q["type"], {"logits": [], "labels": [], "keys": [], "q": [], "train_target": []})
            d["logits"].append([math.log(max(x, 1e-12)) for x in p])
            d["labels"].append(rd.keys.index(label_key(q, label)))
            d["keys"].append(rd.keys)
            d["q"].append(q)
            d["train_target"].append(make_target(q, label, rd.keys, target, args.score_targets, args.score_adjacent))
    out = {"temperatures": {}, "applied_target": args.target, "calib_file": args.calib, "n": {},
           "fits": {"hard": {}, "train": {}}, "nll_before": {}, "nll_after": {},
           "ece_before": {}, "ece_after": {}, "accuracy": {},
           "score_targets": args.score_targets, "score_ordinal_adjacent": args.score_adjacent}
    for t, d in per_type.items():
        kmax = max(len(x) for x in d["logits"])
        n = len(d["logits"])
        lg = torch.full((n, kmax), -1e9)
        mask = torch.zeros((n, kmax), dtype=torch.bool)
        hard = torch.zeros((n, kmax))
        train = torch.zeros((n, kmax))
        for i, (x, tt) in enumerate(zip(d["logits"], d["train_target"])):
            lg[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = True
            hard[i, d["labels"][i]] = 1.0
            train[i, :len(tt)] = torch.tensor(tt)
        fits = {}
        for name, tgt in (("hard", hard), ("train", train)):
            T, before, after = fit_one(lg, tgt, mask)
            fits[name] = {"T": round(T, 4), "nll_before": round(before, 4), "nll_after": round(after, 4)}
            out["fits"][name][t] = fits[name]
        chosen = fits[args.target]
        T = chosen["T"]
        out["temperatures"][t] = T
        out["n"][t] = n
        out["nll_before"][t] = chosen["nll_before"]
        out["nll_after"][t] = chosen["nll_after"]
        for name, temp in (("ece_before", 1.0), ("ece_after", T)):
            confs, correct = [], []
            for i, (x, q, keys) in enumerate(zip(d["logits"], d["q"], d["keys"])):
                z = torch.softmax(torch.tensor(x) / temp, dim=-1).tolist()
                confs.append(max(z))
                correct.append(pick(q, keys, z) == keys[d["labels"][i]])
            out[name][t] = round(ece_10(confs, correct), 4)
        out["accuracy"][t] = round(sum(pick(q, k, torch.softmax(torch.tensor(x), -1).tolist()) == k[l]
                                       for x, q, k, l in zip(d["logits"], d["q"], d["keys"], d["labels"])) / n, 4)
    path = os.path.join(args.adapter, "temperatures.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("TEMPERATURES_DONE", path)


if __name__ == "__main__":
    main()
