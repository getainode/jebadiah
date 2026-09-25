"""Build the Hugging Face model card for a v1 run from its results.json and nonce records."""
import json, glob, os, sys
run, size, repo_id, prev_repo, image = sys.argv[1:6]   # e.g. 9b-v1 9B frontier-infra/jebadiah-9b-v1 frontier-infra/jebadiah-9b-v0 jeb-banner.png
R = f"results/runs/{run}"
d = json.load(open(f"{R}/results.json")); ev = d["eval"]; h = d["headline"]; tr = d["train"]
temps = json.load(open(f"{R}/adapter/temperatures.json"))["temperatures"]
cfg = d["config"]; base = cfg["base_model"]; rev = cfg["base_revision"]
mins = d["minutes"]
def row(name, label, kind):
    o = ev[name]; fl = o.get("flip_rate"); flt = "n/a" if fl is None else f"{fl*100:.1f}%"
    ece = f"{o['ece']:.3f}" + (f" (raw {o['ece_raw']:.3f})" if kind == "score" else "")
    return f"| {label} | {kind} | {o['accuracy']*100:.1f} | {o['majority_floor']*100:.1f} | {o['decision_score_jevals']:.1f} | {ece} | {flt} |"
rows = [row("jevals-pubmedqa", "Jevals PubMedQA (300)", "noul"), row("jevals-banking77", "Jevals Banking77 (300, 77 options)", "choice"),
        row("jevals-helpsteer2", "Jevals HelpSteer2 helpfulness (300, 5 levels)", "score"), row("nimble-eval", "Nimble held-out eval (324)", "mixed"),
        row("kev-transfer-v4__test", "Kev transfer-v4 test (764)", "mixed"), row("kev-decision-v7__test", "Kev decision-v7 test (1,440)", "mixed"),
        row("typed-decisions-test", "typed-decisions test (2,000)", "mixed")]
nim = h["nimble_public"]; nim_txt = ", ".join(f"{k.replace('nimble-public__','')} {v:.1f}" for k, v in nim.items())
# nonce summary (head sets, state and instr variants)
nonce = {}
for f in sorted(glob.glob(f"{R}/nonce/*.json") or glob.glob(f"results/nonce/{run}/*.json")):
    st = os.path.basename(f)[:-5]; ov = json.load(open(f)).get("overall", {})
    for var in ("state", "instr"):
        o = ov.get(var) or {}
        if o: nonce[(st, var)] = o
def nrow(st, label):
    a = nonce.get((st, "state")); b = nonce.get((st, "instr"))
    if not a or not b: return f"| {label} | n/a | n/a | n/a |"
    return f"| {label} | {a['pick_agreement']*100:.1f}% / {b['pick_agreement']*100:.1f}% | {a['any_flip_rate']*100:.1f}% / {b['any_flip_rate']*100:.1f}% | {a['p_max_abs_change_mean']*100:.1f} / {b['p_max_abs_change_mean']*100:.1f} |"
nrows = [nrow("jevals-pubmedqa", "Jevals PubMedQA"), nrow("jevals-banking77", "Jevals Banking77"), nrow("jevals-helpsteer2", "Jevals HelpSteer2"), nrow("nimble-eval", "Nimble 324")]
s4 = "" if size == "9B" else ("\n\nOne honest regression on the 4B: the Nimble 324 held-out set fell from 77.5 (v0) to 71.9 with the rubric "
      "data in the pool, while the 9B gained on it. The smaller model does not absorb 45 percent score questions without its choice and "
      "yes/no judgments slipping; v0 stays the better general 4B, v1 the better calibrated one. A lower rubric share or a higher rank is the next probe.")
img = (f'<img src="{image}" alt="They call me Jeb. He does not talk much. He just decides." width="100%">' if image.endswith("banner.png")
       else f'<p align="center"><img src="{image}" alt="They call me Jeb. He does not talk much. He just decides. A Texas-made original picture." width="420"></p>')
card = f"""---
base_model: {base}
library_name: peft
license: apache-2.0
language:
- en
tags:
- lora
- decision-model
- system-one
- calibrated-probabilities
- typed-decisions
- ainode
datasets:
- LocalLLaMA/typed-decisions
- nvidia/HelpSteer2
- mteb/summeval
model-index:
- name: {repo_id.split('/')[-1]}
  results:
  - task:
      type: text-classification
      name: typed decisions (choice, noul, score)
    dataset:
      name: Jevals suite 0.1.0, PubMedQA (noul)
      type: jevals-pubmedqa
    metrics:
    - type: accuracy
      value: {ev['jevals-pubmedqa']['accuracy']*100:.1f}
    - type: decision_score_jevals
      value: {ev['jevals-pubmedqa']['decision_score_jevals']:.1f}
  - task:
      type: text-classification
      name: typed decisions (choice, noul, score)
    dataset:
      name: Jevals suite 0.1.0, Banking77 (choice, 77-way)
      type: jevals-banking77
    metrics:
    - type: accuracy
      value: {ev['jevals-banking77']['accuracy']*100:.1f}
    - type: decision_score_jevals
      value: {ev['jevals-banking77']['decision_score_jevals']:.1f}
  - task:
      type: text-classification
      name: typed decisions (choice, noul, score)
    dataset:
      name: Jevals suite 0.1.0, HelpSteer2 helpfulness (score)
      type: jevals-helpsteer2
    metrics:
    - type: decision_score_jevals
      value: {ev['jevals-helpsteer2']['decision_score_jevals']:.1f}
  - task:
      type: text-classification
      name: typed decisions (choice, noul, score)
    dataset:
      name: Nimble public human-labelled subsets (13, macro accuracy)
      type: nimble-public
    metrics:
    - type: accuracy
      value: {h['components']['nimble-public (macro over 13)']:.1f}
---

# Jebadiah {size} v1

{img}

Jebadiah (Jeb for short) is an open System One style decision model: it answers typed questions with a
calibrated probability over the option labels instead of generating text. Three question types, the
same three Jev uses: **choice** (pick one of N), **noul** (a yes or no statement, returned as P(yes)) and
**score** (place the state on an ordered rubric). It is trained with AINode's trainer on public data only
and served by AINode's `/v1/decide` and `/v1/systemone` routes on any NVIDIA GPU. The routes answer Jev's
three question types; the request schema is AINode's own (documented on the route), not Jev's exact wire format.

This repository holds the **v1 {size}** adapter: a LoRA on `{base}` (pinned revision `{rev[:8]}`),
one epoch over 14,900 public training questions, trained on one H100 PCIe in {mins['train']:.0f} minutes.
v1 changes two things against [v0]({'https://huggingface.co/' + prev_repo}): score questions train
towards an ordinal target around the human label with the temperature fitted on that target, and the pool
carries two licensed human-rubric sources (HelpSteer2 train, SummEval). v1 is a fixed version; newer versions
are published as new repositories and never replace this one.

Made in Texas.

## Results

Measured by us with AINode's bench, one logit read per question, the same rendered prompt for every
model. Accuracy is the share of questions whose top label is the human label. Decision Score is Jevals'
metric: 100 = perfect, 0 = guessing the label base rates, below 0 = worse than that. These are our
numbers on the public suites, not rows on the Jevals board. Headline (macro over the zero-shot public
sets): **{h['headline']:.1f}** (v0: {'72.5' if size == '9B' else '70.9'}).

| Set (questions) | Type | Accuracy | Floor (majority label) | Decision Score (Jevals) | ECE, temperatures applied | Flips over identical repeats |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Nimble's 13 public human-labelled subsets (3,880 questions), macro accuracy: **{h['components']['nimble-public (macro over 13)']:.1f}**
(per subset: {nim_txt}). For scale, Bespoke's published table puts Nimble-9B at 74.8 and Jev at 76.0 on the
same subsets with their scorer.

Two things to read carefully. HelpSteer2 and SummEval are no longer zero-shot for Jeb: v1 trained on the
HelpSteer2 **train** split (the Jevals and Nimble items come from validation, and no item overlaps) and on
SummEval articles that Nimble does not score (its 15 evaluation articles are held out entirely), so those
rows are held-out items of a seen rubric. And the in-distribution score set (typed-decisions) gives back a
few points of Decision Score against v0, the cost of honest calibration on human rubrics.{s4}

Every eval record (per question: option keys, probabilities, pick, label, repeat, option order) is in
`eval/`, with `eval/results.json` carrying the full metric set.

## Robustness to irrelevant content (nonce test)

Every question on the four head sets was scored once clean and three more times with a fresh UUID planted
in the state (`state`) or appended to the instructions (`instr`). Records are in `eval/nonce/`.

| Set | Pick agreement (state / instr) | Questions with any flip (state / instr) | Mean change of p_max, points (state / instr) |
|---|---|---|---|
{chr(10).join(nrows)}

Choice and yes/no answers are close to immune. On score questions v1 flips its top level more often than
v0 while moving its probabilities far less: a calibrated rubric model sits honestly between adjacent
levels, so a small nudge tips the argmax. Read the expected level and the probabilities (which is what the
route returns as `score`), not the top pick, for rubric questions.

## How it decides

The prompt is AINode's own decide rendering (`ainode.api.decide.build_messages`, source commit
`e5c08938`, `prompt_source_sha256` in `prompt_contract.json`), through the base chat template with
thinking off. The option labels are single tokens; the model's answer is the distribution over those label
tokens at the last prompt position, read in fp32 from the last hidden state and then temperature scaled
per type (`temperatures.json`: choice {temps['choice']:.2f}, noul {temps['noul']:.2f}, score {temps['score']:.2f}). Nothing is generated.

The supported way to run it is AINode, which renders the prompt exactly as trained (a worked
`/v1/systemone` request and response is in the AINode docs issue tracker until the page lands):

```bash
curl -sS https://<your-ainode>/v1/systemone -H "Authorization: Bearer <key>" -H "Content-Type: application/json" -d '{{
  "model": "{repo_id.split('/')[-1].replace('jebadiah-', 'jebadiah/jebadiah-')}",
  "state": {{"ticket": "Customer says the invoice total does not match the quote."}},
  "questions": {{
    "route": {{"type": "choice", "instructions": "Which team should take this ticket?", "criteria": {{"billing": "an invoice, a charge or a refund", "support": "a product question", "sales": "a quote or a renewal"}}}},
    "urgent": {{"type": "noul", "instructions": "The customer is blocked from working.", "criteria": {{"true": "work has stopped", "false": "it can wait"}}}}
  }}
}}'
```

Standalone with transformers and peft, load the base at the pinned revision, then the adapter from this
repository, render the prompt per the contract, and read the label-token logits at the last position.

Thresholds belong to the caller: act on a high probability, confirm or escalate on a middle one, hand a
low one to a person or a bigger model. The model never refuses; policy is built from decisions.

## Training

- **Objective:** cross-entropy over the candidate option-label logits at the answer position. Choice and
  noul use the source's gold distribution where it provides one (typed-decisions) and the hard label
  otherwise; score questions use an ordinal kernel around the human label (adjacent level 0.2 of the
  label's weight), and the score temperature is fitted on that same target.
- **Adapter:** LoRA r=16, alpha=32, dropout 0.05 on every linear projection including the Gated DeltaNet
  projections, {tr['trainable_params']/1e6:.1f}M trainable parameters. bf16, SDPA attention, micro-batch 8, learning rate 1e-4,
  one epoch, {tr['steps']:,} steps, gradient checkpointing.
- **Data (public only), 15,813 questions in the pool:** `LocalLLaMA/typed-decisions` train (Apache-2.0,
  revision `ea930645`); the Kev v7 training sources whose licenses permit derived weights (BoolQ, MNLI,
  DBpedia14 and Kev's contrastive and composition sets, Kev suites revision `a88f56db`); `nvidia/HelpSteer2`
  train split (CC BY 4.0, 2,573 score questions: helpfulness in the Jevals task's exact shape for 1,609
  rows, plus correctness, coherence, complexity and verbosity on 241 of them); SummEval (MIT, via
  `mteb/summeval`, 1,664 score questions over four dimensions on 26 articles, with Nimble's 15 evaluation
  articles excluded). 95/5 family split into {tr['train_examples']:,} training and {tr['calib_examples']} calibration questions.
  Nimble's train set is excluded (no license stated); the Jevals test items are excluded from training.
  No private data of any kind.
- **Compute:** one NVIDIA H100 PCIe 80 GB, {mins['train']:.0f} min training, {mins['eval']:.0f} min evaluation, torch 2.11, PEFT 0.21.

## Limitations

- Single-hop judgments only. Split a chain of inference into hops.
- AINode caps a choice question at 20 options; Banking77's 77 options were scored with an extended
  single-token alphabet for the benchmark only.
- English data, plus Massive's German subset scoring well by accident of the base model.
- Calibration was fitted on the training distribution. Refit the temperatures on your own data before
  trusting a threshold, and remember the served route applies none yet.

## Versioning and license

v1 is frozen. Later versions land as `jebadiah-<size>-v<N>` repositories. The adapter is Apache-2.0,
the base model is Apache-2.0. Evaluation data: Jevals suite 0.1.0 (CC-BY-4.0, "Jevals (jevals.com),
release 2026-09-18"), Nimble public subsets (Bespoke Labs), Kev test sets and typed-decisions test,
each under its own license.
"""
out = f"hf-{repo_id.split('/')[-1]}"
os.makedirs(f"{out}/eval/nonce", exist_ok=True)
open(f"{out}/README.md", "w").write(card)
print("card written:", out, len(card), "chars; em dashes:", card.count("\u2014"))
