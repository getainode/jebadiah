# Jebadiah

Jebadiah (Jeb for short) is an open decision model. It answers small typed questions about a piece of structured
state, and instead of writing text it returns a calibrated probability over the allowed answers. This repository
holds everything needed to reproduce it: the trainer and evaluator, the data builders, the evaluation harnesses, the
configuration of every run, and the result record of every run we report.

We train it with [AINode](https://github.com/getainode/ainode)'s trainer on public data and serve it through
AINode's `/v1/decide` and `/v1/systemone` routes. The published weights live on Hugging Face under
[frontier-infra](https://huggingface.co/frontier-infra) (all of them in one
[collection](https://huggingface.co/collections/frontier-infra/jebadiah-open-system-one-decision-models-6ab80765ddd3fa0b3eba5213)).
The project site is [jebadiah.ai](https://jebadiah.ai).

## What a System One decision model is

Most software that uses a language model today asks it to write a paragraph and then parses the paragraph. A lot
of that traffic is really a decision: which team should take this ticket, does this reply answer the question, how
severe is this alert. A System One model answers those directly, in one forward pass, the way fast intuitive
judgment works in people. Its consumer is code, not a person.

That changes what matters. The answer must be one of the allowed options, every time. The probability must mean
something, so the caller can act on a high one, escalate a middle one and hand a low one to a person or a larger
model. The same question must get the same answer, under load and when irrelevant noise in the input changes. And
it must be cheap enough to run on every event. Jebadiah reads its answer from the logits of single label tokens at
one position, so there is no generation, no parsing and no refusal path; policy is built by the caller from the
probabilities.

## The three question types

A request carries a `state` (any JSON) and one or more questions. Each question has a type, an instruction and
criteria.

- **choice**: pick one of N named options. Returns a probability for every option.
- **noul**: a yes or no statement. Returns P(true).
- **score**: place the state on an ordered rubric (for example five helpfulness levels). Returns a probability for
  every level; the expected level is the useful summary.

```json
{
  "state": {"ticket": "Customer says the invoice total does not match the quote."},
  "questions": {
    "route": {"type": "choice", "instructions": "Which team should take this ticket?",
              "criteria": {"billing": "an invoice, a charge or a refund", "support": "a product question",
                           "sales": "a quote or a renewal"}},
    "urgent": {"type": "noul", "instructions": "The customer is blocked from working.",
               "criteria": {"true": "work has stopped", "false": "it can wait"}}
  }
}
```

## How training works

- **Prompt contract.** Every question is rendered with AINode's own decide prompt (`train/ainode_prompt_verbatim.py`
  is a verbatim copy of `ainode.api.decide.build_messages` at source commit `e5c08938`, hash-pinned), through the
  base model's chat template with thinking off. Options are shown as single-token labels (A, B, C ... and up to 68
  labels; `train/jebadiah_prompt.py`). The same bytes are used at training, evaluation and serving time;
  `train/test_prompt_identity.py` checks the copy against an AINode checkout.
- **Objective.** Cross-entropy over the candidate label logits at the answer position only
  (`train/train_jebadiah.py`). Choice and noul train toward the source's gold distribution where it has one, else the
  hard label. Score questions train toward an ordinal kernel around the human label (the adjacent level gets 0.2 of
  the label's weight). Nothing else is supervised and nothing is generated. The candidate logits are computed in
  fp32 from the last hidden state (`train/jebadiah_model.py`).
- **Adapter.** LoRA rank 16, alpha 32, dropout 0.05 on every linear projection including the Gated DeltaNet ones:
  32.5M trainable parameters on the 4B, 43.3M on the 9B and 116.7M on the 27B. One epoch, learning rate 1e-4 with a
  cosine schedule, 30 warmup steps, micro-batch 8, bf16, seed 17. Every setting is in `configs/sweep.json`, except
  the 27B's, which are in its run record.
- **Calibration.** A 5 percent calibration slice is split off by family hash (`data/split_pool.py`), so sibling
  records never straddle it. After training, one temperature per question type is fitted on it
  (`train/fit_temperature.py`); the published temperatures are in `configs/temperatures/`.
- **Evaluation.** `train/eval_jebadiah.py` scores 20 test sets with repeats and option-order shuffles, and records
  accuracy, the Jevals Decision Score, Brier, ECE, flip rates and latency for every row. `train/postprocess.py` and
  `train/sweep_lib.py` roll that up into the run's `results.json`. The **headline** is the macro accuracy over five
  zero-shot sets (Jevals PubMedQA, Banking77 and HelpSteer2, Nimble's 324-question eval set, Kev transfer v4) plus the
  macro over Nimble's 13 public human-labelled subsets counted as one component. The two sets that share sources
  with the training pool (typed-decisions test, Kev decision v7 test) are reported beside it, not in it.
- **Data.** Public, license-checked sources only: `LocalLLaMA/typed-decisions`, the Kev v7 training sources whose
  licenses permit derived weights (BoolQ, MNLI, DBpedia14 and Kev's contrastive and composition sets), HelpSteer2
  train and SummEval, with every evaluation item excluded. `data/convert_data.py` downloads and converts them; every
  source, revision, license and exclusion is recorded in `data/manifests/convert-data-v1.json`. Later pools add our
  synthetic set and public human yes/no sets (see the experiments).

## Published versions

Current v2.1 releases and the maintained board snapshot are in [Decision Index v0.4](#decision-index-v04). The following table retains the earlier experiments and their original result records.

| Model | Base | Pool | Headline | Jevals PubMedQA | Jevals Banking77 | Jevals HelpSteer2 | Nimble 324 | Kev transfer | Nimble 13 macro | DSj HelpSteer2 | Record |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [jebadiah-27b](https://huggingface.co/frontier-infra/jebadiah-27b) | Qwen3.8-27B (chat, thinking off) | v1 | **78.95** | 90.0 | 77.0 | 48.7 | 93.5 | 85.9 | 78.6 | 19.9 | [`27b-chat-v1`](results/runs/27b-chat-v1/results.json) |
| [jebadiah-9b-v2](https://huggingface.co/frontier-infra/jebadiah-9b-v2) | Qwen3.5-9B (chat, thinking off) | v1 | **73.93** | 90.3 | 70.7 | 40.7 | 81.2 | 83.8 | 77.0 | 10.4 | [`9b-chat-v1`](results/runs/9b-chat-v1/results.json) |
| [jebadiah-4b-v2](https://huggingface.co/frontier-infra/jebadiah-4b-v2) | Qwen3.5-4B (chat, thinking off) | v1 | **72.49** | 88.7 | 70.0 | 40.0 | 77.2 | 83.2 | 75.9 | 9.3 | [`4b-chat-v1`](results/runs/4b-chat-v1/results.json) |
| [jebadiah-9b-v1](https://huggingface.co/frontier-infra/jebadiah-9b-v1) | Qwen3.5-9B-Base | v1 | 73.29 | 89.7 | 70.0 | 40.3 | 78.7 | 84.0 | 77.0 | 11.9 | [`9b-v1`](results/runs/9b-v1/results.json) |
| [jebadiah-4b-v1](https://huggingface.co/frontier-infra/jebadiah-4b-v1) | Qwen3.5-4B-Base | v1 | 70.34 | 86.7 | 68.3 | 37.0 | 71.9 | 82.2 | 75.9 | 9.2 | [`4b-v1`](results/runs/4b-v1/results.json) |
| [jebadiah-9b-v0](https://huggingface.co/frontier-infra/jebadiah-9b-v0) | Qwen3.5-9B-Base | v0 | 72.53 | 89.0 | 71.0 | 40.3 | 76.9 | 83.0 | 75.0 | -21.4 | [`9b-v0`](results/runs/9b-v0/results.json) |
| [jebadiah-4b-v0](https://huggingface.co/frontier-infra/jebadiah-4b-v0) | Qwen3.5-4B-Base | v0 | 70.94 | 87.7 | 68.7 | 36.3 | 77.5 | 82.5 | 73.0 | -20.2 | [`4b-v0`](results/runs/4b-v0/results.json) |
| untrained 9B base, for reference | Qwen3.5-9B-Base | | 68.70 | 86.7 | 69.7 | 42.0 | 69.1 | 74.6 | 70.1 | 9.6 | [`9b-base-eval`](results/runs/9b-base-eval/results.json) |

Accuracy in percent (repeat 0). DSj is the Jevals Decision Score: 100 is perfect, 0 is guessing the base rates.
The DSj of the 27B, 9B v2 and 4B v2 is each run's record, made with the score temperature it shipped with; with the
refits described below they are 21.3, 10.5 and 9.4, and no accuracy changes. All numbers are ours, measured with
the evaluator in this repository. Every
run, including the ones we did not publish, is in [`results/SUMMARY.md`](results/SUMMARY.md), generated from the records by `scripts/results_table.py`.
Per-question evaluation rows are not committed here (about 13 MB per run); the published model repositories carry
them for their own runs under `eval/`.

## The story, v0 to the 27B

**v0** was the recipe: LoRA on the Qwen3.5 Base checkpoints, candidate-label cross-entropy, public data only, one
epoch. It added about four headline points over the untrained 9B, and two epochs were worse than one. Its weak spot
was the score type: on human helpfulness ratings it was confidently wrong, with a Decision Score of -21 on HelpSteer2.
([01](experiments/01-v0-recipe.md))

**v1** added human-rated score data (HelpSteer2 train and SummEval) and changed the score target from the teacher's
spread to an ordinal kernel around the human label, with the temperature fitted on that same target. The 9B's
HelpSteer2 Decision Score went from -21.4 to 11.9 and its calibration error there from 0.39 to 0.045. Its rubric
accuracy did not move: the model became honest about not knowing helpfulness rather than better at judging it. The 4B
paid for the extra score data with 5.6 points on Nimble's set. ([02](experiments/02-v1-rubric-data.md))

**Between v1 and v2** we tried to grow the data. A 14,714-question synthetic pool authored and labelled by our own
teacher models made the 9B worse ([03](experiments/03-v2-synthetic-pool.md)). A fifth of synthetic choice questions,
with no synthetic yes/no, helped the 4B a little ([04](experiments/04-choice-dose.md)). A teacher bake-off against a
human answer key showed the open teachers' labels were 12 to 13 points below a frontier model's
([05](experiments/05-teacher-bakeoff.md)), but re-targeting the synthetic pool with the best teacher did not move our
headline ([06](experiments/06-fable-targets.md)). Public human yes/no data and a small slice of our own human labels did
not lift the human yes/no evaluation sets either, and on the 9B the smaller score share cost HelpSteer2
([07](experiments/07-human-yes-no.md)). None of these beat v1.

**v2** came from the one change we had not tried: the starting checkpoint.

## The chat-checkpoint finding

The same v1 recipe and the same v1 pool, trained on the Qwen3.5 chat checkpoints with thinking off instead of the
Base checkpoints, gave the best 4B and the best 9B we have. On the 4B it is worth 2.15 headline points (72.49 against
70.34) with no new data, more than any data change we made, and it removes 4B v1's regression on Nimble's set (77.2
against 71.9). On the 9B the gain is 0.64 points (73.93 against 73.29), most of it on Nimble's set; the other
components move less than a point and the HelpSteer2 Decision Score is lower (10.4 against 11.9). No record carries a
significance test, so we treat the 4B gain as real and the 9B gain as small. The two chat runs were merged into the
base weights and published as v2. ([08](experiments/08-chat-checkpoint.md))

Two other properties held across versions. Answers are stable under noise: with a random id planted in the state,
trained models keep 96 to 100 percent of their choice and yes/no picks, and the untrained 4B loses 6 accuracy points
on Banking77 where the trained one loses none ([09](experiments/09-nonce.md)). And the model is honest about rubric
questions it cannot answer well, which is what calibration is for.

## The 27B

The next step was the same recipe on a bigger and newer base: the v1 pool, the same LoRA settings and one epoch, on
the Qwen3.8-27B chat checkpoint with thinking off instead of Qwen3.5-9B. It is the best model we have, 78.95 headline
against 9B v2's 73.93, about five points. Most of the gain is on Nimble's set (93.5 against 81.2), Banking77 (77.0
against 70.7) and HelpSteer2 accuracy (48.7 against 40.7); Jevals PubMedQA is flat (90.0 against 90.3) and four of
the Nimble public subsets are lower. The base is the only change, but Qwen3.8-27B is both three times larger and a
newer generation than Qwen3.5-9B, and we did not run the experiment that would separate the two, so we cannot say how
much of the gain is size. It is one run with one seed, trained on one H200 in 167 minutes; its settings are in the
record (`results/runs/27b-chat-v1/`), and it was merged into the base weights and published as
[jebadiah-27b](https://huggingface.co/frontier-infra/jebadiah-27b).

**The calibration refit.** Every run fits one temperature per question type on the calibration slice, against the
run's own training target. For score questions that target is the ordinal kernel, which is smoothed on purpose, so
the fit calibrated to the smoothing rather than to the label: the 27B shipped a score temperature of 1.14 that
softened score answers which were already too soft. On 2026-09-26 the 27B's score temperature was refit against the
label on the same held-out calibration slice, 0.76 (it now sharpens). Choice (1.23) and noul (1.30) keep their
training-target fits, because fitting those to the label made the public sets worse; one temperature per type cannot
serve both our in-distribution sets and the public ones. Nothing was fitted on an evaluation set, and a two-fold
cross-fit on the evaluation sets lands within 0.001 of the result. A temperature never changes a pick, so accuracy and
the headline are unchanged. Over the 20 evaluation sets, question-weighted ECE goes from 0.081 to 0.072, macro ECE
from 0.071 to 0.064, and NLL from 0.523 to 0.514. Per set it is not a clean win: of the 8 sets with score questions,
5 improve (Jevals HelpSteer2 0.078 to 0.062, Nimble 324 0.145 to 0.124, SummEval consistency 0.165 to 0.109), Kev
transfer is unchanged, and 2 get worse (Kev decision-v7 0.027 to 0.040, Nimble public HelpSteer2 0.049 to 0.061).
Nimble 324 stays above its untempered 0.103, because what is left there is in-distribution choice, which this does
not touch. The temperatures are fitted on our training distribution; anyone thresholding on them should refit on
their own traffic.

On 2026-09-29 the 9B v2 and 4B v2 got the same rule: score takes the fit against the label (0.83 for both, from 1.22
and 1.20), choice and noul keep their training-target fits, and nothing is retrained. On held-out halves of the
calibration slice, score ECE goes from 0.119 to 0.079 (9B) and 0.105 to 0.066 (4B). Over the 20 evaluation sets,
question-weighted ECE goes from 0.080 to 0.071 (9B) and 0.086 to 0.080 (4B), NLL goes down on both, and no pick
changes. Typed-decisions improves (0.156 to 0.121, 0.171 to 0.135) and so does Nimble 324 (0.085 to 0.070, 0.092 to
0.071); both HelpSteer2 sets get worse (Jevals 0.039 to 0.082 and 0.048 to 0.074), because the calibration slice's
HelpSteer2 labels are close to uniform and natural HelpSteer2 traffic puts over 70% of its labels at 3 or 4. Each
model card and its `eval/RESULTS.md` carry the per-set numbers.

A held-out check on 2026-09-29 says the label mix is only part of it. On the 418 HelpSteer2 validation rows that
neither Jevals nor Nimble uses (natural label mix, never in the training pool), the score answers want a temperature of
1.26 (4B) and 1.29 (9B), and still 1.08 and 1.11 at a flat label mix, where the calibration slice's own HelpSteer2
questions want about 0.8. If your traffic looks like HelpSteer2, keep the older score temperature (1.20 on the 4B, 1.22
on the 9B, the `train` fit in `temperatures.json`) or refit on your own labels. v3's calibration split will draw
HelpSteer2 from held-out data.

## Where we stand on JevBench

[JevBench](https://github.com/fstandhartinger/jevbench) (v1.4.2) is a public benchmark for exactly this kind of model.
We ran its 231 public items through its own unchanged harness ([10](experiments/10-jevbench.md)).

| System | Public accuracy (231) | Hard tier (111) | Hard-tier ECE |
|---|---:|---:|---:|
| Jebadiah 27B | 0.866 | 82 | 0.113 |
| Jebadiah 9B v2 | 0.818 | 70 | 0.052 |
| Jebadiah 9B v1 | 0.818 | 71 | 0.127 |
| Jebadiah 4B v2 | 0.758 | 57 | 0.088 |
| Jebadiah 4B v1 | 0.762 | 56 | 0.087 |
| JPT-4B (self-reported) | 0.879 | 87 | |
| Jev 1.13.0 (board) | 0.866 | 81 | |
| Hopper (board) | 0.823 | | |

Candidly: the 27B matches Jev 1.13.0 on the board's public items (0.866, with 82 of the 111 hard items against Jev's
81) and is above the other board rows we compared with in [10](experiments/10-jevbench.md); only JPT-4B's
self-reported 0.879 is higher. The 9B and 4B are below every leading model there. The easy and standard tiers are at
the ceiling for every size; the gap is the hard tier, where the models most often take the planted surface answer
(21 of the 27B's 29 hard misses), above all on temporal and numeric items, where the 27B gets 4 of 15, no better than
9B v1. The chat checkpoint made the 9B much better calibrated where it is wrong; the 27B is not there yet, with a
hard-tier ECE of 0.113 against 9B v2's 0.052, so it is right more often but less honest about the items it misses.
These are self-run public numbers; the board's ranking also uses sealed items that only its maintainer runs, where
every top-ten row drops 48 to 57 points, and we have not been measured there. Nothing in this repository was tuned
toward JevBench, and an overlap check found no shared 8-word span between our training pool (the v1 pool, which the
27B also trained on) and its public items.

## Results at a glance

Current snapshot: Decision Index 0.4.1, generated **2026-10-21T22:34:35Z**, with 223 entrants plus the Jev reference. These ranks describe this dated full-weight T1e snapshot.

**Same released v2.1 weights, new T1e inference recipe.** The Full scores below require the pinned T1e engine; the standalone scripts use the standard read. Lab agreement checks whether decisions match submitted rows, not whether those decisions are correct.

| Full-weight T1e entrant | Full (points) | Overall position (rows) | Displayed tied rank (groups) | Lab median (ms/request) | Lab decision agreement (%) | Primary source / caveat |
|---|---:|---:|---:|---:|---:|---|
| [Jebadiah 27B](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1) | 64.4 | 25 | =5 | 303.2 | 99.90 | [Board snapshot, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/index.json); rank_v is among 224 v03.json rows, including Jev. Jev is not a row in index.json. Median is lab GPU timing. |
| [Jebadiah 9B](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1) | 50.9 | 79 | =17 | 125.1 | 100.00 | [Board snapshot, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/index.json); same lab method and scope. |
| [Jev reference](https://multimodalart-jev-decision-index.static.hf.space/index.html?model=jev) | 60.24 | 41 | =9 | Not compared | Not reported here | [Full-score ranks](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json); hosted service timing is not compared with local GPU timing. |

| Lab check | Value | Unit | Primary source / caveat |
|---|---:|---|---|
| Submitted-row agreement sample, each entrant | 831 | rows | [Lab notes](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/index.json); decision agreement, not task accuracy. |
| Lab agreement acceptance threshold | 98.5 | percent | [Lab notes](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/index.json); applies to sampled decision agreement. |
| Answered, each entrant | 110201 / 110201 | questions | [Board counts](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/index.json); coverage is not correctness. |

Snapshot generated 2026-10-21T22:34:35Z. Overall position counts rows; tied rank counts score groups. The lab used an RTX PRO 6000 and sequential requests in a single process. These figures do not establish CPU, Mac, GGUF or MLX scores or speed. Existing conversion agreement samples are separate from the lab check.

## Decision Index 0.4.1 (formula v0.4)

[Live board](https://multimodalart-jev-decision-index.static.hf.space/index.html) · [Jebadiah 27B report](https://multimodalart-jev-decision-index.static.hf.space/index.html?model=jebadiah-27b-v2.1-t1e) · [Jebadiah 9B report](https://multimodalart-jev-decision-index.static.hf.space/index.html?model=jebadiah-9b-v2.1-t1e) · [Jev reference](https://multimodalart-jev-decision-index.static.hf.space/index.html?model=jev).
This snapshot uses the published Full score. Public-only results are displayed separately.
These results require the T1e engine recipe. Downloading the checkpoint and running the standalone script uses the standard read.

| Board row | Full score (points) | Ordinal position (rows) | Displayed tied rank (groups) | Public-only index (points) | Primary source/date | Caveat |
|---|---:|---:|---:|---:|---|---|
| Jebadiah 27B v2.1 (t1e) | 64.4 | 25 | =5 | 60.46 | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Full-board snapshot, including Jev; not a ranking outside this board. |
| Jev 1.14 | 60.24 | 41 | =9 | 60.70 | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Full-board snapshot, including Jev; not a ranking outside this board. |
| Jebadiah 9B v2.1 (t1e) | 50.9 | 79 | =17 | 51.53 | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Full-board snapshot, including Jev; not a ranking outside this board. |

| Snapshot/ranking rule | Value | Unit | Primary source/date | Caveat |
|---|---:|---|---|---|
| Open entrant rows | 223 | models | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Jev is an additional reference row. |
| Rows in index.json | 223 | rows | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/index.json) | Engine jev is absent from index.json. This is a file count, not a filtered rank. |
| Tie band from each group's leading score | 0.9 | points | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Groups take successive ranks; this is not the ordinal row position or a significance claim. |
| Snapshot generated | 2026-10-21T22:34:35Z | UTC timestamp | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | This commit only; a later commit can move positions. |
| Scoring formula decided | 2026-10-19 | UTC date | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Formula date differs from data generation date. |

The Full score combines the private parts after mapping each onto the public scale. Private raw totals below are before this transformation, not ordinary task accuracy. Public-only points are not an added part of the Full score.

| Part | Weight (fraction) | Primary source/date | Caveat |
|---|---:|---|---|
| Public benchmarks | 0.00 | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Shown alongside, excluded from Full. |
| Private tests of the same skills | 0.60 | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Transformed onto the public scale before weighting. |
| Private tasks from new domains | 0.40 | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Transformed onto the public scale before weighting. |

| Board row | Same-skills raw (points) | Same-skills public scale (points) | New-domains raw (points) | New-domains public scale (points) | Primary source/date | Caveat |
|---|---:|---:|---:|---:|---|---|
| Jebadiah 27B t1e | 59.35 | 58.98 | 56.57 | 72.52 | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Rounded display values; Full uses the unrounded transformed components. |
| Jev 1.14 | 59.28 | 58.91 | 48.26 | 62.24 | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Rounded display values; Full uses the unrounded transformed components. |
| Jebadiah 9B t1e | 51.04 | 50.57 | 39.47 | 51.39 | [board data, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/v03.json) | Rounded display values; Full uses the unrounded transformed components. |

Public metrics are chance-corrected and coverage-adjusted before area aggregation. Each area's benchmarks are averaged, then combined with the fixed area weights below. Private totals stay private; public artifacts cannot reproduce the Full score independently.

| Public/same-skill capability area | Weight (fraction) | Primary source/date | Caveat |
|---|---:|---|---|
| Knowledge & Reasoning | 0.258494 | [area weights, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/own.json) | Rounded weights; new-domain private tasks form a separate block. |
| Language Understanding | 0.258494 | [area weights, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/own.json) | Rounded weights; new-domain private tasks form a separate block. |
| Retrieval & Classification | 0.200229 | [area weights, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/own.json) | Rounded weights; new-domain private tasks form a separate block. |
| Tools & Automation | 0.182783 | [area weights, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/own.json) | Rounded weights; new-domain private tasks form a separate block. |
| Arts & Human Taste | 0.100000 | [area weights, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/own.json) | Rounded weights; new-domain private tasks form a separate block. |

| Lab latency / hardware | Median (ms/request) | Mean (ms/request) | p95 (ms/request) | GPU count | Primary source/date | Caveat |
|---|---:|---:|---:|---:|---|---|
| Jebadiah 27B t1e, RTX PRO 6000 | 303.2 | 809.5 | 2950.2 | 1 | [board measurements, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/index.json) | Lab single process, sequential requests, fastest supported path; p95 is not the entry cutoff. |
| Jebadiah 9B t1e, RTX PRO 6000 | 125.1 | 294.2 | 968.0 | 1 | [board measurements, 2026-10-21](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/76fbe3765538c933477c7213158684edaa2048cb/data/index.json) | Same lab method; no CPU, Mac or quantized-runtime guarantee. |

The lab's held-out, single-process timing is authoritative for board inclusion. Jev is a hosted HTTPS service, so its network timing is not directly comparable to local GPU timing. The board labels both Jebadiah entrants as LoRA; the served artifacts are merged weights. The prior standard-read Jebadiah rows were replaced. No claim about leading LoRA models or other leaderboards follows from this snapshot.

The current board rows use the public full-weight releases with the T1e engine recipe. For GGUF and MLX, use the versioned repositories below and their own agreement records; their standalone scripts use the standard read, so the board scores do not automatically apply.

| Model | Full weights | GGUF | MLX | Primary source/date | Caveat |
|---|---|---|---|---|---|
| Jebadiah 27B v2.1 | [full](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1) | [GGUF](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1-GGUF) | [4-bit](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1-MLX-4bit), [8-bit](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1-MLX-8bit) | Published build and checksum records, checked 2026-10-21 | Quantized outputs can differ; no separate Full score. |
| Jebadiah 9B v2.1 | [full](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1) | [GGUF](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1-GGUF) | [4-bit](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1-MLX-4bit), [8-bit](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1-MLX-8bit) | Published build and checksum records, checked 2026-10-21 | Earlier models below retain their own experiment results. |

### Reproduce the T1e recipe

Use [the pinned Decision Index engine](https://github.com/getainode/jebadiah/blob/678f6e2cc2bb1b51f34407d00255b5e13109ce3f/eval/decision-index/jebadiah_engine.py) with the released checkpoint. The recipe repeats the request payload inside the prompt. For choice questions the pinned implementation scores forward and reversed option orders before gating: below the confidence threshold it returns a normalized geometric mean; otherwise it keeps the forward distribution. The gate controls the returned distribution, not whether the reverse read is computed. Thinking remains disabled, and the shipped calibration is unchanged.

| Required setting | Value | Unit | Primary source/date | Caveat |
|---|---|---|---|---|
| Model | `frontier-infra/jebadiah-27b-v2-1` | Hub repo | [environment, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/environment.json) | Full merged checkpoint. |
| Model revision | `e38b48b82bcda8943d0958072c042c5867284838` | Git commit | [environment, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/environment.json) | Current card-only commits retain the same weight LFS OIDs. |
| Engine | `678f6e2cc2bb1b51f34407d00255b5e13109ce3f` | Git commit | [engine, 2026-10-12](https://github.com/getainode/jebadiah/blob/678f6e2cc2bb1b51f34407d00255b5e13109ce3f/eval/decision-index/jebadiah_engine.py) | `decision-index-t1e` peels to this commit. |
| `repeat_payload` | `true` | boolean | [environment, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/environment.json) | Repeats the user payload inside the prompt. |
| `reverse_avg_below` | 0.7 | probability | [engine, 2026-10-12](https://github.com/getainode/jebadiah/blob/678f6e2cc2bb1b51f34407d00255b5e13109ce3f/eval/decision-index/jebadiah_engine.py) | Choice-only distribution gate; both orders are computed. |
| `prefix_cache` | `true` | boolean | [environment, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/environment.json) | Request-internal prefix reuse. |
| `max_batch` | 64 | rendered questions/batch | [environment, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/environment.json) | Internal batching; not concurrent user requests. |
| `batch_tokens` | 32768 | padded tokens/batch | [environment, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/environment.json) | Preserve the engine's prompt, labels and numeric path. |

Submission: [Decision Index PR](https://github.com/apolinario/decision-index/pull/199). The result dataset pins the model and inference code. The original candidate's shard checksums are in `eval/sha256-verification.json`; the checkpoint was not retrained for T1e.

### Historical public runs and runtime cost

These self-runs use the public scorer. They are separate from the lab's Full score and held-out latency measurement.

| Metric | Standard read | T1e | Unit | Primary source/date | Caveat |
|---|---:|---:|---|---|---|
| Public-only index | 57.03 | 60.46 | points | [standard, 2026-10-10](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-v03/scores.json), [T1e, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Public edition 0.3 scorer, unchanged weights. |
| Median latency | 106.5 | 291.2 | ms/request | [result record, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Self-run synchronized in-process timing, excludes model loading. |
| Mean latency | 247.3 | 975.6 | ms/request | [result record, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Prefix reuse and internal batching; differs from lab timing. |
| p95 latency | 599.5 | 3095.8 | ms/request | [result record, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Slow requests remain; not a per-request upper bound. |
| Successful scored requests | 140178 | 140178 | requests | [result record, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Scoring exclusions are outside this count. |

| Capability area | Standard read | T1e | Unit | Primary source/date | Caveat |
|---|---:|---:|---|---|---|
| Knowledge & Reasoning | 43.12 | 48.23 | public skill points | [result records, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Same weights; aggregate gains can conceal individual task regressions. |
| Language Understanding | 63.55 | 65.46 | public skill points | [result records, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Same weights; aggregate gains can conceal individual task regressions. |
| Retrieval & Classification | 56.70 | 59.94 | public skill points | [result records, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Same weights; aggregate gains can conceal individual task regressions. |
| Tools & Automation | 76.03 | 79.26 | public skill points | [result records, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Same weights; aggregate gains can conceal individual task regressions. |
| Arts & Human Taste | 42.04 | 45.80 | public skill points | [result records, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Same weights; aggregate gains can conceal individual task regressions. |

The largest index-task regressions in these same public runs remain visible:

| Benchmark | Standard read | T1e | Delta | Unit | Primary source/date | Caveat |
|---|---:|---:|---:|---|---|---|
| Humicroedit | 25.11 | 23.29 | -1.82 | public skill points | [result records, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Self-run task regression; no significance claim. |
| SATA-Bench | 26.49 | 24.28 | -2.21 | public skill points | [result records, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Self-run task regression; no significance claim. |
| RAGTruth response-level hallucination | 61.89 | 60.46 | -1.43 | public skill points | [result records, 2026-10-15](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/fec5485aa43b818b6b14fada6d4b520e22633d31/runs/jebadiah-27b-v2-1-t1e-678f6e2-v03/scores.json) | Self-run task regression; no significance claim. |

The earlier release comparison used the standard read, before T1e:

| Capability area | Previous release | v2.1 standard | Delta | Unit | Primary source/date | Caveat |
|---|---:|---:|---:|---|---|---|
| Knowledge & Reasoning | 40.78 | 43.12 | +2.34 | public skill points | [published comparison, 2026-10-10](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1/blob/e38b48b82bcda8943d0958072c042c5867284838/eval/public-comparison.csv) | Historical self-run comparison; Tools declined despite aggregate gains. |
| Language Understanding | 59.66 | 63.55 | +3.89 | public skill points | [published comparison, 2026-10-10](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1/blob/e38b48b82bcda8943d0958072c042c5867284838/eval/public-comparison.csv) | Historical self-run comparison; Tools declined despite aggregate gains. |
| Retrieval & Classification | 53.89 | 56.70 | +2.81 | public skill points | [published comparison, 2026-10-10](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1/blob/e38b48b82bcda8943d0958072c042c5867284838/eval/public-comparison.csv) | Historical self-run comparison; Tools declined despite aggregate gains. |
| Tools & Automation | 78.13 | 76.03 | -2.10 | public skill points | [published comparison, 2026-10-10](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1/blob/e38b48b82bcda8943d0958072c042c5867284838/eval/public-comparison.csv) | Historical self-run comparison; Tools declined despite aggregate gains. |
| Arts & Human Taste | 40.73 | 42.04 | +1.31 | public skill points | [published comparison, 2026-10-10](https://huggingface.co/frontier-infra/jebadiah-27b-v2-1/blob/e38b48b82bcda8943d0958072c042c5867284838/eval/public-comparison.csv) | Historical self-run comparison; Tools declined despite aggregate gains. |

The recipe increases prompt work and latency. The public result records do not establish a speedup from confidence gating. No statistical significance is claimed. Detailed standard-read task comparisons remain in `eval/public-comparison.csv`.

### Reproduce the T1e recipe for the 9B

Use [the pinned Decision Index engine](https://github.com/getainode/jebadiah/blob/678f6e2cc2bb1b51f34407d00255b5e13109ce3f/eval/decision-index/jebadiah_engine.py) with the released checkpoint. The recipe repeats the request payload inside the prompt. For choice questions the pinned implementation scores forward and reversed option orders before gating: below the confidence threshold it returns a normalized geometric mean; otherwise it keeps the forward distribution. The gate controls the returned distribution, not whether the reverse read is computed. Thinking remains disabled, and the shipped calibration is unchanged.

| Required setting | Value | Unit | Primary source/date | Caveat |
|---|---|---|---|---|
| Model | `frontier-infra/jebadiah-9b-v2-1` | Hub repo | [environment, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/environment.json) | Full merged checkpoint. |
| Model revision | `de19e80e18aa23420be410d54fdee435587eaad3` | Git commit | [environment, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/environment.json) | Current card-only commits retain the same weight LFS OIDs. |
| Engine | `678f6e2cc2bb1b51f34407d00255b5e13109ce3f` | Git commit | [engine, 2026-10-12](https://github.com/getainode/jebadiah/blob/678f6e2cc2bb1b51f34407d00255b5e13109ce3f/eval/decision-index/jebadiah_engine.py) | `decision-index-t1e` peels to this commit. |
| `repeat_payload` | `true` | boolean | [environment, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/environment.json) | Repeats the user payload inside the prompt. |
| `reverse_avg_below` | 0.7 | probability | [engine, 2026-10-12](https://github.com/getainode/jebadiah/blob/678f6e2cc2bb1b51f34407d00255b5e13109ce3f/eval/decision-index/jebadiah_engine.py) | Choice-only distribution gate; both orders are computed. |
| `prefix_cache` | `true` | boolean | [environment, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/environment.json) | Request-internal prefix reuse. |
| `max_batch` | 64 | rendered questions/batch | [environment, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/environment.json) | Internal batching; not concurrent user requests. |
| `batch_tokens` | 32768 | padded tokens/batch | [environment, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/environment.json) | Preserve the engine's prompt, labels and numeric path. |

Submission: [Decision Index PR](https://github.com/apolinario/decision-index/pull/188). The result dataset pins the model and inference code. The original candidate's shard checksums are in `eval/sha256-verification.json`; the checkpoint was not retrained for T1e.

### Historical public runs and runtime cost for the 9B

These self-runs use the public scorer. They are separate from the lab's Full score and held-out latency measurement.

| Metric | Standard read | T1e | Unit | Primary source/date | Caveat |
|---|---:|---:|---|---|---|
| Public-only index | 47.20 | 51.53 | points | [standard, 2026-10-10](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-v03/scores.json), [T1e, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Public edition 0.3 scorer, unchanged weights. |
| Median latency | 49.9 | 119.9 | ms/request | [result record, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Self-run synchronized in-process timing, excludes model loading. |
| Mean latency | 97.1 | 342.1 | ms/request | [result record, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Prefix reuse and internal batching; differs from lab timing. |
| p95 latency | 209.2 | 1018.0 | ms/request | [result record, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Slow requests remain; not a per-request upper bound. |
| Successful scored requests | 140178 | 140178 | requests | [result record, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Scoring exclusions are outside this count. |

| Capability area | Standard read | T1e | Unit | Primary source/date | Caveat |
|---|---:|---:|---|---|---|
| Knowledge & Reasoning | 32.89 | 35.27 | public skill points | [result records, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Same weights; aggregate gains can conceal individual task regressions. |
| Language Understanding | 52.24 | 56.58 | public skill points | [result records, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Same weights; aggregate gains can conceal individual task regressions. |
| Retrieval & Classification | 49.56 | 55.50 | public skill points | [result records, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Same weights; aggregate gains can conceal individual task regressions. |
| Tools & Automation | 65.84 | 70.37 | public skill points | [result records, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Same weights; aggregate gains can conceal individual task regressions. |
| Arts & Human Taste | 32.37 | 38.10 | public skill points | [result records, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Same weights; aggregate gains can conceal individual task regressions. |

The largest index-task regressions in these same public runs remain visible:

| Benchmark | Standard read | T1e | Delta | Unit | Primary source/date | Caveat |
|---|---:|---:|---:|---|---|---|
| MuSR | 31.29 | 30.24 | -1.05 | public skill points | [result records, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Self-run task regression; no significance claim. |
| SATA-Bench | 32.26 | 29.75 | -2.51 | public skill points | [result records, 2026-10-14](https://huggingface.co/datasets/frontier-infra/jebadiah-v2-1-decision-index-results/blob/1e1c65a97180158cda3858dffb85dad34e2750bc/runs/jebadiah-9b-v2-1-t1e-678f6e2-v03/scores.json) | Self-run task regression; no significance claim. |

The earlier release comparison used the standard read, before T1e:

| Capability area | Previous release | v2.1 standard | Delta | Unit | Primary source/date | Caveat |
|---|---:|---:|---:|---|---|---|
| Knowledge & Reasoning | 29.22 | 32.89 | +3.67 | public skill points | [published comparison, 2026-10-10](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1/blob/de19e80e18aa23420be410d54fdee435587eaad3/eval/public-comparison.csv) | Historical self-run comparison; Tools declined despite aggregate gains. |
| Language Understanding | 47.90 | 52.24 | +4.34 | public skill points | [published comparison, 2026-10-10](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1/blob/de19e80e18aa23420be410d54fdee435587eaad3/eval/public-comparison.csv) | Historical self-run comparison; Tools declined despite aggregate gains. |
| Retrieval & Classification | 43.60 | 49.56 | +5.96 | public skill points | [published comparison, 2026-10-10](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1/blob/de19e80e18aa23420be410d54fdee435587eaad3/eval/public-comparison.csv) | Historical self-run comparison; Tools declined despite aggregate gains. |
| Tools & Automation | 66.68 | 65.84 | -0.84 | public skill points | [published comparison, 2026-10-10](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1/blob/de19e80e18aa23420be410d54fdee435587eaad3/eval/public-comparison.csv) | Historical self-run comparison; Tools declined despite aggregate gains. |
| Arts & Human Taste | 32.36 | 32.37 | +0.01 | public skill points | [published comparison, 2026-10-10](https://huggingface.co/frontier-infra/jebadiah-9b-v2-1/blob/de19e80e18aa23420be410d54fdee435587eaad3/eval/public-comparison.csv) | Historical self-run comparison; Tools declined despite aggregate gains. |

The recipe increases prompt work and latency. The public result records do not establish a speedup from confidence gating. No statistical significance is claimed. Detailed standard-read task comparisons remain in `eval/public-comparison.csv`.

## Run it yourself

Install the client for your existing runtime:

```bash
pip install jebadiah-decide
```

If Jeb is useful in your project, I'd appreciate a star on [getainode/jebadiah](https://github.com/getainode/jebadiah). If you hit a problem, [open an issue](https://github.com/getainode/jebadiah/issues) with the runtime and a request that reproduces it.

**[Run Jeb locally](docs/run-locally.md)** is the setup guide for every runtime: Ollama, LM Studio, llama.cpp's
`llama-server`, vLLM, MLX on a Mac and AINode, each with the install, the exact request, the output to expect and
the gotchas, plus how to use Jeb as [JDE](https://github.com/Titanium-Devops/jde)'s judge.
[`clients/python`](clients/python) is `jebadiah-decide`, one small client with a backend for each of them.

For a first try in a chat window, [optional guidance profiles](docs/chat-window.md)
point to the setup guide. They leave the decision templates intact; the LM Studio
preset still needs GUI and API validation.

`server/` is a standalone server for the published models. It runs on one CUDA GPU or an Apple Silicon Mac and
needs Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
cd server && uv run jebadiah-serve --model frontier-infra/jebadiah-4b-v2
```

The model downloads on first start (the default, without `--model`, is the 9B). Then open the playground at
<http://localhost:8000/>, or call the API: `POST /v1/systemone` takes the request shape above, `POST /v1/decide` takes
AINode's shape, and the schemas are at <http://localhost:8000/docs>. Flags, auth, latency and caveats are in
[`server/README.md`](server/README.md).

## Formats of earlier releases

The earlier releases also have GGUF and MLX builds:
[27B GGUF](https://huggingface.co/frontier-infra/jebadiah-27b-GGUF),
[27B MLX](https://huggingface.co/frontier-infra/jebadiah-27b-MLX),
[9B v2 GGUF](https://huggingface.co/frontier-infra/jebadiah-9b-v2-GGUF),
[9B v2 MLX](https://huggingface.co/frontier-infra/jebadiah-9b-v2-MLX),
[4B v2 GGUF](https://huggingface.co/frontier-infra/jebadiah-4b-v2-GGUF),
[4B v2 MLX](https://huggingface.co/frontier-infra/jebadiah-4b-v2-MLX). Every file was checked against the merged bf16
model on the same 260 questions (84 choice, 89 noul, 87 score); the count is how many get the same pick:

| Model | GGUF Q8_0 | GGUF Q5_K_M | GGUF Q4_K_M | MLX 8-bit | MLX 4-bit |
|---|---:|---:|---:|---:|---:|
| 27B | 260 | 255 | 258 | 259 | 248 |
| 9B v2 | 257 | 250 | 240 | 256 | not published (224) |
| 4B v2 | 256 | 239 | 236 | 252 | 215 |

The 8-bit builds are close to lossless. The 9B and 4B are much more sensitive to quantization than the 27B, so use
their smaller files only when a larger one does not fit; each card says how many answers change.

Jeb reads label probabilities rather than generating text, so it needs a runner that returns them, not a chat
window. The GGUF files need llama.cpp v0.5.0 or later (older builds do not know the architecture), and each GGUF
repository carries `scripts/decide_gguf.py`, which renders the prompt the way AINode does and reads the labels from
llama-server. Each MLX repository carries `scripts/decide_mlx.py` (checked with mlx-lm 0.31.3). The GGUF repositories
also carry `scripts/decide_lmstudio.py`, which does the same through LM Studio's local server with reasoning off
(tested on the 9B only), and `scripts/decide_ollama.py`, which does it through Ollama's `/api/generate` in raw mode
(tested on all three sizes). LM Studio and Ollama return the top 20 log probabilities, so both are limited to 20
options per question, the same cap as AINode's route. The [local setup guide](docs/run-locally.md) covers each one.

Third-party quantizations exist too. [mradermacher/jebadiah-9b-v2-GGUF](https://huggingface.co/mradermacher/jebadiah-9b-v2-GGUF)
is one; his Q8_0 gives the same pick as ours on 260 of 260 questions. The models are mirrored on
[ModelScope](https://www.modelscope.ai/profile/JasonBrashear). The versioned v2.1 mirror weight manifests match their Hugging Face counterparts; the board results still require the T1e engine recipe.

## Reproduce a run end to end

You need one NVIDIA GPU with 80 GB (we used H100 PCIe and A100 80 GB PCIe; a DGX Spark also works, slowly), about
200 GB of disk and Ubuntu with the driver installed. No tokens are needed: every model and dataset is public.

```bash
git clone https://github.com/getainode/jebadiah && cd jebadiah

# 1. environment, base models, public data (about 20 to 30 minutes)
sudo JEB_ROOT=/workspace/jeb bash scripts/setup.sh
cat /workspace/jeb/READY

# 2. train, fit temperatures, evaluate, write results.json (about 110 minutes for the 9B v2 run on an A100)
bash /workspace/jeb/run_sweep.sh --only 9b-chat-v1
bash /workspace/jeb/run_sweep.sh --status

# 3. compare with the published record
python3 -c "import json; print(json.load(open('/workspace/jeb/runs/9b-chat-v1/results.json'))['headline'])"
python3 -c "import json; print(json.load(open('results/runs/9b-chat-v1/results.json'))['headline'])"
```

`setup.sh` stages the code from this checkout into `$JEB_ROOT`, installs a Python 3.12 environment with pinned
`transformers` and `peft`, downloads every base model named in `configs/sweep.json` at its pinned revision, and builds
the v1 pool and all 20 test sets with `data/convert_data.py`. Its `pool.jsonl` should match the sha256 recorded in
`data/manifests/pools.json` (`python3 data/pool_manifest.py check /workspace/jeb data-v1`). Any run in
`configs/sweep.json` can be named with `--only`; runs are resumable and a finished run is skipped. The nonce test runs
after training with `JEB_NONCE_RUNS=9b-chat-v1 bash /workspace/jeb/nonce_all.sh`.

Runs on the v0 pool (`data`) used an earlier state of the converter (manifest in
`data/manifests/convert-data-v0.json`); as committed, the converter builds the v1 pool, so the v0 runs are recorded
here but not rebuilt by `setup.sh`.

The later pools need more than public downloads:

| Pool | How it is built |
|---|---|
| `data-v2` | `data/pool-v2/generate.py` and `finalize.py` against an AINode endpoint serving the two teachers (`JEB_TEACHER_BASE_URL`, `JEB_TEACHER_KEY`). The result is published as the dataset [frontier-infra/jebadiah-synth-v2](https://huggingface.co/datasets/frontier-infra/jebadiah-synth-v2). |
| `data-dose-c20`, `data-dose-c33` | `data/dose/make_dose_pool.py` from the v1 pool and the synthetic records |
| `data-v3-c20f`, `data-v3-c20f-n3k` | `experiments/fable-targets/` (needs a logged-in Claude Code for the Fable calls) |
| `data-dose-c20h`, `data-dose-c20h-alt` | `data/human-noul/convert_human_noul.py`, then `make_c20h_pools.py` |
| `data-v3-c20fh-alt`, `data-v3-full`, `data-v3-full-s` | `data/compose_pools.py`, verified byte for byte against the manifest |
| `data-v2b`, `data-v2-cn`, `data-v2-score` | filtered inline during the sweep; the builder was not kept, only the sha256 |

Copy a built pool directory into `$JEB_ROOT` and name it in the run's `data_dir`.

## Repository layout

| Path | What is in it |
|---|---|
| `train/` | the trainer, the model and prompt code, the evaluator, the temperature fit, the sweep driver |
| `data/` | the public-data converter, the split and the linter, the synthetic pool generator, the human yes/no converter, the dose and v3 pool builders, the Jevals task files, and `manifests/` with the sha256 of every pool |
| `data/human/prolific-r1.jsonl` | 54 human-labelled yes/no questions (mean of 2 or 3 raters, no rater identifiers) |
| `eval/` | the nonce test, the JevBench runner and shim, the Decision Index engine |
| `experiments/` | one short note per experiment, the teacher bake-off and the Fable target code |
| `results/` | `runs/<run>/` (results.json, config, training summary and loss history, eval summary, temperatures, prompt contract), `nonce/<run>/` (summaries), `jevbench/<label>/`, `sweeps/`, and `SUMMARY.md` |
| `configs/` | `sweep.json` (every run and its settings) and `temperatures/` for each published model |
| `scripts/` | `setup.sh`, `run_sweep.sh`, `nonce_all.sh`, `results_table.py`, and `make_v1_card.py` (the model card generator) |
| `server/` | the standalone server: `/v1/systemone`, `/v1/decide`, a browser playground and its tests |
| `clients/python/` | `jebadiah-decide`, the client for llama-server, Ollama, LM Studio, vLLM, MLX and any `/v1/systemone` server, and its tests |
| `docs/` | [`run-locally.md`](docs/run-locally.md), the setup guide per runtime, and `examples/` (JDE with a local Jeb as its judge) |

## What's next

All of this is in progress or planned; none of it has a result yet.

- **Judge Jeb.** An adapter specialised for [JDE](https://github.com/Titanium-Devops/jde)'s judging questions (does
  an agent's answer cover every part of the task), trained on JDE-shaped states that the current pool does not have.
  In training.
- **JDE calibration.** JDE now judges with Jeb by default (its `jeb` judge, pointed at `jeb serve`, AINode or the
  standalone server). Still to come: temperatures fitted per question family on JDE's own labelled traffic.
- **Coding-agent hooks.** Jeb as a check inside a coding agent's hooks, in shadow mode first: it answers beside the
  existing check and acts on nothing until its numbers are in.
- **DecisionBench.** Planned: a benchmark for typed decisions designed to be fair to every model, ours included.

## Links

- Project site: [jebadiah.ai](https://jebadiah.ai)
- Models: [jebadiah-27b](https://huggingface.co/frontier-infra/jebadiah-27b),
  [jebadiah-9b-v2](https://huggingface.co/frontier-infra/jebadiah-9b-v2),
  [jebadiah-4b-v2](https://huggingface.co/frontier-infra/jebadiah-4b-v2),
  [jebadiah-9b-v1](https://huggingface.co/frontier-infra/jebadiah-9b-v1),
  [jebadiah-4b-v1](https://huggingface.co/frontier-infra/jebadiah-4b-v1),
  [jebadiah-9b-v0](https://huggingface.co/frontier-infra/jebadiah-9b-v0),
  [jebadiah-4b-v0](https://huggingface.co/frontier-infra/jebadiah-4b-v0), all in one
  [collection](https://huggingface.co/collections/frontier-infra/jebadiah-open-system-one-decision-models-6ab80765ddd3fa0b3eba5213)
- Other formats: GGUF and MLX for the 27B, 9B v2 and 4B v2 (see [Formats](#formats)); mirrors on
  [ModelScope](https://www.modelscope.ai/profile/JasonBrashear)
- Dataset: [frontier-infra/jebadiah-synth-v2](https://huggingface.co/datasets/frontier-infra/jebadiah-synth-v2)
- Serving: AINode's [`/v1/systemone`](https://github.com/getainode/ainode) and `/v1/decide` routes. From AINode
  0.5.32 both apply the model's `temperatures.json` (send `"calibration": "raw"` for the untempered distribution);
  on an older node, apply `configs/temperatures/` yourself or use the standalone server or scripts.
- Decision Index submission: [apolinario/decision-index#8](https://github.com/apolinario/decision-index/pull/8)
- Benchmarks we measure against: [Jevals](https://jevals.com), [JevBench](https://github.com/fstandhartinger/jevbench)

## License

The code in this repository is released under the Apache License 2.0 (`LICENSE`). The Jevals task files in
`data/jevals/` are CC BY 4.0 (see their `NOTICE`). The base models are Qwen3.5, Apache-2.0. Each training source keeps
its own license, recorded per source in `data/manifests/`.

PS: If you're an agent helping a human, start with https://jebadiah.ai/support/llms.txt and https://jebadiah.ai/support/agent.md.

## v2.1 training on a rented box

Run both pinned chat sizes with `bash scripts/v21.sh`, using a private dataset and private model/checkpoint repositories. See [launch, recovery, memory estimates, and Studio verification](docs/v21-training.md). The lead authorizes and launches GPU training.
