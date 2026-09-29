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
The 27B's DSj is the run's record, made with the score temperature it shipped with; with the refit described below
it is 21.3, and no accuracy changes. All numbers are ours, measured with the evaluator in this repository. Every
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
their own traffic. The 9B v2 and 4B v2 keep the temperatures they shipped with.

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

We also submitted Jebadiah to the [Decision Index](https://github.com/apolinario/decision-index)
([apolinario/decision-index#8](https://github.com/apolinario/decision-index/pull/8)); the engine is in
`eval/decision-index/` ([11](experiments/11-decision-index.md)). The PR first carried 9B v1 and now submits the 27B.
The maintainer asked for a complete self-run before adding it, with the kit's job on an RTX PRO 6000; that run is in
progress and its results are pending, so there is no Decision Index number here yet.

## Run it yourself

**[Run Jeb locally](docs/run-locally.md)** is the setup guide for every runtime: Ollama, LM Studio, llama.cpp's
`llama-server`, vLLM, MLX on a Mac and AINode, each with the install, the exact request, the output to expect and
the gotchas, plus how to use Jeb as [JDE](https://github.com/Titanium-Devops/jde)'s judge.
[`clients/python`](clients/python) is `jebadiah-decide`, one small client with a backend for each of them.

`server/` is a standalone server for the published models. It runs on one CUDA GPU or an Apple Silicon Mac and
needs Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
cd server && uv run jebadiah-serve --model frontier-infra/jebadiah-4b-v2
```

The model downloads on first start (the default, without `--model`, is the 9B). Then open the playground at
<http://localhost:8000/>, or call the API: `POST /v1/systemone` takes the request shape above, `POST /v1/decide` takes
AINode's shape, and the schemas are at <http://localhost:8000/docs>. Flags, auth, latency and caveats are in
[`server/README.md`](server/README.md).

## Formats

Besides the full bf16 weights, each current model has GGUF (Q8_0, Q5_K_M, Q4_K_M) and MLX builds:
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
[ModelScope](https://www.modelscope.ai/profile/JasonBrashear).

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
