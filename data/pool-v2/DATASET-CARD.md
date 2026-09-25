---
license: apache-2.0
language:
- en
pretty_name: Jebadiah synthetic typed decisions v2
size_categories:
- 10K<n<100K
task_categories:
- text-classification
tags:
- typed-decisions
- decision-model
- system-one
- synthetic
- ainode
- calibrated-probabilities
configs:
- config_name: default
  data_files: "data/synth-*.jsonl"
---

# Jebadiah synthetic typed decisions v2

14,714 typed-decision questions over 3,253 synthetic states in 24 families, authored and labelled on
Frontier Infra's own AINode fleet on 2026-09-22 and released under Apache-2.0. This is the choice and yes/no
half of the pool that trains [Jebadiah](https://huggingface.co/frontier-infra/jebadiah-9b-v1), an open System One
decision model. No customer data, no scraped text, no item from any evaluation set we report on.

## How it was made

- **States** were written by DeepSeek V4 Flash (served by AINode on two DGX Sparks) at temperature 0.9, one
  state per call with 3 to 5 questions in the `/v1/systemone` shape: choice questions carry 2 to 12 named options
  with a one-line description each; noul (yes or no) questions carry a description of the true and the false
  side. Every call was seeded with a random industry, persona, tone, length band, edge condition and month so
  states do not collapse to a template. The author was asked for the state and the questions only, never an
  answer, and any output with an answer-like field, an answer in the instructions, or a state field that gives
  the answer away was rejected. Unordered options were shuffled.
- **Labels** are the teachers' own probability distributions: every question was rendered exactly as the
  Jebadiah trainer renders it and read once through AINode's `/v1/decide` on two teachers, DeepSeek V4 Flash
  (`fraserprice/DeepSeek-V4-Flash-DSpark`) and Qwen3.8 27B (`unsloth/Qwen3.8-27B-NVFP4`). `gold` is the plain
  mean of the two distributions and `label` its argmax. Both teachers' distributions are kept in `provenance`.
  A question was dropped when the teachers' top picks disagreed and both were under 0.6, or when the averaged
  top probability was under 0.4. Nothing was sharpened.
- **Quality gates:** states deduplicated by normalised text and by 5-gram Jaccard over 0.8; the 20-option cap;
  yes/no kept between 40 and 60 percent true per family; the trainer's linter (no label-like key inside a
  question object) on every file; and an overlap check against 7,445 states from the 20 public evaluation sets
  Jebadiah reports on (highest 5-gram containment of any kept state: 0.077, zero dropped).

## Families

| family | states | choice | noul | teacher agreement | mean top probability |
|---|---:|---:|---:|---:|---:|
| helpdesk_ticket | 126 | None | None | 0.808 | 0.864 |
| cs_chat | 138 | None | None | 0.843 | 0.881 |
| agent_tool_selection | 141 | None | None | 0.781 | 0.841 |
| invoice_receipt | 147 | None | None | 0.726 | 0.817 |
| meeting_actions | 129 | None | None | 0.927 | 0.949 |
| incident_alert | 126 | None | None | 0.806 | 0.844 |
| email_routing | 130 | None | None | 0.891 | 0.911 |
| content_policy | 135 | None | None | 0.75 | 0.833 |
| product_review | 124 | None | None | 0.84 | 0.896 |
| log_severity | 156 | None | None | 0.887 | 0.892 |
| code_review_comment | 128 | None | None | 0.86 | 0.897 |
| pr_triage | 127 | None | None | 0.857 | 0.885 |
| scheduling_dispatch | 147 | None | None | 0.747 | 0.808 |
| sales_lead | 126 | None | None | 0.839 | 0.878 |
| form_validation | 139 | None | None | 0.724 | 0.787 |
| claim_evidence | 128 | None | None | 0.827 | 0.891 |
| summary_faithfulness | 134 | None | None | 0.763 | 0.832 |
| translation_adequacy | 144 | None | None | 0.73 | 0.813 |
| voice_intent | 131 | None | None | 0.814 | 0.858 |
| jde_echo | 140 | None | None | 0.871 | 0.906 |
| jde_parts | 125 | None | None | 0.832 | 0.88 |
| jde_coverage | 131 | None | None | 0.84 | 0.886 |
| jde_added_requirements | 135 | None | None | 0.9 | 0.922 |
| jde_parrot | 166 | None | None | 0.929 | 0.954 |

Teacher agreement is the share of questions where the two teachers' top picks match; mean top probability
is the averaged distribution's top value.

## Record format

One JSON object per line, one state per record, in the pool format the Jebadiah trainer reads. Top-level
keys: `id, set, subset, source, license, family, state, questions, label, target, provenance`. `questions` is an object keyed by question id; each question has `type`
(`choice` or `noul`), `instructions`, `criteria` (choice: option name to description; noul: `true` and
`false` descriptions), `gold` (probability per option, in option order), `label` (the argmax) and
`provenance` (each teacher's distribution, the author seed attributes and the model ids). `family` is
`<family>:<state hash>`, so a family-hash split keeps a state's questions on one side.

## Intended use and limits

Training and calibrating decision models (typed answers with probabilities), and studying teacher agreement
on typed questions. The labels are teacher distributions, not human labels: they carry the teachers' biases,
and a question the teachers agreed on can still be wrong. The states are synthetic and English; the five
`jde_*` families copy the shape of a help-desk team's live judgments (echo, parts, coverage, added
requirements, parrot) on invented tickets. The audit sample in `SAMPLE.md` shows 200 questions with both
distributions. Frontier Infra, Made in Texas.
