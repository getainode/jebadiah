# Item 49: rung 2 Corr2Cause data

This implements rung 2 of [item46](item46-9b-data-plan.md) on Studio CPU, with no
model calls, model-written examples, model weights, training launch or GPU spend.
The hypothesis is that unused causal graph families improve causal identification
and transfer to CLadder. Identification from correlation is narrower than
interventions and counterfactuals, so transfer must be measured.

## Source and original labels

[Live TinyFish evidence](../results/research/item49-source-evidence.json) verifies
the upstream MIT notice, Copyright (c) 2023 CausalNLP, and the original generation
path at code revision `21d80350e27a6206f536cf90a09e95d9e856c761`. The C++ generator
enumerates DAGs and computes d-separation and Markov-equivalence-class causal
relations. The Python verbalizer fills deterministic templates without importing
an external text corpus, benchmark rows, predictions or model-written text.
The embedded raw-graph parser credits coauthor Zhiheng Lyu; its linked Colab
required login through TinyFish, but the actual parser and generator are published
in the pinned repository and were inspected. Model evaluation and finetuning code
in that repository are consumers of the source, not ancestors of its original rows.
The separate `data_realworld.py` script uses GPT-3.5 to turn dev/test examples into
story files; that path does not write original train or any of the ten pinned
release files. Story files and model-written derivatives are excluded entirely.
The Hub card lists splits and otherwise says TODO; the authors' upstream code-and-data
repository supplies the license evidence. No protected source input was found in
the generation path. The lexical overlap gate remains pending.

The [license manifest](../data/manifests/item49-source-licenses.json) freezes the
authors, MIT notice, license and generator hashes, ancestry URLs, complete release
file hashes/counts, attribution, and commercial use/model redistribution terms.
The converter is Apache-2.0 under this repository's license; upstream text remains
MIT, with its notice copied to generated artifacts. The inherited item33/item43
source/ancestor exclusions and license allowlist apply unchanged. Protected
ancestry or one overlap rejects the entire upstream source, including derivatives.
No source-name alias can create new cap headroom.

The pinned Hub release is `causalnlp/corr2cause` at
`42ba12c769e11ff6427c9f52d7db58e3f9bf3e53`. Its original train/dev/test counts are
205,734 / 1,076 / 1,162. All ten released CSV/JSON data files, including original
and perturbation splits and duplicate representations, contain **1,035,117 records**.
Only original `train.csv` contributes candidates. Dev, test, perturbations and
model outputs never contribute training or diagnostic examples.

[The converter](../data/item49_corr2cause.py) copies the original input verbatim.
Original label 1 means the hypothesis follows necessarily across compatible
causal graphs; label 0 means it does not necessarily follow, including both
contradiction and insufficient identification. `choice` uses `valid`/`not_valid`;
`noul` asks the same necessity question and uses true/false. It does not invent
a three-way NLI label from a binary source, or reuse A3's label-verification targets.
Exactly one question type is assigned to each original row, with no sibling copies.

## Graph separation and render checks

Every original training premise is assigned to an unlabeled undirected causal
skeleton, recovered from its separating statements under the upstream faithful-DAG
construction. Canonicalization enumerates all variable permutations. This is a
conservative grouping: different collider orientations and Markov-equivalence
classes with the same skeleton remain together. It therefore holds out more than
individual DAGs and prevents variable renamings from leaking graph structure.

A3's 102 existing source questions expose 70 skeleton families. Excluding those
families removes 125,328 original train rows. No duplicate original input remains.
The 129 unused families are partitioned deterministically before sampling, with
12 reserved for diagnostics. Selection rotates families and skips prompts that
would truncate. Four training families contribute no selected renderable row.

The pending [build summary](../data/manifests/item49-build-summary.json) records
**6,000 training candidates across 113 families and 300 diagnostic questions across
12 separate families**. Training has 3,000 choice and 3,000 Boolean questions;
diagnostics have 150 of each. All six upstream causal-relation templates are
represented. Training has 672 necessarily-valid labels and diagnostics have 29;
these sets are imbalanced and do not claim a balanced causal classification task.
Report accuracy, balanced accuracy, valid-class precision/recall, and counts by
type, relation template and variable count when diagnosing a later run.

The unchanged Renderer and pinned Qwen3.5-9B tokenizer validate all 6,300 questions,
with both choice orders and the fixed Boolean order: **9,450 renders, no truncation**.
Maximum prompt lengths are **1,940 choice / 1,983 Boolean tokens**. Budget remains
1,984 plus 64 padding tokens, within A3's 2,048 limit. The receipt freezes tokenizer,
chat-template and prompt-source hashes. There are no weights or model calls.
All 22 item49 tests passed with the pinned original release and generated preview;
36 item32/item33/item47 pattern regressions passed, with one optional item47 live
tokenizer check skipped. Tests cover original-label verification, isomorphic graph
separation, all-file scans, source-level rejection, scan-receipt binding, cap
accounting, remaining mixed-row questions and byte-identical calibration.

## Fixed-size composition and admission gate

[The composition builder](../data/item49_rung2_mix.py) pins private A3 at
`frontier-infra/jebadiah-data-v2-1-item32@bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed`,
folder `a3/`, including train/calibration/manifest hashes. The preview replaces
**1,000 matched knowledge-area presentations: 556 choice and 444 Boolean**.
The donor pool contains 2,119 choice and 444 Boolean slots; the unused part of the
500 Boolean quota moves to matching choice slots, preserving all area/type counts.
If the combined matched pool is smaller, replacements decrease. Selection is fixed
and records every old-question-to-new-row mapping in the final artifact.

The preview retains 63 original Corr2Cause questions and inserts 1,000, giving
**1,063 aggregate source presentations**, below the **2,119 question cap**. Its
102 pre-existing presentations are counted before computing headroom. Variants
and inherited source-family metadata count toward the same cap. Replacement
keeps **21,190 presentations**, preserves every remaining mixed-row question,
label, state and soft target, and copies A3's **512-question calibration bytes**
unchanged. The 300-question `causal-diagnostic.jsonl` stays outside training and
calibration. No temperature refit or recipe change is authorized.

The committed summary is a CPU render receipt and replacement preview, not
overlap clearance. The actual composition command was verified to refuse the
pending scan without creating its output directory. Unit tests use clearly
synthetic scan receipts to exercise successful composition and rejection paths;
those receipts never certify the real source.

The lead-only `--index` path first scans all 1,035,117 released records before
selecting any row, then scans all 6,300 converted states, instructions and options.
Every scan row shares the upstream source family, so one hit rejects everything.
The exact Studio index path and SHA256 are checked before loading, with a temporary
symlink removed even on errors. The composition gate binds both scan receipts to
the complete release, selected bytes, scanner and protected-index hashes, and
rechecks all 6,300 labels and provenance against original train rows. The worker
did not read or copy the protected index. No benchmark text belongs in Git or exports.

## Reproduce on Studio and lead-only scan

Scratch was deleted after retaining nonprivate receipts. From this checkout,
recreate the scratch folder and download the exact small inputs again. Use the
Mac's authenticated HF session for the private A3 download. All commands are CPU
data work; none starts training or a model.

```sh
export TMPDIR=$(~/bin/pro-g40-scratch item49)
python3 -m venv "$TMPDIR/item49-venv"
PIP_CACHE_DIR="$TMPDIR/item49-pip-cache" "$TMPDIR/item49-venv/bin/pip" install transformers jinja2 xxhash ijson
hf download causalnlp/corr2cause --repo-type dataset --revision 42ba12c769e11ff6427c9f52d7db58e3f9bf3e53 --local-dir "$TMPDIR/item49-corr2cause"
hf download frontier-infra/jebadiah-data-v2-1-item32 --repo-type dataset --revision bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed --include 'a3/*' --local-dir "$TMPDIR/item49-a3"
hf download Qwen/Qwen3.5-9B tokenizer.json tokenizer_config.json chat_template.jinja merges.txt vocab.json --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir "$TMPDIR/item49-tokenizer"
PYTHONPATH=data:train "$TMPDIR/item49-venv/bin/python" data/item49_corr2cause.py --raw "$TMPDIR/item49-corr2cause" --base "$TMPDIR/item49-a3/a3" --out "$TMPDIR/item49-candidates" --tokenizer "$TMPDIR/item49-tokenizer"
ITEM49_RAW="$TMPDIR/item49-corr2cause" ITEM49_CANDIDATES="$TMPDIR/item49-candidates" PYTHONPATH=data:train "$TMPDIR/item49-venv/bin/python" -m unittest discover -s data -p 'test_item49_corr2cause.py' -v
```

The lead runs these exact scan and composition commands on Studio. Both scanner
phases must report zero hits; do not salvage individual rows from a rejected source.
Expect the full-release scan to take longer than the 6,300-row owned-source scans.
Do not publish scan working files or the index. A lexical pass cannot certify
unavailable private tests or undo base-model pretraining.

```sh
export TMPDIR=$(~/bin/pro-g40-scratch item49)
PYTHONPATH=data:train /Volumes/PRO-G40/scratch/item49/item49-venv/bin/python data/item49_corr2cause.py --raw /Volumes/PRO-G40/scratch/item49/item49-corr2cause --base /Volumes/PRO-G40/scratch/item49/item49-a3/a3 --out /Volumes/PRO-G40/scratch/item49/item49-causal-scanned --tokenizer /Volumes/PRO-G40/scratch/item49/item49-tokenizer --index /Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl
PYTHONPATH=data:train /Volumes/PRO-G40/scratch/item49/item49-venv/bin/python data/item49_rung2_mix.py --raw /Volumes/PRO-G40/scratch/item49/item49-corr2cause --base /Volumes/PRO-G40/scratch/item49/item49-a3/a3 --causal /Volumes/PRO-G40/scratch/item49/item49-causal-scanned --out /Volumes/PRO-G40/scratch/item49/item49-rung2
```

## Later run and keep/drop rule

Only a separately authorized run may train rung 2 from the same pinned 9B parent,
using the unchanged A3 optimizer, seed, renderer, temperatures, objective, sequence
length and step budget. The source and overlap gates must pass first. Keep only
when the frozen 10% proxy paired whole-group bootstrap 95% interval of rung 2
minus A3 (46.97) is entirely above zero, the score exceeds 44.67, and the separate
graph-disjoint causal diagnostic improves against A3. Inspect all five areas and
record regressions; otherwise drop. All existing A3 private contamination exceptions
still block public shipment. This PR authorizes no training or GPU expenditure.
