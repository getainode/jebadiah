# Item 47: rung 1 owned multi-document evidence data

This implements rung 1 of [item46](item46-9b-data-plan.md), targeting claim verification
and hallucination with original executable evidence worlds. Data preparation and verification
ran on Atlas CPU only. No model call, model-written example, external corpus, training launch
or GPU spend is part of this work. All example text is rendered by the finite rule-based
sentence grammar in [the generator](../data/item47_evidence_worlds.py).

## Complete owned source

The frozen finite source has **6,000 training candidates and 300 diagnostic questions**.
Each of 1,000 training and 50 diagnostic worlds produces supported, refuted and insufficient
variants, each with a three-way `choice` and a Boolean `noul` question. `noul` asks whether
the evidence proves the claim, so both refuted and insufficient map to false while their
symbolic statuses remain distinct in the audit metadata and three-way task.

Training uses forward two-, three- and four-document chains, a converging three-document
join and a branching three-document join. Diagnostics hold out the entire reverse-two,
reverse-three, diamond-four, cycle-three and crosslink-four rule/template families. These
are rule holds, not random rows or changed identifiers. Every diagnostic family has ten
worlds, all six siblings, and both claim polarities. Training has 1,200 questions per rule;
diagnostic has 60 per rule. Evidence statuses are balanced, as are question types, and half
of claims are explicitly negated. Document-count distributions are 1,200/3,600/1,200 in
training and 60/120/120 in diagnostics for two/three/four documents.

A conjunctive reference query binds entity IDs and the exact period across documents.
A symbolic relational join supplies the proof bindings and fact IDs. A second oracle
exhaustively enumerates assignments and agrees on every question. A contradiction changes
one terminal status value. Insufficient evidence deletes exactly one join premise or one
terminal fact, retaining wrong-period and wrong-entity distractors. Every document is
necessary: dropping any document from a supported world yields insufficient evidence.
An explicitly recorded status is single-valued per entity/period; a different value can
refute a positive claim or prove a negative claim. An absent value or unresolved reference
proves neither polarity. Facts from a different entity or period cannot fill a proof gap.

The rendered input contains only scope rules, two to four document texts, the reference
query and claim. Proofs, symbolic worlds, group IDs and status labels are record metadata,
never fed into the prompt. The fixed relation vocabulary and explicit queries are a
controlled compositional diagnostic, not a recreation of natural encyclopedia prose or a
claim of benchmark transfer. Diagnostic vocabulary is shared; executable graph families
are held out. Keep/drop still requires the measured frozen proxy, because simpler item33
G1 grounding did not improve overall performance.

## Provenance and admission

Code and generated text use the repository's Apache-2.0 license. The [source license
manifest](../data/manifests/item47-source-licenses.json) freezes the generator SHA256,
seed, original authors, license hash, notice, ancestry, commercial training and model
redistribution terms, complete-split counts and hashes. There are no external inputs,
model outputs, imported examples or benchmark-derived sentence templates. Item33/item43
provide builder utilities and admission mechanics, not records or evidence templates.
The inherited full source/ancestor exclusions are enforced unchanged. No Wikipedia,
HoVer, FEVER, HotpotQA or other Decision Index source or descendant is used as an
incoming source.

The complete 6,300-question source must pass the pinned item33 full-suite scanner **before
admission**, including both splits and every rendered state, instruction and option text.
All candidates have the same upstream scan family: one overlap rejects the whole source.
The builder checks the Studio protected path and index SHA256 before loading the index,
uses only a temporary symlink, and removes that symlink even on failure. Never copy or
export the protected index or its benchmark text. A lexical pass is not proof against
unavailable private tests or base-model pretraining.

The committed [build summary](../data/manifests/item47-build-summary.json) is an Atlas CPU
render receipt and replacement preview, **not overlap clearance or an admitted composition**.
The actual composition invocation was tested against the pinned A3 and refused pending
Studio clearance without creating its output directory. Tests use explicitly synthetic scan
fixtures to exercise both successful composition and failure paths; they do not certify
an overlap pass. Final composition and any training remain pending the Studio scan.

## Fixed-size replacement and prompt contract

[The composition builder](../data/item47_rung1_mix.py) pins private dataset
`frontier-infra/jebadiah-data-v2-1-item32`, revision
`bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed`, folder `a3/`, with immutable train,
calibration and manifest hashes. The live preview replaces **1,000 of 21,190 presentations
(4.719207%)**, below the aggregate upstream cap of **2,119 (10%)**. No prior item47 exposure
exists in the pinned A3; the builder rejects unexpected pre-existing exposure instead of
renaming it. All variants of this generator count as the same source.

Incoming questions have area `language`; 500 `choice` and 500 `noul` questions replace
matching area/type slots. The live matched donor pool is 4,547 choice and 1,031 Boolean
slots. If either pool is smaller, that type's quota decreases without borrowing unmatched
slots. The complete area/type histogram stays unchanged, including unassigned legacy
rows. Donor/source/type histograms are recorded before and after, aggregating per-record
locators while retaining source names and revisions; the final builder also
writes an old-question-to-new-row slot map. Unchanged questions in mixed rows retain
their state, question, label and soft target. Calibration is copied byte for byte with its
original **512 questions**. The 300-question `evidence-diagnostic.jsonl` is separate from
training and calibration and must never fit temperatures.

Selection rotates rules and statuses across world numbers, choosing the same worlds for
both types. Each training rule contributes 200 presentations. The 1,000-question subset
contains 340 supported, 330 refuted and 330 insufficient questions, 490 negated claims,
and 200/600/200 questions from two/three/four-document worlds. It comprises 165 complete
six-question flip groups and five supported-only groups needed to reach exactly 1,000.
Diagnostics retain all six siblings in every held-out world.

The unchanged Jeb `Renderer` and pinned Qwen3.5-9B tokenizer were used for all 6,300
questions in canonical, reversed and seeded-shuffled option orders: **18,900 renders,
zero truncation**, maximum **785 choice / 752 Boolean prompt tokens**. Prompt budget is
1,984 with a 64-token padding reserve under the unchanged 2,048 sequence limit. All 13 item47 tests and 32 item32/item33/item43 regression tests passed. A live CPU
`DecideCollator` check validates shuffled gold-wire-key mapping, Boolean labels, soft targets
and candidate padding. No renderer, serving, optimizer, epoch, learning-rate, rank,
temperature, objective or training-step change is proposed.

The entire 21,190-question preview was rendered with the same budget and complete
candidate keys. Its maximum prompt is 1,984 tokens; zero incoming questions truncate.
The pinned original A3 has 43 truncating questions; one is a selected donor, leaving
42 inherited truncations handled by the unchanged renderer. Every retained state,
question, label and soft target was verified against the pinned A3. This receipt does not
claim that legacy A3 is free of truncation or its existing private contamination exceptions.

## Atlas verification and exact Studio gate

Atlas used an existing CPU Python environment containing Transformers, Jinja2, xxhash
and CPU torch. Download only pinned tokenizer files and pinned A3 data, never weights.
Use an item47 scratch directory; large files must stay on the external scratch volume.
The following are reproducible Atlas commands with those dependencies installed:

```sh
export TMPDIR=/data/orca/scratch/item47
export HF_HOME=/data/orca/toolchains/hf
mkdir -p "$TMPDIR"
hf download Qwen/Qwen3.5-9B tokenizer.json tokenizer_config.json chat_template.jinja merges.txt vocab.json --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir "$TMPDIR/item47-tokenizer"
hf download frontier-infra/jebadiah-data-v2-1-item32 --repo-type dataset --revision bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed --include 'a3/*' --local-dir "$TMPDIR/item47-a3-download"
ITEM47_TOKENIZER="$TMPDIR/item47-tokenizer" python3 -m unittest discover -s data -p test_item47_evidence_worlds.py -v
python3 data/item47_evidence_worlds.py --out "$TMPDIR/item47-evidence" --tokenizer "$TMPDIR/item47-tokenizer"
```

Without `--index`, status is `pending_studio_scan`; the final composer refuses these
artifacts. The lead runs this exact preparation and scan/composition command on Studio
from this checkout, with an installed CPU Python environment. Both build output directories
must be empty. Stop if the protected SSD path is unavailable; do not fall back to `/tmp`.
The index SHA256 must be
`cf54ade9013c05db74f4c70925287382708de965fbf3cba62145ee821adad0ff`.

```sh
test -f /Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl || exit 1
export TMPDIR=/Volumes/PRO-G40/scratch/item47
export HF_HOME="$TMPDIR/item47-hf"
mkdir -p "$TMPDIR"
hf download Qwen/Qwen3.5-9B tokenizer.json tokenizer_config.json chat_template.jinja merges.txt vocab.json --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir "$TMPDIR/item47-tokenizer"
hf download frontier-infra/jebadiah-data-v2-1-item32 --repo-type dataset --revision bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed --include 'a3/*' --local-dir "$TMPDIR/item47-a3-download"
python3 data/item47_evidence_worlds.py --out /Volumes/PRO-G40/scratch/item47/item47-evidence-scanned --tokenizer /Volumes/PRO-G40/scratch/item47/item47-tokenizer --index /Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl
python3 data/item47_rung1_mix.py --base /Volumes/PRO-G40/scratch/item47/item47-a3-download/a3 --evidence /Volumes/PRO-G40/scratch/item47/item47-evidence-scanned --out /Volumes/PRO-G40/scratch/item47/item47-rung1
```

The composer binds the complete scan report to the candidate hash, count, pinned index
and scanner hash, requires zero removed records/hits and source-level rejection, verifies
frozen licenses/data/render contracts, audits executable labels and full finite-source
identity, rejects ID/family/state leakage and checks aggregate exposure. Passing artifact
checks creates the final composition, copies calibration unchanged, and exports its hashes
and replacement map. An overlap fails the entire source; no record-only salvage.

Before writing more than 10 GB on Studio, run `df -h /System/Volumes/Data` and stop if free
space is below 50 GB. After the lead saves nonprivate receipts, delete item47 scratch,
including tokenizer/A3 downloads and the item-local HF cache. Do not export benchmark
payloads or retain models. Atlas scratch is removed before this worker settles.

## Future measurement, not authorization

Rung 1 starts independently from the pinned 9B parent with the unchanged A3 recipe.
Hypothesis: cross-document proof joins and minimal fact flips improve evidence sensitivity
on HoVer/RAGTruth and on the separate owned diagnostic, without harming the overall proxy.
Keep only if the frozen 10% proxy paired whole-group bootstrap **95% interval of rung 1
minus A3 (46.97) is entirely above zero**, score exceeds **44.67**, and the source-specific
independent diagnostic improves. Otherwise drop. Inspect all five areas and record any
regression. Preserve the unchanged 0.05 export gate and local-hardware CI policy.

For the independent diagnostic, use the existing local evaluator and unchanged A3
per-type temperatures. Report three-way accuracy, Boolean support precision/recall,
per-rule/document-count/polarity accuracy, and complete-group correctness across the
three minimal variants. Improvement means higher macro accuracy over the five held-out
rule families using the same questions for A3 and the candidate; also report pair/group
correctness so a local win cannot hide failure to react to fact flips. These diagnostic
results are required alongside the proxy keep rule, not a replacement for it.

This PR includes a pre-run line in [the run log](run-rules.md). It authorizes no training,
proxy model run, GPU job or release. Existing private A3 source exceptions still require
a separately measured scrub before public shipment.
