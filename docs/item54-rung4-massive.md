# Item 54: rung 4 MASSIVE en-US intent data

CPU data work only. No model calls, generated utterances, teacher labels, audio,
training, inference, evaluation runs or GPU spend. This implements both rung 4
composition forms against pinned private A3. The committed receipt is a render
and composition preview, with **pending Studio overlap clearance**. The worker
never opened or copied the protected index. Final composition refuses pending,
rejected, incomplete, mismatched or unbound scan receipts.

## Admission and provenance

MASSIVE 1.1 is CC BY 4.0 data; this repository's converter is Apache-2.0.
Original license bytes, release notices, authors, attribution, commercial-use and
model-redistribution permission, data hashes, metadata hashes, revision and
changes are recorded in `data/manifests/item54-source-licenses.json`.
`NOTICE-MASSIVE.txt` accompanies every candidate and composed dataset.
The source metadata is frozen at AmazonScience/massive revision
`ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`; the fixed 1.1 archive has its own
SHA256 pin. Metadata revision alone is not used to trust the S3 archive.

Live TinyFish checked the edition 0.3 source catalog and complete manifest at
Decision Index revision `9eb2dbe2a358004c8782c66e40a83ac07b953fec`, plus ToolRet's
upstream task map at `c4181d9`. HWU64, SLURP and MASSIVE are absent by name.
MASSIVE's original notice identifies SLURP English text as its parent. Section
3.1 of the SLURP paper identifies the same original HWU collection released by
Liu et al. (2019). HWU64 is a subset of that collection. Decision Index's home
appliance simulator is independently generated, not a download of this corpus.
ToolRet's pinned source map contains none of this lineage. The manifest retains
CLINC150, BANKING77, SGD, ToolRet and all item46 source/ancestor exclusions.

The lead resolved the item43/item46 policy difference: item43's extra HWU name
exclusions concern N2 taxonomy reuse, rather than contamination. Item54 follows
item46 and treats **HWU, HWU64, SLURP, MASSIVE and every translation or variant as
one source with a fixed 2,119-question cap**. An excluded ancestor would reject
the entire source. This provenance check is not an overlap clearance.

A live source audit against SLURP revision
`8eb16545762be97ace75334109d73824217311f1` found all 16,521 MASSIVE en-US IDs in
SLURP with identical original partitions. At that inspected parent revision,
15,374 utterances and 16,292 intent labels match exactly: 1,147 texts and 229
labels differ. These differences are recorded, not silently rewritten. MASSIVE
1.1 supplies the converter's text and gold. Full evidence and parent-file hashes
are in `results/research/item54-source-evidence.json`.

## Candidates and prompt contract

The released en-US pool is 11,514 train, 2,033 dev and 2,974 test. Only original
**train** rows supply the 6,000 training candidates and 300 separate diagnostic
questions. Dev/test text is never selected as a candidate. The entire frozen
release, all 52 locales and all splits, supplies the lead's raw scan input:
859,092 records. Scanning every locale conservatively covers the upstream source
rather than clearing only a sample or only the English derivative.

MASSIVE's ID maps to its original SLURP ID. A3's 102 inherited MASSIVE rows were
resolved by both utterance and their original train-row locators; all original
IDs and normalized utterances already exposed in A3 train or calibration are
excluded. Another two A3 text collisions were excluded. All duplicated or empty
en-US utterance groups are excluded across the complete en-US release before
selection, including repeats across original splits. Exactly 123 train rows
were excluded for duplicate/empty utterances and 104 for A3 exposure.

Each original ID receives one question type, and every ID appears in only one
candidate split. Diagnostic selection precedes training selection, by a seeded
round robin over all 60 intents. Both splits cover all 60 intents. Train has
3,000 choice and 3,000 yes/no; diagnostic has 150 of each. The held-out diagnostic
is original-ID and normalized-utterance disjoint, not intent-disjoint. It tests
new human utterances for the same intent catalog.

Choice uses a 16-option menu from the original 60 intent names, mechanically
replacing underscores with spaces. Same-scenario intent neighbors receive
priority as distractors. Yes/no verifies whether a proposed original intent
matches the source annotation. False verification labels use another known
intent, never an invented OOS label or an uncertain request relabeled as OOS.
No native OOS performance is claimed. This is not N2's wide-menu intervention.

The existing pinned Qwen tokenizer, Jeb Renderer, chat template and prompt
contract are unchanged. All 6,300 rows were rendered: choice in canonical,
reversed and seeded-shuffled order, plus the ordinary yes/no order. **12,600
renders passed with zero truncation**, complete distinct candidate token IDs,
and maximum prompt lengths of 288 choice / 119 yes/no tokens. The prompt budget
is 1,984 plus a 64-token reserve within the unchanged 2,048-token limit.

## Both composition forms

A3 is `frontier-infra/jebadiah-data-v2-1-item32` at
`bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed`, folder `a3/`. All input hashes,
including its manifest, are checked. The original 512-question calibration file
is copied byte for byte; diagnostic data stays separate. Temperatures are not
refitted. Existing private A3 contamination exceptions still prevent shipment.

The plan requests 1,000 matched replacements, reduced when the donor pool is
smaller. A3 has only **749 retrieval choice/yes-no donor slots**, 583 choice and
166 yes/no. Both forms use the exact same deterministic 749 incoming rows.
No extra language or unassigned-area donor is used to reach 1,000.

| Form | Presentations | Incoming | Removed | HWU/SLURP/MASSIVE exposure | Cap |
|---|---:|---:|---:|---:|---:|
| Replace matched A3 slots | 21,190 | 749 | 749 | 749 | 2,119 |
| Add to untouched A3 | 21,939 | 749 | 0 | 851 | 2,119 |

Replacement removes all 102 inherited MASSIVE presentations among the matched
donor pool and preserves A3's complete area/type histogram. Mixed-question rows
keep their original state and every untouched question, label and target.
Addition preserves every original training byte in its original order, then
appends the same 749 selections. It retains the inherited 102 presentations;
its increased exposure is an explicit experimental difference. The fixed cap
remains 2,119 even though the larger additive denominator could permit more.
The composition proof records donor IDs, incoming IDs, counts and histograms.

The builder binds the source/converter/license hashes, original candidate rows,
labels, partition, ancestry evidence, notices, renderer contract and both scan
phases. It regenerates selection from original train bytes and requires exact
candidate equality. The raw scan covers all 859,092 release rows, and the
converted scan covers all 6,300 candidates with source-level rejection. An
upstream hit rejects MASSIVE entirely, without record-only salvage. The lead's
scanner verifies the exact Studio index path and SHA256 and uses a temporary
symlink that is removed on success or failure; no index copy is made.

## Reproduction and exact lead-only Studio commands

Run from this checkout with the existing authenticated `hf` session. Downloads
are source data, tokenizer files and pinned A3 only. Preparation verifies every
frozen input hash. Scratch is removed when the worker finishes, so the following
commands recreate it. No model weights are downloaded.

```sh
export TMPDIR=$(~/bin/pro-g40-scratch item54)
export UV_CACHE_DIR="$TMPDIR/item54-uv-cache"
python3 data/item54_fetch.py --scratch "$TMPDIR"
uv venv "$TMPDIR/item54-venv"
uv pip install --python "$TMPDIR/item54-venv/bin/python" transformers torch xxhash jinja2 huggingface-hub ijson
"$TMPDIR/item54-venv/bin/python" data/item54_massive.py --raw "$TMPDIR/item54-raw" --base "$TMPDIR/item54-a3-download/a3" --out "$TMPDIR/item54-preview-release" --tokenizer "$TMPDIR/item54-tokenizer"
ITEM54_SCRATCH="$TMPDIR" "$TMPDIR/item54-venv/bin/python" -m unittest discover -s data -p 'test_item54*.py' -v
```

The **lead alone** runs the following exact scan and final composition commands.
The scan must report zero hits/removals in both phases, a passed source-level
status, 859,092 raw records and 6,300 converted records. The two final datasets
are created only after that clearance. Existing nonempty output directories
are refused, so a failed scan cannot be silently resumed as a passed build.

```sh
export TMPDIR=$(~/bin/pro-g40-scratch item54)
"$TMPDIR/item54-venv/bin/python" data/item54_massive.py --raw "$TMPDIR/item54-raw" --base "$TMPDIR/item54-a3-download/a3" --out "$TMPDIR/item54-scanned" --tokenizer "$TMPDIR/item54-tokenizer" --index /Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl
"$TMPDIR/item54-venv/bin/python" data/item54_rung4_mix.py --raw "$TMPDIR/item54-raw" --base "$TMPDIR/item54-a3-download/a3" --incoming "$TMPDIR/item54-scanned" --out "$TMPDIR/item54-rung4-replace" --mode replace
"$TMPDIR/item54-venv/bin/python" data/item54_rung4_mix.py --raw "$TMPDIR/item54-raw" --base "$TMPDIR/item54-a3-download/a3" --incoming "$TMPDIR/item54-scanned" --out "$TMPDIR/item54-rung4-add" --mode add
```

Item54 tests cover actual source-to-gold conversion, ancestry exclusions, original
IDs, duplicate-group rejection, matched donors, cap aggregation, both writers,
calibration bytes, additive prefix bytes, partial-scan rejection, whole-source
rejection and symlink cleanup. Positive final-writer tests use explicitly
synthetic scan receipts in temporary fixtures, not real contamination clearance.
The committed summary separately records actual CPU previews and gate refusals.
The item33/item43/item49/item51 pattern regressions were also run locally.
No GitHub-hosted CI or workflow was added.

## Hypothesis, diagnostic and future run log

Hypothesis: more independently held-out human intent requests improve CLINC and
retrieval after the CLINC loss in rungs 1 through 2b. Evaluate the two forms
separately from the pinned A3 parent, holding seed, recipe, optimizer, objective,
rank, LR, sequence length, menu contract and temperatures fixed. Replacement
preserves exposure; addition changes total presentations and must be reported
as an exposure experiment. No training is authorized or launched by this PR.

The primary independent diagnostic is macro intent accuracy on the 150 held-out
choice questions, covering all 60 intents. Also report ordinary accuracy by
type, intent and scenario, and yes/no precision, recall and balanced accuracy.
Use the existing local Scorer through `eval_jebadiah.run_set`, full-menu candidate
logits and unchanged A3 temperatures. `scripts/item54_diagnostic_report.py`
provides an offline report wrapper following the item48 fix: persist **only
`extra['timing']`**, never the evaluator's tuple-keyed question map, and save
**each completed pass** atomically. This wrapper launches no evaluation.

Keep a form only if its frozen proxy paired whole-group bootstrap 95% interval
of candidate minus A3 (46.97) is entirely above zero, its retrieval-area interval
is also above zero, its score exceeds v2's 44.67, and the primary independent
intent diagnostic improves. Inspect all five areas and record regressions.
A local intent gain alone cannot pass the keep rule.

| Run log | A3 | Candidate | Delta vs A3 / paired 95% CI | Independent intent diagnostic | Decision |
|---|---:|---:|---|---|---|
| Rung 4 replace vs A3 | 46.97 | Not run | Pending | Not run | Pending Studio scan and separate training authorization |
| Rung 4 add vs A3 | 46.97 | Not run | Pending | Not run | Pending Studio scan and separate training authorization |

After the lead exports required receipts and private compositions, remove only
item54's scratch directory. The worker deleted its downloads, tokenizer, A3
copy, SLURP parent checkout, local environment and previews at completion.
No merged model existed or was downloaded for this task.
