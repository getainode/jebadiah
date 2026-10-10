# Item 55: rung 5 fresh DeepMind multi-step arithmetic

CPU data preparation on the Mac Studio, with no model calls, training, inference,
GPU spend or benchmark index access by the worker. Both final composition forms
are implemented and refuse to run until the lead completes the Studio scan.
The committed summary contains live render checks and composition previews,
not overlap clearance or a measured model result.

## Frozen source and rights

Use the unmodified `google-deepmind/mathematics_dataset` generator at commit
`427f45075f84b8b9774950196ad63867ca20ffb3`, with fixed seed `55020261010`.
No pre-generated release was downloaded. TinyFish live extraction verified the
pinned repository, Apache-2.0 license, arithmetic module and edition 0.3 source
catalog. The catalog contains GSM8K and CLadder but none of the inspected
DeepMind source names. Local inspection of the pinned generator traced the
selected modules to internal procedural number and expression sampling, without
an imported question corpus. This is a provenance screen, not a lexical overlap
clearance or a claim about base-model pretraining.

The license manifest freezes every upstream Python file, license, README, setup,
converter, live evidence receipt and the complete finite generated universe.
Apache-2.0 allows commercial use, derivative works and redistribution subject to
its license and attribution requirements. The source is attributed to DeepMind
Technologies Limited and David Saxton, Edward Grefenstette, Felix Hill and
Pushmeet Kohli. Every candidate and final composition carries the original
`LICENSE-DeepMind.txt` and `NOTICE-DeepMind.txt`. The converter and offline report
helper are Apache-2.0. Item46's entire source/ancestor exclusion list applies;
unknown ancestry or any overlap rejects the whole source without row salvage.

The selected modules are `add_sub_multiple`, `mul_div_multiple` and `mixed`.
Each generates 4,000 questions, with three to seven arithmetic operations and
entropy between 6 and 10. Calls use one pure module and no external entity
replacement. Exact integer/rational questions and original upstream wording are
preserved. An independent whitelist Python AST interpreter using `Fraction`,
plus a separate operation-tree interpreter, must agree with upstream gold on
all 12,000 questions. Neither interpreter uses floating point, `eval`, model
predictions or benchmark answers.

These are symbolic exercises rather than natural transaction word problems.
Transfer to GSM8K is a hypothesis. The builder does not paraphrase GSM8K or add
model-written stories. Its answer offsets are deterministic arithmetic, not
plausible human error annotations.

## Candidate splits, conversion and renderer

The complete finite source contains 12,000 fresh questions. All raw questions,
expressions and answers, and all corresponding converted states, instructions
and answer options form the **24,000-record scan input before sampling**. An
infinite generator cannot be enumerated; the admitted source is this frozen
finite universe, with no downloaded upstream train/test release. The source
and scan stream hashes prevent quietly changing this universe later.

Training has 6,000 candidates: 2,000 per module, with 3,000 four-answer `choice`
and 3,000 `noul` exact-answer verification questions. The separate diagnostic
has 300 candidates: 100 per module, with 150 of each type. Each original problem
gets exactly one type and appears in only one split. Choice distractors are
exact answer plus 1, minus 1 and plus 2, giving four distinct rational values.
Noul verifies a proposed exact answer or a deterministic nonzero offset.

Families use the global ordered operation tree with numeric constants and
numeric signs erased. Associative Add/Mul nodes are flattened because the
upstream renderer hides their grouping. Module names and wording variants do
not create separate families. A hash of this structure reserves one fifth of
families for diagnostic, across all three modules. Training has 2,729 selected
structures; diagnostic has 189, with zero overlap. Unselected siblings of held
structures never enter training. This tests unseen operator programs while
retaining the same upstream wording families, rather than claiming a wording
or vocabulary holdout.

The original Qwen tokenizer, Renderer and prompt contract are pinned unchanged.
All 6,300 selected questions were rendered, choice in canonical, reversed and
seeded-shuffled order and noul in its ordinary order. **12,600 renders passed,
zero truncation, maxima 153 choice and 135 noul tokens**. Candidate token IDs
were complete and distinct. Budget remains 1,984 prompt tokens plus a 64-token
reserve inside the original 2,048-token maximum.

## Two forms against A3

A3 is the private dataset `frontier-infra/jebadiah-data-v2-1-item32`, revision
`bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed`, folder `a3/`. Train, calibration
and manifest hashes are checked. Its arithmetic/reasoning rows use `knowledge`,
as in item49; there is no `reasoning` area in this pinned A3. The matched donor
pool has 2,119 knowledge-choice and 444 knowledge-noul presentations. Deterministic
allocation selects 556 choice and 444 noul, with the exact same incoming IDs
in both forms. Existing DeepMind mathematics exposure is zero. All known
DeepMind mathematics source-name variants aggregate into one cap.

| Form | Presentations | Incoming | Removed | Source exposure / fixed A3 cap |
|---|---:|---:|---:|---:|
| Matched replacement | 21,190 | 1,000 | 1,000 | 1,000 / 2,119 |
| Addition to intact A3 | 22,190 | 1,000 | 0 | 1,000 / 2,119 |

The source fraction is 4.7192% of the original A3 in both forms. Replacement
preserves its complete area/type histogram. Mixed donor rows retain their
state and every untouched question, label and target. Addition retains every
original train byte in order and appends the same selected rows. Addition
increases training exposure by design and must be evaluated as its own run.
Both forms copy A3's 512-question calibration byte for byte and keep the 300
math diagnostic questions separate. Temperatures are unchanged. Existing
private A3 exceptions still block public model shipment.

The builder checks source rights, source/scan hashes, all raw and converted
records, deterministic selection, candidate hashes, full scanner version and
zero-hit receipt, render contract, ID/family/state isolation and aggregate cap.
Pending, rejected, truncated, incomplete or unbound receipts fail before any
final output directory is created. Preview calculations do not bypass this gate.

## Reproduction and exact lead-only Studio scan command

The worker removes all item55 scratch after PR delivery, including generated
source/candidates, downloads and the CPU environment. The following reconstructs
those inputs from pinned revisions. Run from this PR checkout on the Studio.
If the SSD scratch helper fails, stop without falling back to the internal disk.
No weights are downloaded. The existing authenticated git/HF sessions are used.

```sh
set -euo pipefail
TMPDIR=$(~/bin/pro-g40-scratch item55)
export TMPDIR
export UV_CACHE_DIR="$TMPDIR/item55-uv-cache" HF_HOME="$TMPDIR/item55-hf-cache"
export PYTHONPATH="$PWD/data:$PWD/train:$PWD/scripts"
uv venv --python 3.11 "$TMPDIR/item55-venv"
uv pip install --python "$TMPDIR/item55-venv/bin/python" -r data/manifests/item55-cpu-requirements.txt
git clone https://github.com/google-deepmind/mathematics_dataset.git "$TMPDIR/item55-upstream"
git -C "$TMPDIR/item55-upstream" checkout 427f45075f84b8b9774950196ad63867ca20ffb3
hf download Qwen/Qwen3.5-9B tokenizer.json tokenizer_config.json chat_template.jinja merges.txt vocab.json --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir "$TMPDIR/item55-tokenizer"
hf download frontier-infra/jebadiah-data-v2-1-item32 --repo-type dataset --revision bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed --include 'a3/*' --local-dir "$TMPDIR/item55-a3-download"
```

Only the lead runs the next commands. The exact protected index path and SHA256
are checked before loading; the scanner temporarily symlinks the index and
removes that symlink on success or failure. It never copies the index or exports
benchmark payloads. The output must be empty. All 24,000 records must pass,
otherwise neither form is admissible.

```sh
"$TMPDIR/item55-venv/bin/python" data/item55_mathematics.py --upstream /Volumes/PRO-G40/scratch/item55/item55-upstream --base /Volumes/PRO-G40/scratch/item55/item55-a3-download/a3 --tokenizer /Volumes/PRO-G40/scratch/item55/item55-tokenizer --out /Volumes/PRO-G40/scratch/item55/item55-scanned --index /Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl
"$TMPDIR/item55-venv/bin/python" data/item55_rung5_mix.py --base /Volumes/PRO-G40/scratch/item55/item55-a3-download/a3 --incoming /Volumes/PRO-G40/scratch/item55/item55-scanned --out /Volumes/PRO-G40/scratch/item55/item55-rung5-replace --mode replace
"$TMPDIR/item55-venv/bin/python" data/item55_rung5_mix.py --base /Volumes/PRO-G40/scratch/item55/item55-a3-download/a3 --incoming /Volumes/PRO-G40/scratch/item55/item55-scanned --out /Volumes/PRO-G40/scratch/item55/item55-rung5-add --mode add
```

For a worker-side render-only build, omit `--index` and choose another empty
output folder; its manifest remains `pending_studio_scan`. Tests on the complete
frozen build use:

```sh
ITEM55_CANDIDATES="$TMPDIR/item55-scanned" ITEM55_UPSTREAM="$TMPDIR/item55-upstream" "$TMPDIR/item55-venv/bin/python" -m unittest discover -s data -p 'test_item55*.py' -v
"$TMPDIR/item55-venv/bin/python" -m unittest test_item33_skill_data test_item49_corr2cause test_item53_spacenli test_item54_massive test_item54_diagnostic_report -v
```

On the Studio, all 20 item55 tests passed with the complete finite source,
including seed replay, independent exact oracles, structure isolation, both
composition forms, retained mixed rows, unchanged additive bytes, pending-scan
refusal and corrupt scan-receipt rejection. The pattern regression run passed
66 tests and skipped five optional old-source integration checks. Its synthetic
scanner fixtures do not access the protected index. A first regression attempt
needed the existing item49 dependency `ijson`; the frozen CPU requirements
include it and the complete rerun passed.

## Hypothesis, diagnostic and future run log

Hypothesis: explicit multi-step integer/rational operation programs improve
arithmetic/reasoning decisions, while acknowledging limited word-problem
coverage. Each future experiment starts from the pinned 9B parent and original
A3 recipe, with fixed renderer, options, optimizer, LR, rank, seed, sequence
length, temperatures and objective. No training or model evaluation is launched
or authorized by this data PR. Replacement and addition must be compared
independently, not treated as cumulative rungs.

The frozen proxy uses paired whole-group bootstrap against A3 (46.97). **Keep
only when the overall paired 95% interval of candidate minus A3 is entirely
above zero, score exceeds 44.67, and the separate 300-question diagnostic's
choice macro module accuracy improves**. Otherwise drop. Report all five proxy
areas and arithmetic/verification diagnostics, including regressions.

Future diagnostic scoring uses the existing local `eval_jebadiah.run_set`,
complete menus, a single pass and unchanged A3 temperatures. The offline
`item55_diagnostic_report.save_completed_pass` repeats item48's fixed wrapper:
it saves each completed pass with only `extra['timing']`, discarding the internal
tuple-keyed question map that cannot be JSON serialized. It reports overall,
question type, module and structure accuracy, choice macro module accuracy,
and verification precision/recall/balanced accuracy. It performs no model
loading, training, inference or external writes.

| Run log | A3 | Candidate | Delta / paired 95% CI | Diagnostic | Decision |
|---|---:|---|---|---|---|
| Rung 5 replace vs A3 | 46.97 | Not run | Pending | Not run | Pending lead Studio scan and separate training authorization |
| Rung 5 add vs A3 | 46.97 | Not run | Pending | Not run | Pending lead Studio scan and separate training authorization |
