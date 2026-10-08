# Item 33: local rung 3 skill data

Rung 3 is one additive data change for the measured A1/A3 winner. This PR builds
and verifies the addition only. No GPU job, teacher/model inference, dataset
publication or training launch is included. Neither A1 nor A3 has been selected
by this builder, and this work makes no claim of improved proxy performance.

## Sources and provenance

The source-level license ledger is
[`data/manifests/item33-source-licenses.json`](../data/manifests/item33-source-licenses.json).
Every source has an explicit license, URL, immutable content revision, ancestry
and commercial-use decision. Procedural sources also record the executable
Python generator hash, Apache-2.0 generator license, seed and empty external
input list. No synthetic teacher model or frontier API is used.

- **Intent routing:** Liu, Eshghi, Swietojanski and Rieser's original human
  home-domain utterance collection, CC BY 4.0, original repository revision
  `f6071b496b17d71e6eb43f543af0707f4ff30557`. The original README documents
  human collection/manual annotation and says the cross-validation files are
  shuffled conversions of the master tables. Both complete master tables are
  scanned, including annotations later rejected for training. Attribution and
  license-file hashes are in the ledger. The converter removes IRR annotations,
  empty utterances, conflicting annotations and normalized duplicates. It
  creates 16-option routing decisions and proposed-intent verification checks.
  The selected corpus covers 60 eligible original intent labels.
- **When to call a tool:** original finite worlds evaluated by an executable
  policy oracle. They cover forbidden writes, complete fresh context for the
  same record, stale or unrelated context, unavailable capabilities and missing
  required arguments. Five possible next actions and balanced proposed-action
  checks teach both tool use and avoiding unnecessary calls.
- **Grounding:** original finite evidence worlds with executable claim checks.
  Responses are prose generated from atomic claims. The oracle counts supported
  claims using entity identity, values, absent facts and explicit negation.
  A missing fact supports neither a positive nor a negative assertion. These
  are controlled short response-grounding examples, not broad factual QA.

Dataset research used `~/.local/bin/tinyfish search query` and
`~/.local/bin/tinyfish fetch content get` on the original repository and cards.
Bitext was not selected because its license is outside the existing v2.1
allowlist. xLAM and existing factuality mixtures were not selected because their
ancestry and generator licensing would need additional proof.

## Counts and family separation

| Skill | Train choice | Train noul | Train score | Calibration choice | Calibration noul | Calibration score | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Intent routing | 900 | 450 | 0 | 100 | 50 | 0 | 1,500 |
| Tool timing | 900 | 450 | 0 | 100 | 50 | 0 | 1,500 |
| Grounding | 450 | 450 | 450 | 50 | 50 | 50 | 1,500 |
| Total | 2,250 | 1,350 | 450 | 250 | 150 | 50 | 4,500 |

There are 4,050 training questions in 144 families and 450 calibration questions
in 16 families. Every question is a native Jeb `choice`, `noul` or `score`
request, with labels outside the model-visible question payload.

Human families contain every paraphrase of the same original elicitation
intent, including both question types. Each procedural family contains all
variants of a domain/template world. Family hashing assigns complete families;
procedural calibration is stratified to include all five scenario modes.
Round-robin sampling balances families within each skill and type. Calibration
includes all five tool actions. IDs and identical states are checked across
splits. No calibration sibling is selected for training.

## Source-level overlap gate

[`data/item33_full_suite_scan.py`](../data/item33_full_suite_scan.py) reuses the
item 25 scanner's functions and matching logic: canonical exact states,
normalized exact leaves of at least 80 characters, normalized 13-word spans,
and verified 5-word-shingle containment of at least 0.8 for leaves of at least
20 words. State, instructions and option/rubric text are scanned.

The item 25 protected index covers all 140,620 official suite requests before
scoring exclusions, including display-only sources and the rebuilt GSM8K
rows. Its pinned SHA256 is verified before deserializing it. The protected suite
payload hashes were checked live against the official edition 0.3 manifest used
by item 31. The scanner origin and index hashes are in the report. The blacklist
also includes protected source names, aliases, indirect parents and derivatives
from the v2.1 policy.

All source records are scanned before any selection. The item 25 family-removal
field is set to the upstream source ID so one hit rejects that entire source,
including every train/calibration candidate. A rejected source stops the build;
there is no row-level rescue or permission switch. Both raw human master tables
and every adapted human question are checked, together with each complete
finite procedural corpus.

| Source ID | Scanned raw and/or adapted records | Overlaps | Decision |
|---|---:|---:|---|
| hwu-original | 92,132 | 0 | retain |
| item33-tool-worlds | 1,500 | 0 | retain |
| item33-evidence-worlds | 1,500 | 0 | retain |
| Total | 95,132 | 0 | retain |

[`data/manifests/item33-overlap-scan-report.json`](../data/manifests/item33-overlap-scan-report.json)
contains IDs, hashes and counts only, without benchmark text or labels. Zero
scanner matches do not prove semantic independence or downstream improvement.

## Reproduce locally

Use a Python environment with `transformers`, `torch`, `tokenizers` and
`xxhash`. No CUDA runtime is needed. Fetch the original source with Git and
check out the ledger's pinned revision. The builder itself performs no network
operations. Use the private/local item 25 protected index and frozen 9B tokenizer.

```sh
git clone https://github.com/xliuhw/NLU-Evaluation-Data /tmp/item33-hwu
git -C /tmp/item33-hwu checkout f6071b496b17d71e6eb43f543af0707f4ff30557
python data/item33_skill_data.py \
  --hwu /tmp/item33-hwu \
  --out /tmp/item33-release \
  --index /tmp/item25-data/protected.pkl \
  --tokenizer /tmp/item25-data/tokenizer
python -m unittest discover -s data -p test_item33_skill_data.py -v
python data/lint_data.py --train \
  /tmp/item33-release/train.jsonl /tmp/item33-release/calib.jsonl
```

Use an empty output directory. Input file hashes, generator hashes, licenses and
ancestry are checked before building. Raw source tables, generated questions,
scan candidates and built train/calibration text remain local. Only code and
metadata are committed. If the item 25 index is unavailable, recreate it with
that item's `protect` stage against the complete official suite; do not replace
it with a proxy-only index or skip scanning.

The live build used `/tmp/item33-venv/bin/python`. All 4,500 selected questions
passed the frozen 9B canonical renderer, including reversed choice order, with
zero truncation. The longest prompt is 287 tokens under the 1,984-token budget
with 64 tokens reserved beneath v2's 2,048-token sequence length. The prompt
source, chat template and tokenizer hashes match v2's contract.

Twenty local tests pass. They exercise executable action precedence, wrong
entity/freshness boundaries, absent facts and negation, independent checks of
the rendered grounding prose, source/license and every protected-ancestry term,
deterministic family sampling, all tool actions in calibration, duplicate/state
leakage rejection, and a seeded 13-word overlap in an unselected option that
removes its whole source and clean siblings. Both built splits pass the training
linter. Compilation and whitespace checks pass. No GitHub-hosted CI was run.

## Applying the one change

After item 32 supplies its frozen proxy scores and paired interval, the lead
selects A3 only if its interval versus A1 is above zero; otherwise retain A1.
Add the 4,050 skill training questions to that pinned best mix, preserving its
recipe, original training rows, original calibration, rank, sequence length,
learning rate, ordinal targets and prompt contract. The separate 450-question
skill calibration split is held out for diagnostic use; it does not silently
replace the baseline temperature-fitting calibration.

The actual composed mix must verify the winner's input hashes, run cross-source
ID/state/family checks, and satisfy the source-level/license policy before any
shipping decision. PR 14 explicitly describes A1/A3 as private diagnostic mixes
with contamination exceptions; this clean addition does not turn those bases
into shipping candidates. No exception has been used for the three new sources.
Do not silently scrub the base or change another recipe knob inside the rung 3
comparison. Any required base cleanup needs a separately measured clean floor.

The run log records the proposed R3 comparison. Keep only if the paired complete-
group bootstrap 95% interval of R3 minus the selected best mix is above zero on
the unchanged frozen edition 0.3 proxy. Selection, composition, training and that
measurement remain with the lead; this PR deliberately contains no launcher.
