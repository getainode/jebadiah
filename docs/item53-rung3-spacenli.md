# ITEM 53: SpaceNLI, 9B rung 3

Data preparation only. No training, evaluation, model calls, model weights or GPU use.
The Apache-2.0 converter preserves upstream text and expert labels. Its private output
uses the existing language-area `choice` contract: entailment, contradiction and neutral.

Live TinyFish verified the [pinned MIT notice](https://github.com/kovvalsky/SpaceNLI/blob/8c10e94d238737be97142f5a7ffdffa49a6a6ab9/LICENSE),
[upstream construction](https://github.com/kovvalsky/SpaceNLI) and
[paper, section 3](https://aclanthology.org/2023.naloma-1.2.pdf).
The initial 56 spatial inference problems came from Nam (1995) and Zwarts and Winter
(2000), followed by expert revisions, NP placeholders, a toy grammar, a 171-entity
mini-world and selection restrictions. The paper's experiments use models trained on
other NLI corpora; those training corpora and model predictions are not the ancestry
of the SpaceNLI examples. No excluded Decision Index 0.3 ancestor was found in this
construction evidence. This is a provenance screen, not overlap clearance or proof
about unavailable private tests.

The source is `kovvalsky/SpaceNLI` at `8c10e94d238737be97142f5a7ffdffa49a6a6ab9`.
Upstream publishes one **unsplit 32,000-row file**, not an original train/test split.
The lead explicitly authorized that original file and deterministic local partitioning
because the original-train restriction excludes imported benchmark test splits.
No Tasksource mirror rows are imported. All upstream file hashes, attribution,
MIT notice and commercial-use permission are in
[the licence manifest](../data/manifests/item53-source-licenses.json);
[live evidence](../results/research/item53-source-evidence.json) records the findings.

## Counts and isolation

Related alphabetic pattern variants share their numeric seed. XML sibling groups
also share one family, including groups that span numeric seeds. All families exposed
by A3 train or calibration are excluded from incoming data. Every existing SpaceNLI
state must map back to upstream premises and hypotheses; unknown mapping fails closed.
This conservative grouping leaves 1,800 unique unused-pattern rows across four families.
Two families are reserved for the diagnostic, including all their unselected siblings.

| Split | Questions | Entailment | Contradiction | Neutral | Families |
|---|---:|---:|---:|---:|---:|
| Training candidates | 1,200 | 600 | 200 | 400 | 2 |
| Separate diagnostic | 300 | 150 | 0 | 150 | 2 |
| Reserved diagnostic siblings, unused | 300 | | | | |

The 6,000 target is an upper bound; unused families cannot support that many candidates.
Only one remaining family has contradiction examples. It stays in training to preserve
all three training labels. Consequently this independent diagnostic **does not measure
contradiction accuracy**, and its narrow coverage limits any later claim of transfer.
The diagnostic is never training data or temperature-fitting calibration.

The pinned Qwen tokenizer and chat template passed 4,500 renders: canonical, reversed
and seeded shuffled option order for every selected question. Maximum prompt length is
118 tokens, zero truncations, with the unchanged 1,984-token prompt budget and 64-token
reserve within 2,048. Prompt, template and tokenizer hashes are in
[the CPU build receipt](../data/manifests/item53-build-summary.json).

## Both composition forms

Both modes select the same deterministic 1,000 incoming questions: 502 entailment,
163 contradiction and 335 neutral. A3 already contains 102 SpaceNLI presentations.
The aggregate exposure is 1,102, below the conservative 2,119 cap calculated from the
original A3 size. Source aliases and variants count together.

| Mode | Original presentations | Incoming | Removed | Final presentations | Source share of final |
|---|---:|---:|---:|---:|---:|
| `replace` | 21,190 | 1,000 | 1,000 | 21,190 | 5.2006% |
| `add` | 21,190 | 1,000 | 0 | 22,190 | 4.9662% |

Replacement uses only language-area choice slots, with 4,462 eligible donors, excluding
existing SpaceNLI. Other questions in mixed rows retain their original state, labels
and targets. Addition preserves the complete original train bytes as a prefix.
Both copy the original 512-question calibration byte for byte. ID, family and normalized
state checks reject collisions with new candidates or the diagnostic. Existing private
A3 contamination exceptions remain a separate shipping blocker.

The committed counts are previews. **Neither final composition is built until the Studio
full-source scan passes.** The gate binds the receipt to the frozen 64,000-record scan
file, current converter/scanner, upstream hashes, selected row membership, original A3
hashes, licence manifest and complete render checks.

## Reproduction and exact Studio commands

On Atlas, only tokenizer files, upstream data and the small pinned A3 files were downloaded.
Dependencies are `transformers`, `jinja2` and `xxhash`; no torch or model loading is needed.

```sh
export TMPDIR=/data/orca/scratch/item53
export HF_HOME=/data/orca/toolchains/hf
mkdir -p "$TMPDIR"
git clone https://github.com/kovvalsky/SpaceNLI.git "$TMPDIR/item53-spacenli"
git -C "$TMPDIR/item53-spacenli" checkout 8c10e94d238737be97142f5a7ffdffa49a6a6ab9
hf download Qwen/Qwen3.5-9B tokenizer.json tokenizer_config.json chat_template.jinja merges.txt vocab.json --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir "$TMPDIR/item53-tokenizer"
hf download frontier-infra/jebadiah-data-v2-1-item32 --repo-type dataset --revision bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed --include 'a3/*' --local-dir "$TMPDIR/item53-a3"
PYTHONPATH=data:train python3 -m unittest discover -s data -p test_item53_spacenli.py -v
PYTHONPATH=data:train python3 data/item53_spacenli.py --upstream "$TMPDIR/item53-spacenli" --base "$TMPDIR/item53-a3/a3" --tokenizer "$TMPDIR/item53-tokenizer" --out "$TMPDIR/item53-built"
```

The lead runs these commands on Studio from this PR checkout. All outputs must be new
empty directories. Use the external SSD; no protected text leaves Studio. The builder
checks the exact index path and SHA256 before loading it, symlinks it temporarily and
removes the symlink even on failure. Scan all 32,000 raw source rows **and** all 32,000
mapped states/instructions/options, before sampling. A single overlap rejects the entire
source and aborts composition; record-only rescue is prohibited.

```sh
export TMPDIR=/Volumes/PRO-G40/scratch/item53
mkdir -p "$TMPDIR"
git clone https://github.com/kovvalsky/SpaceNLI.git "$TMPDIR/item53-spacenli"
git -C "$TMPDIR/item53-spacenli" checkout 8c10e94d238737be97142f5a7ffdffa49a6a6ab9
hf download Qwen/Qwen3.5-9B tokenizer.json tokenizer_config.json chat_template.jinja merges.txt vocab.json --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir "$TMPDIR/item53-tokenizer"
hf download frontier-infra/jebadiah-data-v2-1-item32 --repo-type dataset --revision bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed --include 'a3/*' --local-dir "$TMPDIR/item53-a3"
PYTHONPATH=data:train python3 data/item53_spacenli.py --upstream "$TMPDIR/item53-spacenli" --base "$TMPDIR/item53-a3/a3" --tokenizer "$TMPDIR/item53-tokenizer" --out "$TMPDIR/item53-scanned" --index /Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl
PYTHONPATH=data:train python3 data/item53_nli_mix.py --base "$TMPDIR/item53-a3/a3" --candidates "$TMPDIR/item53-scanned" --out "$TMPDIR/item53-replace" --mode replace
PYTHONPATH=data:train python3 data/item53_nli_mix.py --base "$TMPDIR/item53-a3/a3" --candidates "$TMPDIR/item53-scanned" --out "$TMPDIR/item53-add" --mode add
```

No training command is authorized here. For a separately authorized run, the hypothesis
is that unused spatial patterns improve inference without harming overall decisions.
Compare each mode independently with A3 (46.97), using the frozen 10% proxy and paired
whole-group bootstrap. Keep only if the 95% difference interval is wholly above zero,
the score exceeds v2's 44.67 and the separate diagnostic improves. Inspect all five areas.
Addition also changes epoch exposure and total size; it cannot isolate replacement effects.
Hold the A3 renderer, temperatures, sequence length, objective and recipe fixed, subject
to the lead's separately frozen additive exposure policy. Preserve the 0.05 export gate.
Use the item48 report wrapper if evaluation is later authorized; no evaluation runs here.
