# Item 43: N2 owned wide-choice data

Local CPU data work only. No model inference, teacher text, training launch or GPU spend.
The only N2 experiment change is replacement of approximately 10% of A3 training choice
question slots by owned wide-choice questions. Optimizer, adapter, seed, sequence length,
question presentation count, source exclusions and original calibration remain A3's.
A3 is a private diagnostic recipe with its existing exceptions; this work does not authorize shipping.

## Owned data and limitations

`data/item43_wide_choice.py` defines three original 16-by-16 Cartesian taxonomies:
product material/object, routing color/destination, and intent action/object. Rules generate
requests and complete menus, without imported taxonomies or external text. A match needs
both attributes. One-attribute neighbors are hard distractors; at least 29 survive every
menu. Half the requests have their exact category omitted and must select `none`, rendered
as `none: None of these`. This is closed-menu rejection, including categories that exist in
the universe but are absent from the offered menu. It does not establish performance on
natural paraphrases or genuinely novel semantic categories. These simple tasks isolate
full-option competition, rather than claiming to reconstruct any benchmark distribution.

The finite source contains 6,144 questions: 5,760 train and 384 diagnostic. Each size
32/64/128/255 has 1,440 train and 96 diagnostic questions. Domains and match/no-match are
balanced in the complete generated splits. A whole first-attribute family per domain is
held out, with every menu size and match/no-match sibling in the same split. The held-out
attribute can still appear as a distractor in training; this holds out request families,
not vocabulary. Training
replacement selection is deterministic and balanced by size; domain and answer counts
in the selected subset should be read from the composition receipt rather than assumed.
No Decision Index dataset or descendant supplies data. The inherited item33 ancestry
exclusions and Apache-2.0 license allowlist gate the source manifest; the entire finite
source, including diagnostic rows, must pass the item25 full-suite scanner before composition.
An overlap rejects the whole owned source. No partial salvage or benchmark text in Git.

## Renderer and scoring contract

No renderer or production serving change is needed. The existing `train/jebadiah_prompt.py`
Renderer preserves AINode's label alphabet through its ordinary single-token range, then
uses its existing extended alphabet for wider menus. For the pinned Qwen tokenizer there
are 588 available extended single-token labels. Sizes 32 and 64 use `ainode`; 128 and 255
use `extended`. This is Jeb's existing independent implementation, not imported Nimble code.
Candidate IDs are checked distinct, ordinary and single-token at the answer boundary.

N2 training uses the existing `DecideCollator` and candidate cross-entropy: shuffle the
complete menu, map the gold wire key to its displayed label token, gather all 32/64/128/255
candidate logits at the last real prompt position, mask padded slots with negative infinity,
and softmax jointly over all real options. `option_logits` computes the candidate head in
fp32. There is no top-20 read, shortlist, menu chunking, generation or independent binary
scoring. CPU tests exercise the actual candidate head and `Scorer.score_rendered` with 255
candidates, a mixed 32/255 batch, masking, the final candidate's prediction and gradient.
They use a tiny fixed numeric head, not a downloaded language model. A separate live
CPU check uses the real training `DecideCollator` with the pinned tokenizer, shuffled
32/255 menus, padding and gold-key-to-label mapping. All eight N2 tests and 24 item32/item33
pattern regression tests passed on Atlas.

Future proxy and diagnostic scoring must use the existing local `Scorer` through
`train/eval_jebadiah.py` for the owned diagnostic. It gathers all allowed label logits,
divides by the unchanged A3 temperature per type, softmaxes across the full menu and maps
argmax back to the original wire key. The N2 export must carry the original A3 temperature artifact so the local evaluator
reads that same dictionary. Temperature is unchanged, not refit on the 384 owned
diagnostic questions. Diagnostic metrics are exact-match accuracy by domain, option count
and match/no-match, plus no-match precision and recall, on that separate holdout.
The frozen proxy remains the established acceptance test, using its pinned Jebadiah engine
and paired group bootstrap. Keep only when overall and Retrieval paired 95% intervals
vs A3 are both above zero and score exceeds 44.67; otherwise drop.

AINode PR 301, commit `bfb90a8185f75a6201207220b683960a46ce1312`, was inspected in the
local AINode checkout. Its `/v1/decide` normalizer rejects more than 20 options with a
`DecideError`, returned as HTTP 400 `invalid_request_error`; `/v1/systemone` rejects with
422. The engine has `top_logprobs=20`. That HTTP path cannot score N2 and must not be used
for this experiment. This was a source check, not a live HTTP or language-model check.

All 6,144 rows were rendered in canonical, reversed and seeded-shuffled menu order:
18,432 renders, zero truncation. Maximum prompt tokens at sizes 32/64/128/255 were
268/428/749/1,384. Budget is 1,984 plus a 64-token padding reserve, under A3's 2,048 maximum.
Pinned tokenizer, prompt and chat-template hashes are in the build summary.

## Commands and scan gate

Use a local environment with transformers, jinja2, xxhash and CPU torch for tests. Downloads
are tokenizer files and pinned A3 data only, no weights. On Atlas the tested commands were:

```sh
export TMPDIR=/data/orca/scratch/item43
export HF_HOME=/data/orca/toolchains/hf
~/.local/bin/hf download Qwen/Qwen3.5-9B tokenizer.json tokenizer_config.json chat_template.jinja merges.txt vocab.json --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir "$TMPDIR/tokenizer"
~/.local/bin/hf download frontier-infra/jebadiah-data-v2-1-item32 --repo-type dataset --revision bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed --include 'a3/*' --local-dir "$TMPDIR/a3-download"
ITEM43_TOKENIZER="$TMPDIR/tokenizer" python3 -m unittest discover -s data -p 'test_item43_wide_choice.py' -v
python3 data/item43_wide_choice.py --out "$TMPDIR/wide" --tokenizer "$TMPDIR/tokenizer"
```

Without `--index` the build has `pending_studio_scan`, and the composition builder refuses it.
The committed build summary is an Atlas render and composition preview receipt, **not an
overlap clearance**. Final composition and training remain gated on the Studio scan.

The lead runs the following exact scan command on Studio from this checkout, with the same
frozen tokenizer prepared at `/Volumes/PRO-G40/scratch/item43/tokenizer`. The output must
be empty. The program verifies both this exact protected path and its pinned SHA256 before
loading it, scans all 6,144 candidates, and temporarily symlinks the index without copying it.
The symlink is removed even on scanner failure. No benchmark payload is exported.

```sh
export TMPDIR=/Volumes/PRO-G40/scratch/item43
python3 data/item43_wide_choice.py --out /Volumes/PRO-G40/scratch/item43/wide-scanned --tokenizer /Volumes/PRO-G40/scratch/item43/tokenizer --index /Volumes/PRO-G40/caches/jeb/protected-0.3-cf54ade9.pkl
python3 data/item43_n2_mix.py --base /Volumes/PRO-G40/scratch/item43/a3-download/a3 --wide /Volumes/PRO-G40/scratch/item43/wide-scanned --out /Volumes/PRO-G40/scratch/item43/n2
```

The pinned A3 has 11,708 choice slots. Integer replacement is floor(11,708 / 10) = 1,170,
9.993167%: 293/293/292/292 at the four sizes. Total question presentations stay 21,190.
`replacement-slots.json` maps each old question ID to its new owned row. Mixed-question rows
keep their original state and all remaining questions/labels/targets; only the replaced
question becomes an independent owned-state row. Selected domains are product 384, intent 404 and routing 382; labels are 594 match and
576 no-match. Original calibration is copied byte for
byte. The separate `wide-diagnostic.jsonl` is not added to training or calibration.
The builder pins A3 input hashes, checks owned hashes, rejects cross-source and train/holdout
ID/family/state collisions, and fails closed until the Studio scan has passed.

No training command is authorized or launched by this PR. Before a future run, the lead
must finish the scan/composition, validate the manifest, then use the unchanged A3 recipe
and scoring contract above. Delete item43 scratch after exporting the required nonprivate
receipts; never copy the private protected index from Studio.
