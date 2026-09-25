# 03. The v2 synthetic pool

**Question.** Does a large pool of synthetic choice and yes/no questions, authored and labelled by our own teacher
models, lift the model by bringing the score share down to about a quarter?

**Pool.** `data/pool-v2/`: DeepSeek V4 Flash wrote 3,253 structured states across 24 families (support, routing,
code review, incident alerts, invoices, scheduling and others; see `families.py`) with 3 to 5 questions each.
Every question was then answered by two teachers (DeepSeek V4 Flash and Qwen3.8 27B) through AINode's `/v1/decide`
with the exact prompt Jebadiah trains on, and the gold is the mean of their distributions. Questions where the
teachers disagreed and neither was sure, or whose mean top probability was under 0.4, were dropped; states that leaked
the answer were rejected by a linter before labelling. Result: 14,714 questions (7,858 choice, 6,856 noul), published
as the dataset `frontier-infra/jebadiah-synth-v2` (card: `data/pool-v2/DATASET-CARD.md`). Joined with the v1 pool it
is 30,527 questions at 23.2 percent score. The highest 5-gram containment of any kept state in the 20 reported test
sets is 0.077.

| run | pool | headline | Jevals PubMedQA | Nimble 324 | human yes/no pooled 6 | DSj HelpSteer2 |
|---|---|---:|---:|---:|---:|---:|
| `9b-v1` | data-v1 | 73.29 | 89.7 | 78.7 | 84.6 | 11.9 |
| `9b-v2` | data-v2 (all synthetic) | 72.21 | 87.0 | 74.7 | 84.0 | 1.5 |
| `9b-v2b` | data-v2b (families with teacher agreement >= 0.82, synthetic noul capped at 3,000) | 71.57 | 86.3 | 75.0 | 83.0 | 6.0 |

**What we learned.** More synthetic data made the 9B worse, not better. Choice sets such as Banking77 went up a
little (70.0 to 72.0), but the human yes/no sets, Nimble 324 and the score calibration all went down. Filtering by
teacher agreement did not help. This is why v2 was not published from this pool, and it set up the next three
experiments: how much synthetic choice helps on its own (04), whether better teacher labels help (05, 06), and
whether human yes/no data should replace synthetic yes/no (07).

**Not reproducible from this repository.** The `data-v2b` filter was applied inline during the sweep and its builder
was not kept. `data/manifests/pools.json` records the sha256 of its train and calib files.

**Records.** `results/runs/{9b-v2,9b-v2b}/`, `results/nonce/{9b-v2,9b-v2b}/`, the generator's own manifest
`data/manifests/pool-v2-data-v2.json`.
