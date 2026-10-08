# Item 32: private v2.1 regression ablations

These diagnostic checkpoints are private and never submitted or promoted. The
frozen edition 0.3 proxy has ID-list SHA256
`74d8162296f404a676624d87bf98abb920d46acff3c38be6a12504131bc43405`.
The comparison baseline is 9B v2, proxy 44.67. The r1 proxy scored 38.12 with
paired difference -6.5528 and 95% interval [-8.2586, -4.5931].

## Controlled recipes

Both runs start from `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`
and use the archived `9b-chat-v1` recipe: rank 16, alpha 32, 2,048 tokens,
learning rate 0.0001, cosine schedule, warmup 30, one full epoch, batch 8,
accumulation 1, seed 17, dropout 0.05, all-linear targets, ordinal score targets
with adjacent mass 0.2, and held-out evaluation every 200 steps on up to 358
questions. Both enable item 27 grouping, padding to 64 and checkpoint threshold
512. Backbone autocast stays disabled. Grouping changes sample order relative
to archived v2; it is shared between A1 and A3 and is a residual baseline
comparison confounder.

A1 preserves the exact archived legacy training/calibration split and removes
BoolQ, DBpedia14, MNLI and all SummEval dimensions. The source-only removal
leaves 10,632 training questions and 517 calibration questions. Preflight
full-suite overlap scanning additionally flagged 42 retained HelpSteer2 records
in 22 families. The lead explicitly authorized record-only removal for these
private diagnostics, leaving **10,595 training questions and 512 calibration
questions**. Any shipping candidate requires whole-family exclusion. The
private ID-only overlap ledger records family, record count, benchmark run ID,
and hashed span example ID, without text.

A3 adds 10,595 questions from new upstream sources, for **21,190 questions**
and exactly 50% new data. Selection uses a deterministic, label-blind,
whole-record round robin across upstream datasets. Rubric/config variants
share one cap. Each new upstream dataset is limited to 10% of total questions.
Variants of an existing upstream dataset, including HelpSteer2, are not new
sources. New rows sharing a normalized state or family with legacy training or
calibration are omitted. Both runs use the identical A1 calibration file.

Every selected subset must appear in the pinned v2.1 license manifest, with
approval and license fields, and all ancestry must pass its entire protected
source exclusion list. Inputs are verified against archived and frozen SHA256
hashes before construction. No private records or labels are committed.

## Reproduction

Build private mixes using the final full-suite overlap scanner hit ledger:

```bash
python data/item32_ablation_mix.py \
  --legacy /path/to/archived/data-v1 \
  --v21 /path/to/pinned/v21-release \
  --overlap-hits /path/to/private/contamination-hits.json \
  --tokenizer /path/to/pinned/Qwen3.5-9B-tokenizer \
  --output /tmp/item32-final-mixes
```

Upload those directories to private dataset
`frontier-infra/jebadiah-data-v2-1-item32`, then run both commands without waiting
between submissions. `HF_TOKEN` must already be supplied from Bitwarden.

```bash
python scripts/item32_launch.py a1 DATASET_COMMIT_SHA
python scripts/item32_launch.py a3 DATASET_COMMIT_SHA
```

The committed JSON configs specify the same recipe and private destinations.
The first training allocations used hard timeouts of 100 and 150 minutes. A3
failed before step 1 because an options-only prompt exceeded 2,048 tokens. The
builder now checks every eligible new record against the pinned tokenizer,
reserving 64 tokens for choice-order variation, and records per-source exclusions
for prompts that cannot fit. Long state is truncated under the unchanged v2
renderer. The A3 retry uses 135 minutes. Two proxy jobs reserve
35 minutes each. At $2.75/hour, the combined maximum allocation is $14.6667,
under the $15 approval. The failed A3 job ran 652 seconds, estimated $0.4981;
the reduced retry allocation gives a revised combined ceiling of $14.4773.
Failed or retried allocations count against this same cap. Private spend evidence records job IDs, durations and estimated charges.

## Interpretation

Compare A1 with v2 to estimate the scrub cost, including the disclosed 37-question
additional training removal and batching-order confounder. Compare A3 with A1
to estimate adding new upstream sources at bounded dilution under the same
recipe. A3 versus r1 also changes rank, length, question-type mix, data volume,
source balance and epoch completion. These two runs cannot uniquely attribute
any r1 residual to rank, truncation or partial-epoch exposure.

The final report must include paired group-bootstrap intervals versus v2,
five area deltas, BANKING77, When2Call, RAGTruth, API-Bank and CLINC150+OOS,
coverage/failures, immutable artifacts, actual epoch completion and spend.
No full-suite score is inferred from this proxy.
