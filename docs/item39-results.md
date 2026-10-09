# Item 39: private 27B A3 transfer

This run transfers the accepted A3 mix and recipe to the exact chat parent used
by published Jebadiah 27B. It is a frozen proxy experiment, not a leaderboard
submission. The candidate and checkpoint repositories remain private.

Prelaunch keep/drop row was committed in `3cfa6a8` before job submission.
Retention requires a positive paired 95% interval versus published 27B and a
passing unchanged merge gate: maximum probability shift at most 0.05 and zero
confident flips. A3's inherited private diagnostic source-policy exception
continues to block shipping even if the statistical gate passes.

## Pinned recipe

Parent: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
confirmed from the published 27B model card and the existing pipeline pin.
Data: `frontier-infra/jebadiah-data-v2-1-item32@bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed`,
using unchanged `a3/train.jsonl`, `a3/calib.jsonl` and `a3/manifest.json`.
The 21,190-question mix and identical 512-question calibration split are those
validated in PR 14. No new mixture was constructed.

Rank 16, alpha 32, length 2048, LR 1e-4, cosine schedule, warmup 30, one epoch,
ordinal score targets with adjacent mass 0.2, dropout 0.05, all-linear LoRA,
seed 17, grouped lengths, padding to 64, checkpoint threshold 512, and backbone
autocast disabled. The fp32 candidate head is unchanged.

Batch 1 and accumulation 8 preserve effective batch 8 on the 96 GB device.
The bf16 checkpoint is 55.56 GB; the published adapter has 116.7M trainable
parameters, about 1.87 GB for fp32 master parameters, gradients and Adam states.
Item27's measured 9B batch-eight peak was 66.56 GB at rank 64. Simply replacing
its 19.3 GB checkpoint with this 55.56 GB checkpoint exceeds 96 GB before
accounting for larger activations, so batch eight is not a defensible fit.
Batch one is the smallest permitted per-device batch; live peak memory will be
reported after training. This is a capacity accommodation, not a measured
microbatch quality equivalence claim.

## Published baseline

The unchanged official edition 0.3 scorer gives **55.08** on the frozen
11,079-ID manifest, with 11,079 ok results, zero missing IDs, zero failures and
10,380 complete groups.

| Area | Published 27B |
|---|---:|
| Knowledge & Reasoning | 42.28 |
| Language Understanding | 58.55 |
| Retrieval & Classification | 55.21 |
| Tools & Automation | 76.71 |
| Arts & Human Taste | 39.36 |

The baseline is extracted from the existing complete edition 0.3 results in
`jbrashear/jebadiah-27b-index-results@45c7cba5a2d90042b3e563c5e42470d353725818`,
path `runs/jebadiah-27b-1c0d794f-v03/results.jsonl.gz`.
Its inference revision is `1c0d794fc24b24b15175c6a912654fa5cd76e1bb`;
the current published revision is `dfa369766fe41a093a4807081cf3057800809abb`.
All model weights, configuration, tokenizers, prompt contract, temperatures and
inference scripts have matching Hub blob or LFS hashes. Only README.md and
eval/RESULTS.md differ. Existing results therefore avoid duplicate inference
without changing the published baseline. Extracting and scoring the frozen
subset incurs zero new GPU spend.

## Jobs and budget

Training job: `jbrashear/6ac90847fee2c90070179a42`, code revision `3cfa6a8`.
Hardware: HF Jobs RTX PRO 6000. Training cap 420 minutes; each potential proxy
cap 80 minutes. At $2.75/hour, the initial allocation ceiling with two proxies
is $26.5833, leaving $8.4167 within the $35 cap for failed/recovery allocations.
One baseline proxy allocation is unnecessary after existing coverage validation.

Measured 9B A3 total runtime was 3,021 seconds. Parameter-count scaling to 27B
estimates 9,063 seconds, about 2.5 hours, before the batch-one overhead and a
longer merge/proxy. The initial honest estimate is 3 to 5 hours and will be
replaced by live measured throughput. No full-suite inference is authorized.

## Completed paired results

| Model | Proxy | Delta versus published | Paired 95% interval |
|---|---:|---:|---|
| Published 27B | 55.08 | baseline | baseline |
| Private 27B A3 | 56.62 | +1.54 | [+0.38, +2.84] |

Both scores cover all 11,079 requests with zero failures and matching frozen
payload hashes. The 2,000-draw paired complete-group bootstrap uses seed
20261008, the unchanged official scorer, and the frozen manifest SHA256
`74d8162296f404a676624d87bf98abb920d46acff3c38be6a12504131bc43405`.
Deltas and intervals use unrounded scores. The positive aggregate interval
passes the predeclared statistical gate, so retain this private diagnostic.

| Area | Published | A3 | Delta | Paired 95% delta interval |
|---|---:|---:|---:|---|
| Knowledge & Reasoning | 42.28 | 40.81 | -1.47 | [-4.48, +2.25] |
| Language Understanding | 58.55 | 64.21 | +5.66 | [+3.06, +8.41] |
| Retrieval & Classification | 55.21 | 57.60 | +2.39 | [+0.37, +4.17] |
| Tools & Automation | 76.71 | 75.03 | -1.68 | [-2.57, -0.83] |
| Arts & Human Taste | 39.36 | 42.23 | +2.87 | [+0.25, +5.85] |

Language contributes the largest gain. Tools regresses beyond the paired
noise despite the aggregate improvement. Knowledge also declines, with an
interval spanning zero. The contrast tests the complete accepted A3 mix and
recipe transfer, including its additional full-epoch exposure, against the
published 27B; it cannot isolate source content from mixture balance or
optimizer exposure. Microbatch one changes grouping windows and dropout draw
partitioning while preserving effective batch eight, which is an additional
capacity-driven comparison limitation. Intervals are conditional on this
frozen proxy, not full-suite or training-seed uncertainty.

## Live training and export proof

One full epoch completed: 21,190 questions, 2,649 optimizer steps, epoch 1.0,
10,731.9 training seconds. Peak allocated GPU memory was 71.44 GB with batch
one and accumulation eight. All 116,727,808 trainable parameters and realized
configuration fields match the requested recipe and exact chat parent.

The original merge gate passed: 496 merged LoRA pairs, 260/260 identical picks,
zero confident flips, maximum probability shift 0.0084695816, unchanged limit
0.05. Temperatures are choice 1.0608, noul 1.0405, score 0.6524. All 18 merged
shards total 55,563,006,776 bytes; tokenizer, prompt contract, temperatures and
inference scripts are present and pinned. Candidate model revision:
`2be961761bde8e1c641cd1d7c75afd60cc37662d`.
Final checkpoint/evidence revision:
`4a8665a12c7539edcd520a5ee3bf156c036904e2`.
Both repositories are private and were never submitted.

## Final spend and validation

| Job | Stage | Running seconds | Estimated USD |
|---|---|---:|---:|
| Training 6ac90847fee2c90070179a42 | COMPLETED | 11,797 | 9.0116 |
| Proxy 6ac936b4fee2c9007017bd1b | COMPLETED | 2,585 | 1.9747 |
| Existing published proxy subset | reused | 0 new | 0 |

Estimated total **$10.98625**, below the $35 cap, using provider running
seconds at $2.75/hour. Final billing can differ. No failed or capacity-retry
jobs were needed. GPU inference completed in 2,360 seconds; the initial
3 to 5 hour estimate was consistent with measured completion. Both jobs
settled successfully, with no remaining cloud work.

Local checks: five frozen-proxy tests, four A3 source-policy tests, realized
config and final epoch assertions, all 11,079 payload identity checks for each
model, unchanged scoring and 2,000 paired draws, candidate shard/payload
verification, compilation and whitespace checks. No GitHub-hosted CI ran.

Private immutable candidate results:
`jbrashear/jebadiah-9b-v2-1-index-results@1bc289de2332c25610ed4e1a548350f8873a7413`,
path `runs/item39-27b-a3-2be96176-proxy-0.3-10pct/results.jsonl.gz`.
Detailed score, bootstrap, hash equivalence, training summary, memory samples,
merge evidence and spend are stored under `analysis/item39-27b-a3` in that
private dataset. All item39 PRO-G40 scratch downloads, suite, extracted kit,
virtual environment, caches and generated analysis are deleted after durable
upload; no model weights were downloaded to the Mac and no local merged model
was created. Review and merge remain with the lead at
https://github.com/getainode/jebadiah/pull/20.
