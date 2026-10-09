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

## Remaining

Training, live memory measurement, unchanged merge verification, candidate
proxy, 2,000 paired bootstrap draws, five area deltas, final spend, durable
private evidence, scratch cleanup, and lead review remain. Local checks so far:
five proxy tests, config equivalence, compilation and whitespace checks.
No GitHub-hosted CI was launched. PR: https://github.com/getainode/jebadiah/pull/20.
