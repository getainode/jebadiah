# ITEM 56: outside-data probe B results

**Drop probe B; retain A3.** Probe seed 17 scores 46.6834, seed 18 scores 46.2156, and their mean is 46.4495 versus the two-seed A3 mean 46.8194. The paired whole-group difference is -0.3699, with 95 percent interval [-1.4626, +0.6483]. The mean and interval fail the predeclared win rule. Mean CLINC macro-F1 improves 0.7469 points, so the CLINC guard passes. This losing probe does not authorize splitting into single rungs or model promotion.

Jason chose option B on Desk #646 on 2026-10-10: preserve all A3, add all four scanned outside sources, train seeds 17 and 18, and compare with the two-seed A3 baseline. The decision rule was committed in docs/run-rules.md before launch: a higher unrounded mean, paired bootstrap interval entirely above zero, and mean CLINC loss no greater than 2.5 points. Neither matched seed loses more than 5 CLINC points. The merge gate remains maximum probability shift 0.05 and zero confident flips.

All four scored arms cover the identical frozen 11,079 run IDs, with every raw payload hash matching, all rows successful and zero whole-case failures. The unchanged official scorer is recomputed on each of 2,000 identical paired complete-group draws within benchmark/domain/track strata, averaging each arm’s two seed scores on every draw. The interval is conditional on the frozen proxy and does not estimate training-seed variance. Original A3 and its seed-17 repeat have identical served weights, temperatures and answers, as proved in item52.

## Frozen proxy scores

| Model | Headline | Knowledge | Language | Retrieval | Tools | Arts |
|---|---:|---:|---:|---:|---:|---:|
| a3-s17 | 46.97 | 31.64 | 50.32 | 50.29 | 67.73 | 33.40 |
| a3-s18 | 46.66 | 30.97 | 53.23 | 50.84 | 62.29 | 33.33 |
| probe-s17 | 46.68 | 32.59 | 52.42 | 51.06 | 61.53 | 32.39 |
| probe-s18 | 46.22 | 30.20 | 53.41 | 50.71 | 61.57 | 31.95 |

| Area | Mean difference | Paired 95% interval |
|---|---:|---:|
| knowledge | +0.09 | [-2.71, +2.80] |
| language | +1.14 | [-0.63, +3.00] |
| retrieval | +0.32 | [-1.29, +2.01] |
| tools | -3.46 | [-6.52, -1.06] |
| arts | -1.19 | [-3.25, +0.87] |

Matched headline differences are -0.29 for seed 17 and -0.45 for seed 18. Mean Tools difference is -3.46, within the approximately 6-point operational seed-noise allowance. Matched Tools differences are -6.20 and -0.71 points. The proxy-conditional Tools interval is below zero, but two seeds and the joint exposure change do not establish source attribution.

## Requested benchmarks

Raw benchmark scores are percentages; macro-F1 is identified explicitly. These are descriptive point scores, with complete coverage for every arm.

| Benchmark | Metric | A3 s17 | A3 s18 | Probe s17 | Probe s18 | Mean difference |
|---|---|---:|---:|---:|---:|---:|
| CLINC150 | macro-F1 | 76.82 | 74.35 | 77.41 | 75.26 | +0.75 |
| CLadder | accuracy | 65.80 | 64.60 | 65.80 | 63.60 | -0.50 |
| ANLI | macro-F1 | 60.74 | 61.72 | 62.41 | 61.35 | +0.65 |
| GSM8K | accuracy | 48.86 | 56.44 | 46.59 | 40.91 | -8.90 |
| HoVer | accuracy | 67.58 | 68.58 | 68.08 | 68.08 | +0.00 |
| BBH | accuracy | 67.21 | 69.02 | 71.20 | 68.30 | +1.63 |

Matched CLINC macro-F1 improvements are 0.5926 and 0.9013 points. Neither single-seed loss flag is set. Math’s own diagnostic gain does not transfer to GSM8K, whose mean raw accuracy falls 8.90 points.

## Independent source diagnostics

All 16 model/source passes contain 300 questions each, one repeat, full menus and zero truncations. No diagnostic question is trained or used for calibration. Each model uses temperatures fitted by the original A3 command on unchanged calibration bytes. The item48 timing-only wrapper preserves each completed pass and avoids the tuple-keyed internal question map. All strata, counts and verification metrics are retained in results/runs/item56/diagnostic-summary.json; raw predictions remain private.

| Source | Primary metric | A3 s17 | A3 s18 | Probe s17 | Probe s18 | Mean difference |
|---|---|---:|---:|---:|---:|---:|
| Corr2Cause | balanced accuracy | 51.36 | 53.26 | 59.61 | 73.40 | +14.19 |
| SpaceNLI | macro label accuracy | 96.33 | 97.67 | 99.00 | 91.33 | -1.83 |
| MASSIVE | choice macro intent accuracy | 84.44 | 85.28 | 87.22 | 87.78 | +2.64 |
| DeepMind math | choice macro module accuracy | 49.33 | 52.00 | 78.00 | 86.00 | +31.33 |

The causal, intent and arithmetic diagnostics improve in the two-seed mean, while SpaceNLI declines. These diagnostic gains do not override the failed frozen-proxy rule. This single combined probe changes both source composition and exposure, so it does not isolate any source’s effect.

## Exact additive composition

Private dataset frontier-infra/jebadiah-data-v2-1-item47, folder probe-b/, is pinned at 1de4a087bf4aa8e7ffd2de99f0abefea7a7b2c80. All uploaded file hashes were independently downloaded and reverified. The final train has 24,939 presentations: all 21,190 original A3 presentations as a byte-identical prefix plus 3,749 exact scanned additions. Calibration retains its original 512 questions byte for byte.

| Source | Scanned added questions | Existing A3 exposure | Combined exposure |
|---|---:|---:|---:|
| Corr2Cause | 1,000 | 102 | 1,102 |
| SpaceNLI | 1,000 | 102 | 1,102 |
| HWU/SLURP/MASSIVE together | 749 | 102 | 851 |
| DeepMind math | 1,000 | 0 | 1,000 |

Every source stays under the fixed original-A3 cap of 2,119 questions. HWU, HWU64, SLURP, MASSIVE and translations share one cap. Adding every source increases total exposure and changes A3’s original 50 percent aggregate new-source mixture; nothing was removed to conceal that change. Every supplied NOTICE and license is kept, including SpaceNLI’s exact upstream MIT license restored from the pinned source manifest. Existing private A3 contamination exceptions remain a shipping blocker.

Source pins and all file hashes are in data/manifests/item56-input-pins.json. Corr2Cause scanned rung2 is 2242e85fae6f8aff3fc3cea3fa95869d99132522, its exact rung2b addition is c9fcf494319527d12888e0f5854980c11a51b91d, SpaceNLI/MASSIVE additive sources are da7fbc3260d42103ed44f557b198bfe05af61976, and scanned math is 764c465df998127df8801ddfc4bda49daa606721. Math’s zero-hit Studio receipt binds all 24,000 records to the source-universe and scanner hashes. The live test independently replayed item51’s selection from scanned rung2 and proved exact rung2b byte equality.

## Training, gates and cost

Both seeds use original A3 runtime 86a8203d792463536eb009a4d8da6c8681ffdd99 and Qwen/Qwen3.5-9B parent c202236235762e1c871ad0ccb60c8ee5ba337b9a. Rank 16, alpha 32, dropout 0.05, length 2048, LR 1e-4, one epoch, effective batch eight, ordinal targets, item27 speed flags and autocast off remain unchanged. The guarded seed-18 patch changes exactly one byte in the original seed literal. Live checkpoint configurations match every A3 recipe field; differences are train identity and routing, plus seed 18 for that run. Temperature fitting uses the original A3 command on unchanged calibration, rather than item51’s static-temperature override.

Both completed exactly 3,118 optimizer steps for one epoch, with 24,939 train and 512 calibration presentations. Both report 36 inherited truncated presentations under the unchanged length limit; diagnostics have none. Seed 17’s gate shift is 0.0110408 with 259/260 identical picks, while seed 18’s is 0.0126655 with 260/260 identical picks; both have zero confident flips. Live environments match: Python 3.12.11, torch 2.11.0+cu128, CUDA 12.8, driver 580.178.04 and kernel 6.12.103. Both final package freezes exactly match their respective item52 A3 control freezes and the original locked packages. Proxy Python 3.11.13, torch 2.8.0+cu128, transformers 5.17.0, inference code hashes and runner hash also match both controls. Observed host differences and comparisons are recorded in results/runs/item56/environment-comparison.json; the historical original A3 full freeze and immutable image digest were not retained, as documented in item52. The full provenance, temperature values, notices, environment freezes and served weight SHA256s are recorded under results/runs/item56/s17 and s18.

| Job | ID | Status | Running seconds | Estimated USD |
|---|---|---|---:|---:|
| Train/export s17 | [6ac9fd16095c5780893107cb](https://huggingface.co/jobs/jbrashear/6ac9fd16095c5780893107cb) | COMPLETED | 3484 | 2.6614 |
| Train/export s18 | [6ac9fd16fee2c90070185309](https://huggingface.co/jobs/jbrashear/6ac9fd16fee2c90070185309) | COMPLETED | 3485 | 2.6622 |
| proxy_17 | [6aca0b00095c578089310efb](https://huggingface.co/jobs/jbrashear/6aca0b00095c578089310efb) | COMPLETED | 1128 | 0.8617 |
| proxy_18 | [6aca0b00095c578089310efc](https://huggingface.co/jobs/jbrashear/6aca0b00095c578089310efc) | COMPLETED | 1129 | 0.8624 |
| diagnostic | [6aca0b00fee2c90070185c38](https://huggingface.co/jobs/jbrashear/6aca0b00fee2c90070185c38) | COMPLETED | 768 | 0.5867 |

**Total estimated running-time compute: $7.6343, below the $20 cap.** All jobs used RTX PRO 6000 at $2.75/hour, without retries or recovery. Reserved timeouts were 150 minutes per training job, 35 per proxy and 45 for diagnostics, totaling $19.0208. The authenticated Atlas CLI was used because the connector hardware enum rejected RTX PRO 6000 before creating any jobs.

Private probe models are frontier-infra/jebadiah-9b-v2-1-probe-b-s17 at e1cdfee0d3659494b6451bf2fb3f08007f77a552 and frontier-infra/jebadiah-9b-v2-1-probe-b-s18 at cda11af970b00b1c39b7af02b1b6c19ff4340370. Training code was committed before launch at a0960e5c90ec6d24d376550628515114a70c9d95, and evaluation code at 9bd4979acbdef044d33ce3df4607239a73c63b8b. Frozen proxy runner is 587f4f496b9cf1d83f0bc8e495379496f65afa2f and suite is e57106b5e0698e74bd1a88b3b4c19b94a0dc8328.

Private paired analysis is persisted under jbrashear/jebadiah-9b-v2-1-index-results at 320506197e65a3efeb550d5e91e18de5487a715a, folder analysis/item56-probe-b; the upload receipt records all hashes. Raw inference is private in the immutable result revisions recorded in results/runs/item56/job-record.json. Local scoring summaries and all 2,000 bootstrap draws are under results/runs/item56/analysis.

Six builder tests and two recipe/allocation tests passed on Atlas, including real pinned inputs, exact scanned additions, unchanged calibration, missing-math rejection, tampering rejection, failed-scan rejection and allocation limits. An identity analysis using the same two baseline inputs in both arms returned exactly zero difference and interval. No GitHub-hosted CI was used. PR 32 is for lead review and merge only; models remain private and this result authorizes no promotion.

Scratch cleanup is recorded in results/runs/item56/cleanup.json. No merged model was downloaded to Atlas.
