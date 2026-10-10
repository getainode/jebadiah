# ITEM 50: drop Corr2Cause rung 2 against A3

Rung 2 scored **45.95** against A3 **46.97** on the frozen 10% proxy. The difference is **-1.02**, with paired whole-group bootstrap 95% interval **[-2.49, +0.28]**. All 11,079 requests succeeded. The interval crosses zero and fails the predeclared keep rule. **Drop this rung; retain A3.** The score exceeds the 44.67 floor and the separate causal balanced accuracy improves, but the overall rule still fails.

The unchanged official kit recomputed 2,000 paired complete-group draws, stratified by benchmark/domain/track, with seed 20261008 and eight Studio CPU workers. Both result files match every frozen run ID and payload hash. The interval is conditional on this proxy, rather than full-suite or training-seed uncertainty. Run-ID SHA256 is `74d8162296f404a676624d87bf98abb920d46acff3c38be6a12504131bc43405`.

Five areas use chance-adjusted skill points. Their paired intervals are descriptive; the overall interval determines keep/drop.

| Area | A3 | Rung 2 | Difference | Paired 95% interval |
|---|---:|---:|---:|---|
| Knowledge | 31.64 | 30.49 | -1.14 | [-4.45, +2.07] |
| Language | 50.32 | 51.25 | +0.93 | [-1.48, +3.37] |
| Retrieval | 50.29 | 50.52 | +0.23 | [-2.00, +2.45] |
| Tools | 67.73 | 61.71 | -6.02 | [-10.66, -3.12] |
| Arts | 33.40 | 34.28 | +0.88 | [-1.75, +3.45] |

Requested benchmark results follow. Native values are percentages; ANLI uses macro-F1 and the other rows use accuracy. Skill deltas are the official chance-adjusted points.

| Benchmark | Requests | A3 native | Rung 2 native | Skill difference |
|---|---:|---:|---:|---:|
| CLadder | 500 | 65.80 | 64.40 | -2.80 |
| HoVer claim verification | 401 | 67.58 | 67.83 | +0.50 |
| ANLI | 320 | 60.74 | 62.34 | +2.40 |
| GSM8K | 264 | 48.86 | 50.38 | +1.77 |
| BBH fixed-option tasks | 552 | 67.21 | 69.75 | +3.68 |

Tools had the largest area loss. When2Call native accuracy fell from 68.94% to 56.40%, a loss of 16.72 skill points over 367 requests. Home appliance exact accuracy fell by one of nine cases, from 33.33% to 22.22%. BFCL lost 2.38 skill points, ToolRet lost 1.80, and API-Bank was unchanged.

The independent causal diagnostic contains 300 original-train questions across 12 graph families excluded from training and calibration. Only 29 labels are necessarily-valid, so **overall balanced accuracy was declared as the primary diagnostic metric before launch**. The same full-menu scorer, one repeat, maximum length 2,048 and pinned A3 temperatures evaluated both immutable models. Both runs scored all 300 questions with zero truncations.

| Causal diagnostic metric | A3 | Rung 2 | Difference in percentage points |
|---|---:|---:|---:|
| Accuracy | 90.00% | 89.67% | -0.33 |
| Balanced accuracy | 51.36% | 80.42% | +29.07 |
| Valid-class precision | 33.33% | 47.62% | +14.29 |
| Valid-class recall | 3.45% | 68.97% | +65.52 |

A3 had TP/FP/FN/TN = 1/2/28/269 and 270 correct answers; rung 2 had 20/22/9/249 and 269 correct answers. The majority-label accuracy floor is 271/300, or 90.33%. The valid-class gain is substantial while ordinary accuracy falls by one answer. CLadder also fell by 1.40 native percentage points.

Counts and accuracy by question type, original relation template and variable count are below. Undefined balanced accuracy in a stratum with only one gold class is shown as N/A. Graph-family metrics and all predictions remain in the private diagnostic artifact.

| Stratum | Questions | Valid labels | A3 accuracy | Rung 2 accuracy | A3 balanced | Rung 2 balanced |
|---|---:|---:|---:|---:|---:|---:|
| template:child | 53 | 0 | 100.00% | 100.00% | N/A | N/A |
| template:has_collider | 58 | 11 | 82.76% | 77.59% | 54.55% | 72.24% |
| template:has_confounder | 55 | 4 | 92.73% | 92.73% | 50.00% | 61.52% |
| template:non-child descendant | 50 | 2 | 96.00% | 96.00% | 50.00% | 50.00% |
| template:non-parent ancestor | 44 | 0 | 100.00% | 100.00% | N/A | N/A |
| template:parent | 40 | 12 | 65.00% | 70.00% | 46.43% | 78.57% |
| type:choice | 150 | 10 | 93.33% | 90.67% | 50.00% | 76.43% |
| type:noul | 150 | 19 | 86.67% | 88.67% | 51.87% | 82.26% |
| variables:5 | 124 | 12 | 88.71% | 88.71% | 49.11% | 82.59% |
| variables:6 | 176 | 17 | 90.91% | 90.34% | 52.94% | 78.89% |

The single change replaced 1,000 matched knowledge presentations: 556 choice and 444 Boolean, using original-train Corr2Cause from unused graph families. Private data is `frontier-infra/jebadiah-data-v2-1-item47@2242e85fae6f8aff3fc3cea3fa95869d99132522`, folder `rung2/`. Both Studio scans passed with zero direct or remaining hits over 1,035,117 release records and 6,300 converted records. Training remains 21,190 presentations; aggregate Corr2Cause exposure is 1,063, below the 2,119 cap. All area/type counts are preserved and the 512-question A3 calibration is byte-identical. The causal diagnostic never enters training or temperature fitting.

Qwen/Qwen3.5-9B parent `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, rank 16, alpha 32, dropout 0.05, length 2,048, LR 1e-4, one epoch, seed 17, ordinal score targets, effective batch eight, item27 speed flags and autocast off match A3. Training code is `e79026a8aa46dfb131c1bb19ef70829a566f7901`; the pre-launch log was committed before submission. The final checkpoint proves step 2,649 and epoch 1.0. Training took 2,418.1 seconds with 64.81 GB peak allocated GPU memory and fp32 candidate logits. The trainer reports 36 truncated presentations, matching the earlier unchanged-A3-data D0/L5 runs; item49 reports zero truncations across all 9,450 new-source render checks.

The initial CUDA merge verifier passed the unchanged 0.05 gate: maximum probability shift 0.0159397721, zero confident flips, and 259/260 identical picks. No export recovery or additional optimizer steps were needed. A3 temperatures were copied byte-for-byte, SHA256 `48424adbce6d97faa73dc7a6201df939e4d0119d97b90c4ceadfd3e1578c741b`. `NOTICE-Corr2Cause.txt` is preserved in both private repositories and the merged export.

Private merged model revision is `frontier-infra/jebadiah-9b-v2-1-rung2@9ef5c8edd7d18a6ac57e0973bcea55dca0a99705`. Checkpoints stay in its private `-checkpoints` repository. All indexed merged shards were checked at that revision, and both live evaluations loaded it successfully. No model was made public or submitted. The inherited A3 source-policy exceptions remain.

HF Jobs used RTX PRO 6000. Initial timeout ceilings were 130 minutes training, 35 proxy, 45 diagnostic and 45 for one possible export recovery, totaling $11.6875 at $2.75/hour. The first diagnostic completed both model passes but failed while saving the tuple-keyed internal question map returned alongside timing. The wrapper now saves only timing metadata and persists each completed model pass. A CPU serialization check exercised the native tuple-keyed auxiliary shape. The corrected 35-minute retry completed and exactly reproduced both first-attempt overall metric tables. The failed allocation is included below; no capacity retry or export recovery occurred.

| Job | ID | Outcome | Running seconds | Estimated cost |
|---|---|---|---:|---:|
| training | [6ac97101fee2c9007017ede7](https://huggingface.co/jobs/jbrashear/6ac97101fee2c9007017ede7) | COMPLETED | 3213 | $2.4544 |
| proxy | [6ac97da8095c57808930b972](https://huggingface.co/jobs/jbrashear/6ac97da8095c57808930b972) | COMPLETED | 1122 | $0.8571 |
| diagnostic_attempt1 | [6ac97da8fee2c9007017f3d4](https://huggingface.co/jobs/jbrashear/6ac97da8fee2c9007017f3d4) | ERROR | 679 | $0.5187 |
| diagnostic | [6ac980d2095c57808930ba49](https://huggingface.co/jobs/jbrashear/6ac980d2095c57808930ba49) | COMPLETED | 689 | $0.5263 |

**Total estimated compute: $4.3565, including the failed diagnostic attempt, below the $12 cap.** Estimates use reported running seconds; final billing can differ. Training plus export took 53.55 minutes and proxy took 18.70, close to item44's measured 54.02 and 18.73 minutes. The initial end-to-end ETA was 90 to 120 minutes, including setup and diagnostics, with a possible recovery allowance.

Durable paired scoring and bootstrap artifacts are in the private results dataset under `analysis/item50-rung2/`. Candidate result input revision is `fd65a48aeb425a19e9b43a07a28a72bb1642d2b4`, A3 result input revision is `2cbd8b7394e0cb23c0af144220e623fc4f2298c2`, and diagnostic checkpoint revision is `928aadb8c12a0a8068f8c252b7ef5fa19aa789af`. [The committed receipt](../data/manifests/item50-results-receipt.json) records full hashes, all five areas, requested benchmarks, diagnostic strata, job states and costs. The source kit remains pinned at `587f4f496b9cf1d83f0bc8e495379496f65afa2f` and suite at `e57106b5e0698e74bd1a88b3b4c19b94a0dc8328`.

Studio validation passed manifest hashes, unchanged A3 recipe fields, calibration hash, area/type preservation, source cap, graph separation, diagnostic oracle/majority/serialization metric checks, script compilation, complete frozen ID/payload matching, official scoring and all 2,000 paired draws. No GitHub-hosted CI was used. PR 27 remains open for lead review and merge. Item50 scratch `/Volumes/PRO-G40/scratch/item50` was deleted after retaining durable evidence, including 179.4 MiB of dataset, suite, result, metadata, source-kit, environment/cache and temporary files. No model weights were downloaded to Studio; GPU job storage is ephemeral.
