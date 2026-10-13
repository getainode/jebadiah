# T2b: scrubbed A3 plus the T2b tranche

Desk #854 approved default A at 15:25 on 2026-10-12 with a USD 65 hard compute cap at USD 2.75/hour. T2b tests only the addition of the 256,965-question tranche to the scrubbed A3 baseline, using seeds 17 and 18. Keep if the paired complete-group bootstrap 95% interval of the two-seed mean difference against T2b-0 is entirely above zero. T2b-0's unrounded proxy mean is 44.115051, reported as 44.12. Report heldout family results against its 79.53% mean. Tools deltas within 6 points are seed noise; CLINC macro-F1 losses beyond 2.5 points are regressions.

## Private data and smoke

The scrubbed baseline is `frontier-infra/jebadiah-data-t2b`, folder `t2b0/`, revision `28f80a3c1430c27297e0944978bfda9310298623`. The tranche and heldout use revision `092e5a61556ee473339d41090b66da3aa530d7a0`. The composed folder `t2b/` is pinned at `bc416bc8485780d73aa4c22fb4cd772232faf69e` in the same private dataset.

Composition retains all 17,793 scrubbed A3 questions as a byte-identical prefix, then all 256,965 tranche questions in their existing order. Incoming IDs and the house normalized-state SHA256 are deduplicated in first-occurrence order. No additional row was removed. The final training set has 274,758 questions in 270,190 rows. The 356-question, 124-row scrubbed calibration is byte-identical. Calibration and heldout IDs, normalized states and family IDs are excluded from training.

The tranche lacked the `family_id` metadata required by the original pipeline. Composition adds `t2b-state:<house normalized-state SHA256>` without changing questions, labels, targets or source provenance. This is a schema adaptation, with no change to training content or recipe. The original pipeline's `rows`, `questions` and SHA256 manifest contract is verified on the full composed data.

The unchanged repository scanner checked all 270,314 training and calibration rows against the supplemented suite index SHA256 `b92290742245132a59839fa565ecc70d691d26033533cb48eb3da702aebd9851`. It found zero direct hits, rejected families or invalid empty states.

Before paid launch, the original A3 pipeline ran locally with `--setup-only`, the full real data, exact recipe arguments and a local base placeholder. It passed data and manifest validation and all 852 pinned prompt-identity renders in under 15 minutes. It did no optimizer steps, CUDA exercise or model loading. Paid smoke cost: USD 0. A local API signature mismatch was corrected before any network submission and cost USD 0.

The CPU evaluation integration smoke also compares T2b-0 with itself through the official scorer and paired two-seed mean path. It reproduces 44.115051 and zero score, CLINC and Tools differences. Its two-replicate interval is a software check only; final reported intervals use 2,000 draws.

A local IPv6 route interruption temporarily stopped monitoring. IPv4 connectivity remained available, so the local HF control client now binds IPv4 while preserving HTTPS verification. Monitoring retries read-only inspections after reporting an outage; it never submits a replacement paid job. The original cloud jobs and their deadlines remain unchanged.

## Runtime and budget

Reuse T2b-0's launch path and helper-code revision `8f5fd57d35ba51119fae8787d8875f666645f492`, running original A3 commit `86a8203d792463536eb009a4d8da6c8681ffdd99`. Base Qwen/Qwen3.5-9B stays pinned at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. LoRA rank 16, alpha 32, learning rate 0.0001, one epoch, microbatch 8, accumulation 1, 2,048-token limit, length grouping, objective, optimizer, schedule, calibration procedure and 0.05 merge gate are unchanged. The seed-18 patch changes only the original seed literal. Heldout uses the final unmerged adapter and original training evaluator, untempered candidate argmax, batch 8, 600 questions from each of five families; heldout never fits temperatures.

Each seed reserves 630 minutes for setup, training, calibration, merge, persistence and heldout. Each proxy reserves 40 minutes. These four ceilings total USD 61.4167, below USD 65. Even a hypothetical 15-minute paid smoke would make USD 62.1042. The actual local smoke uses no paid GPU. There are no automatic paid retries. The durable ledger counts every submitted job at its full allocation ceiling and reserves all remaining planned stages before admitting any submission.

Seed 17 training job: `6acd4366de32f9b0f43237aa`. Seed 18 training job: `6acd436cdd8177968b430f73`. Models and checkpoints are private under `frontier-infra/jebadiah-9b-t2b-s17` and `frontier-infra/jebadiah-9b-t2b-s18`, with separate `-checkpoints` repositories.

Both training/heldout jobs and both proxies completed. See verified results below. The code and documentation PR remains unmerged.


## Verified results

| Seed | Proxy | Difference vs matching T2b-0 | Paired 95% CI | Heldout family mean |
|---|---:|---:|---|---:|
| 17 | 36.190177 | -8.363430 | [-9.8376, -6.4140] | 79.37% |
| 18 | 36.943880 | -6.732616 | [-8.5300, -4.6379] | 79.90% |

Two-seed proxy mean **36.567028**, -7.552972 points against the stated 44.12 reference. Paired difference against the unrounded T2b-0 mean 44.115051: -7.548023, 95% CI [-8.7429, -5.9500]. Bootstrap decision: **drop**. CLINC mean regression flag: True. No model promotion or PR merge performed.

Both seed comparisons and the two-seed mean use 2,000 complete catalog/group bootstrap replicates with seed 20261008. All four runs use the same paired draws. These intervals measure uncertainty conditional on the frozen proxy, not training-seed variance. All 11,079 frozen proxy IDs per run are unique, successful and payload-bound to the verified suite. The unchanged official scorer is recomputed for every draw.

| Heldout family | Seed 17 | Seed 18 | Two-seed mean |
|---|---:|---:|---:|
| tasksource/SDOH-NLI | 84.67% | 87.50% | 86.08% | 92.75% | -6.67 |
| google-research-datasets/circa | 75.67% | 69.50% | 72.58% | 81.42% | -8.83 |
| michiel/xstance | 82.50% | 84.50% | 83.50% | 83.67% | -0.17 |
| google-research-datasets/poem_sentiment | 69.00% | 73.83% | 71.42% | 66.50% | +4.92 |
| zeroshot/twitter-financial-news-sentiment | 85.00% | 84.17% | 84.58% | 73.33% | +11.25 |

Heldout equal-family mean **79.63%**, +0.10 points against 79.53%. Each seed covers all 3,000 questions, with zero heldout truncation.

CLINC macro-F1 deltas: seed 17 -57.1298, seed 18 -63.0800, two-seed mean -60.1049 points. Losses beyond 2.5 count as regressions. Tools deltas: seed 17 -17.2528, seed 18 -8.6999, mean -12.9763; mean 95% CI [-16.3471, -9.8094]. Tools deltas within 6 are noise.

| Job | ID | Timeout minutes | Running seconds | Estimated USD |
|---|---|---:|---:|---:|
| train-s17 | `6acd4366de32f9b0f43237aa` | 630 | 17630.000 | 13.4674 |
| train-s18 | `6acd436cdd8177968b430f73` | 630 | 17694.000 | 13.5162 |
| proxy-s17 | `6acd88a79496b09289e61ea0` | 40 | 1124.000 | 0.8586 |
| proxy-s18 | `6acd88a89496b09289e61ea3` | 40 | 1124.000 | 0.8586 |

Total duration-based compute estimate **USD 28.7008**, including every paid attempt. Local smoke and pre-submission signature correction cost USD 0. No paid retry occurred. All HF jobs completed. This is a running-duration estimate at USD 2.75/hour, not an invoice.

Seed 17: 34345 optimizer steps, one epoch; original A3 recipe and prompt fields verified. Training truncations: 30. Original merge gate passed at maximum probability shift 0.012491, zero confident flips. Private model revision `673bdfc172ba2e85cee507af4433cc5b67fa4caf`.

Seed 18: 34345 optimizer steps, one epoch; original A3 recipe and prompt fields verified. Training truncations: 30. Original merge gate passed at maximum probability shift 0.012125, zero confident flips. Private model revision `f852c68739d0444e7dad1fc8f5ad20df89838129`.

Complete answers and outputs remain private. Evaluation analysis is persisted in `jbrashear/jebadiah-9b-v2-1-index-results`, folder `analysis/t2b-training`, revision `c9608b8927dd6639f236efedf2c1f78715773832`. Only code and documentation are in this PR.
