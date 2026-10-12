# T2b: scrubbed A3 plus the T2b tranche

Desk #854 approved default A at 15:25 on 2026-10-12 with a USD 65 hard compute cap at USD 2.75/hour. T2b tests only the addition of the 256,965-question tranche to the scrubbed A3 baseline, using seeds 17 and 18. Keep if the paired complete-group bootstrap 95% interval of the two-seed mean difference against T2b-0 is entirely above zero. T2b-0's unrounded proxy mean is 44.115051, reported as 44.12. Report heldout family results against its 79.53% mean. Tools deltas within 6 points are seed noise; CLINC macro-F1 losses beyond 2.5 points are regressions.

## Private data and smoke

The scrubbed baseline is `frontier-infra/jebadiah-data-t2b`, folder `t2b0/`, revision `28f80a3c1430c27297e0944978bfda9310298623`. The tranche and heldout use revision `092e5a61556ee473339d41090b66da3aa530d7a0`. The composed folder `t2b/` is pinned at `bc416bc8485780d73aa4c22fb4cd772232faf69e` in the same private dataset.

Composition retains all 17,793 scrubbed A3 questions as a byte-identical prefix, then all 256,965 tranche questions in their existing order. Incoming IDs and the house normalized-state SHA256 are deduplicated in first-occurrence order. No additional row was removed. The final training set has 274,758 questions in 270,190 rows. The 356-question, 124-row scrubbed calibration is byte-identical. Calibration and heldout IDs, normalized states and family IDs are excluded from training.

The tranche lacked the `family_id` metadata required by the original pipeline. Composition adds `t2b-state:<house normalized-state SHA256>` without changing questions, labels, targets or source provenance. This is a schema adaptation, with no change to training content or recipe. The original pipeline's `rows`, `questions` and SHA256 manifest contract is verified on the full composed data.

The unchanged repository scanner checked all 270,314 training and calibration rows against the supplemented suite index SHA256 `b92290742245132a59839fa565ecc70d691d26033533cb48eb3da702aebd9851`. It found zero direct hits, rejected families or invalid empty states.

Before paid launch, the original A3 pipeline ran locally with `--setup-only`, the full real data, exact recipe arguments and a local base placeholder. It passed data and manifest validation and all 852 pinned prompt-identity renders in under 15 minutes. It did no optimizer steps, CUDA exercise or model loading. Paid smoke cost: USD 0. A local API signature mismatch was corrected before any network submission and cost USD 0.

The CPU evaluation integration smoke also compares T2b-0 with itself through the official scorer and paired two-seed mean path. It reproduces 44.115051 and zero score, CLINC and Tools differences. Its two-replicate interval is a software check only; final reported intervals use 2,000 draws.

## Runtime and budget

Reuse T2b-0's launch path and helper-code revision `8f5fd57d35ba51119fae8787d8875f666645f492`, running original A3 commit `86a8203d792463536eb009a4d8da6c8681ffdd99`. Base Qwen/Qwen3.5-9B stays pinned at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. LoRA rank 16, alpha 32, learning rate 0.0001, one epoch, microbatch 8, accumulation 1, 2,048-token limit, length grouping, objective, optimizer, schedule, calibration procedure and 0.05 merge gate are unchanged. The seed-18 patch changes only the original seed literal. Heldout uses the final unmerged adapter and original training evaluator, untempered candidate argmax, batch 8, 600 questions from each of five families; heldout never fits temperatures.

Each seed reserves 630 minutes for setup, training, calibration, merge, persistence and heldout. Each proxy reserves 40 minutes. These four ceilings total USD 61.4167, below USD 65. Even a hypothetical 15-minute paid smoke would make USD 62.1042. The actual local smoke uses no paid GPU. There are no automatic paid retries. The durable ledger counts every submitted job at its full allocation ceiling and reserves all remaining planned stages before admitting any submission.

Seed 17 training job: `6acd4366de32f9b0f43237aa`. Seed 18 training job: `6acd436cdd8177968b430f73`. Models and checkpoints are private under `frontier-infra/jebadiah-9b-t2b-s17` and `frontier-infra/jebadiah-9b-t2b-s18`, with separate `-checkpoints` repositories.

Results and measured costs are pending. This document will be completed after both proxies and paired bootstraps finish. The code and documentation PR will remain unmerged.
